import os
import requests
import concurrent.futures
import pandas as pd
import numpy as np
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import time
from typing import List, Dict, Tuple
from schema import SCHEMA

load_dotenv()

# ===========================
# CONFIG
# ===========================
GPT_MODEL_NAME = "google/mt5-small"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABEL_CARDINALITY = 3

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Define missing WIKI_URL variable
WIKI_URL = "https://en.wikipedia.org/w/api.php"

# Initialize OpenAI client for DeepSeek API
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

OUTPUT_DIR = r"D:\Github\AR03-Wikipedia\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================
# FUNCTIONS
# ===========================
def fetch_pages(titles):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "titles": "|".join(titles),
    }
    r = requests.get(WIKI_URL, params=params)
    r.raise_for_status()
    return {
        page["title"]: page.get("extract", "")
        for page in r.json()["query"]["pages"].values()
    }


def batch_fetch(titles, batch_size=5):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = []
        for i in range(0, len(titles), batch_size):
            futures.append(ex.submit(fetch_pages, titles[i : i + batch_size]))
        for future in concurrent.futures.as_completed(futures):
            results.update(future.result())
    return results


def deepseek_chat(messages, model="deepseek-chat", stream=False):
    """Interact with DeepSeek API using OpenAI SDK."""
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, stream=stream
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error interacting with DeepSeek API: {e}")
        return None


# ----------------- DeepSeek API Client -----------------
class DeepseekClient:
    def __init__(
        self, api_key: str, model: str = "deepseek-chat", temperature: float = 1.0
    ):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 5

    def generate_dataset_with_prompt(self, task: dict, count: int) -> List[dict]:
        prompt = f"Generate {count} examples for task: {task['name']}\nSchema: {json.dumps(task['schema']['fields'], indent=2)}"
        req_body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": min(count * 200, 4000),
            "messages": [
                {"role": "system", "content": "You are a dataset generator."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.api_url, json=req_body, headers=headers, timeout=60
                )
                resp.raise_for_status()
                return json.loads(resp.json()["choices"][0]["message"]["content"])
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"[ERROR] Failed after {self.max_retries} attempts: {e}")
                time.sleep(self.retry_delay * (2**attempt))
        return []


# ----------------- Validation -----------------
def validate_data_quality(
    entries: List[Dict], task: Dict
) -> Tuple[List[Dict], List[str]]:
    valid_entries = []
    issues = []
    required_fields = task.get("schema", {}).get("fields", {})

    for i, entry in enumerate(entries):
        entry_issues = []
        for field_name, field_config in required_fields.items():
            # Fix: handle both dict and str field_config
            is_required = False
            if isinstance(field_config, dict):
                is_required = field_config.get("required", False)
            # If field_config is a string, assume not required (legacy/simple schema)
            if is_required and field_name not in entry:
                entry_issues.append(f"Missing required field: {field_name}")
        if not entry_issues:
            valid_entries.append(entry)
        else:
            issues.extend([f"Entry {i+1}: {issue}" for issue in entry_issues])

    return valid_entries, issues


# ----------------- Export -----------------
def export_to_jsonl(data: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def export_to_csv(data: List[Dict], file_path: str):
    import csv

    fieldnames = list(data[0].keys())
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# ===========================
# PIPELINE (ENHANCED)
# ===========================
# Added Gradio interface to allow users to select number of rows and export file format via a web interface.


# ----------------- Wikipedia API Integration -----------------
def fetch_wikipedia_data(query):
    """Fetch data from Wikipedia API based on the query."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "titles": query,
    }
    response = requests.get(WIKI_URL, params=params)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    return {page["title"]: page.get("extract", "") for page in pages.values()}


# ----------------- DeepSeek API Integration -----------------
def deepseek_generate_with_wikipedia(task, prompt, wiki_query, max_retries=3):
    """Interact with DeepSeek API using Wikipedia data as context and a strict prompt."""
    try:
        # Fetch Wikipedia data
        wiki_data = fetch_wikipedia_data(wiki_query)
        context = "\n".join(wiki_data.values())

        # Use the provided prompt (which should be strict for JSON output)
        payload = {
            "model": "deepseek-chat",
            "temperature": 1,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": "You are a dataset generator."},
                {"role": "user", "content": f"{prompt}\nContext: {context}"},
            ],
        }
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}

        for attempt in range(max_retries):
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                json=payload,
                headers=headers,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to generate data: {e}")
        return None


# ----------------- Gradio Interface -----------------
def gradio_interface():
    import gradio as gr

    # Dynamically generate tasks from the schema
    with gr.Blocks() as demo:
        gr.Markdown(
            """# Dataset Generator with Wikipedia Context
        Select a task, number of rows, export file format, and Wikipedia query to generate and export a dataset using DeepSeek API."""
        )

        api_key = gr.Textbox(
            label="DeepSeek API Key", placeholder="Enter your DeepSeek API key"
        )
        model_name = gr.Dropdown(
            label="Select Model",
            choices=["deepseek-chat", "deepseek-reasoner"],
            value="deepseek-chat",
        )

        all_tasks = get_all_tasks()
        task_name = gr.Dropdown(
            label="Select Task",
            choices=all_tasks,
            value=all_tasks[0] if all_tasks else None
        )
        rows = gr.Number(label="Number of Rows", value=10)
        file_format = gr.Radio(
            label="Export File Format", choices=["jsonl", "csv"], value="jsonl"
        )
        wiki_query = gr.Textbox(
            label="Wikipedia Query", placeholder="Enter Wikipedia query"
        )

        temperature_presets = {
            "Coding / Math": 0.0,
            "Data Cleaning / Data Analysis": 1.0,
            "General Conversation": 1.3,
            "Translation": 1.3,
            "Creative Writing / Poetry": 1.5,
        }

        temperature = gr.Dropdown(
            label="Select Temperature Preset",
            choices=list(temperature_presets.keys()),
            value="General Conversation",
        )

        generate_button = gr.Button("Generate and Export")
        test_button = gr.Button("Test API Key")
        output_message = gr.Textbox(label="Output Message")

        tasks_button = gr.Button("Show Available Tasks")
        tasks_output = gr.Textbox(label="Available Tasks")

        # --- Schema Editor UI ---
        schema_editor = gr.Code(
            label="Edit Task Schema (JSON)",
            value=json.dumps(
                SCHEMA["task_specific_fields"], ensure_ascii=False, indent=2
            ),
            language="json",
        )
        save_schema_btn = gr.Button("Save Schema")
        reload_schema_btn = gr.Button("Reload Schema")
        schema_status = gr.Textbox(label="Schema Status", interactive=False)

        # --- Single Task Schema Editor UI ---
        edit_task_dropdown = gr.Dropdown(
            label="Select Task to Edit",
            choices=list(SCHEMA["task_specific_fields"].keys()),
            value=list(SCHEMA["task_specific_fields"].keys())[0],
        )
        single_task_editor = gr.Code(
            label="Edit Selected Task Schema (JSON)",
            value=json.dumps(
                SCHEMA["task_specific_fields"][
                    list(SCHEMA["task_specific_fields"].keys())[0]
                ],
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )
        save_task_btn = gr.Button("Save Task Schema")
        task_schema_status = gr.Textbox(label="Task Schema Status", interactive=False)

        def save_schema(new_schema_json):
            try:
                parsed = json.loads(new_schema_json)
                with open("custom_schema.json", "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
                SCHEMA["task_specific_fields"] = parsed
                return "[INFO] Schema saved and loaded."
            except Exception as e:
                return f"[ERROR] Failed to save schema: {e}"

        def reload_schema():
            import os

            if os.path.exists("custom_schema.json"):
                try:
                    with open("custom_schema.json", "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    SCHEMA["task_specific_fields"] = loaded
                    return (
                        json.dumps(loaded, ensure_ascii=False, indent=2),
                        "[INFO] Schema reloaded.",
                    )
                except Exception as e:
                    return gr.update(), f"[ERROR] Failed to reload schema: {e}"
            else:
                return gr.update(), "[WARN] No custom schema file found."

        def update_task_editor(selected_task):
            return json.dumps(
                SCHEMA["task_specific_fields"][selected_task],
                ensure_ascii=False,
                indent=2,
            )

        def save_single_task_schema(selected_task, new_task_json):
            try:
                parsed = json.loads(new_task_json)
                SCHEMA["task_specific_fields"][selected_task] = parsed
                # Save the whole schema to file
                with open("custom_schema.json", "w", encoding="utf-8") as f:
                    json.dump(
                        SCHEMA["task_specific_fields"], f, ensure_ascii=False, indent=2
                    )
                return "[INFO] Task schema saved and loaded."
            except Exception as e:
                return f"[ERROR] Failed to save task schema: {e}"

        tasks_button.click(
            lambda: "\n".join(
                [f"{task}: {description}" for task, description in tasks.items()]
            ),
            inputs=[],
            outputs=tasks_output,
        )
        test_button.click(test_deepseek_api, [api_key], output_message)
        generate_button.click(
            lambda api_key, model_name, task_name, rows, file_format, wiki_query, temperature: generate_and_export_with_wikipedia(
                api_key, model_name, task_name, rows, file_format, wiki_query, temperature
            ),
            [api_key, model_name, task_name, rows, file_format, wiki_query, temperature],
            output_message,
        )
        save_schema_btn.click(save_schema, inputs=schema_editor, outputs=schema_status)
        reload_schema_btn.click(
            lambda: reload_schema(), inputs=None, outputs=[schema_editor, schema_status]
        )
        edit_task_dropdown.change(
            update_task_editor, inputs=edit_task_dropdown, outputs=single_task_editor
        )
        save_task_btn.click(
            save_single_task_schema,
            inputs=[edit_task_dropdown, single_task_editor],
            outputs=task_schema_status,
        )

    demo.launch()


def get_all_tasks():
    import os
    import json
    from schema import SCHEMA
    tasks = list(SCHEMA["task_specific_fields"].keys())
    if os.path.exists("custom_schema.json"):
        with open("custom_schema.json", "r", encoding="utf-8") as f:
            custom = json.load(f)
        tasks += [k for k in custom.keys() if k not in tasks]
    return tasks


def get_task_schema(task_name):
    import os
    import json
    from schema import SCHEMA
    if os.path.exists("custom_schema.json"):
        with open("custom_schema.json", "r", encoding="utf-8") as f:
            custom = json.load(f)
        if task_name in custom:
            return custom[task_name]
    return SCHEMA["task_specific_fields"].get(task_name, {})


def generate_and_export_with_wikipedia(
    api_key, model_name, task_name, rows, file_format, wiki_query, temperature
):
    """Generate and export dataset using DeepSeek API and Wikipedia context."""
    temperature_presets = {
        "Coding / Math": 0.0,
        "Data Cleaning / Data Analysis": 1.0,
        "General Conversation": 1.3,
        "Translation": 1.3,
        "Creative Writing / Poetry": 1.5,
    }
    temp_value = temperature_presets.get(temperature, 1.0)
    client = DeepseekClient(api_key, model=model_name, temperature=temp_value)
    task_schema = {
        "name": task_name,
        "schema": {"fields": get_task_schema(task_name)},
    }
    entries = []
    prompt = (
        f"Generate {rows} examples for task: {task_name} in valid JSON array format. "
        f"All data, text, and labels must be in Thai language only. "
        f"Only output the JSON array, no explanation, no markdown, no headings, no code block, no commentary."
    )
    for _ in range(int(rows)):
        output = deepseek_generate_with_wikipedia(task_schema, prompt, wiki_query)
        if output:
            try:
                entries.extend(json.loads(output))
            except Exception:
                import re

                match = re.search(r"(\[.*\])", output, re.DOTALL)
                if match:
                    try:
                        entries.extend(json.loads(match.group(1)))
                    except Exception as e:
                        return f"[ERROR] Failed to parse extracted JSON: {e}\nOutput: {output}"
                else:
                    return f"[ERROR] Failed to parse DeepSeek output as JSON.\nRaw Output:\n{output}"
    valid_entries, issues = validate_data_quality(entries, task_schema)
    if issues:
        return f"[WARN] Found issues: {issues}"
    output_file = os.path.join(
        OUTPUT_DIR, f"output_{task_name.lower().replace(' ', '_')}.{file_format}"
    )
    if file_format == "jsonl":
        export_to_jsonl(valid_entries, output_file)
    elif file_format == "csv":
        export_to_csv(valid_entries, output_file)
    return f"[INFO] Dataset for task '{task_name}' exported to {output_file}"


# ----------------- Test DeepSeek API Key -----------------
def test_deepseek_api(api_key):
    """Test the DeepSeek API key by making a simple request."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "deepseek-chat",
            "temperature": 1,
            "max_tokens": 10,
            "messages": [
                {"role": "system", "content": "Test API key."},
                {"role": "user", "content": "Hello DeepSeek."},
            ],
        }
        response = requests.post(
            "https://api.deepseek.com/chat/completions", json=payload, headers=headers
        )
        response.raise_for_status()
        return "[INFO] API key is valid."
    except Exception as e:
        return f"[ERROR] API key test failed: {e}"


def multi_turn_conversation(api_key):
    """Demonstrate multi-turn conversations using the DeepSeek API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Round 1
    messages = [
        {"role": "user", "content": "What's the highest mountain in the world?"}
    ]
    response = client.chat.completions.create(model="deepseek-chat", messages=messages)

    messages.append(response.choices[0].message)
    print(f"Messages Round 1: {messages}")

    # Round 2
    messages.append({"role": "user", "content": "What is the second?"})
    response = client.chat.completions.create(model="deepseek-chat", messages=messages)

    messages.append(response.choices[0].message)
    print(f"Messages Round 2: {messages}")


# Run Gradio interface
if __name__ == "__main__":
    gradio_interface()
