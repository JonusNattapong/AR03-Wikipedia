import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# โหลดโมเดลและ Tokenizer ที่คุณ fine-tune
model_name_or_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

# สร้าง pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

# ฟังก์ชันสำหรับ Gradio
def predict(text):
    preds = classifier(text)[0]
    # เรียง label ตาม score สูงสุด
    preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
    top_label = preds_sorted[0]['label']
    top_score = preds_sorted[0]['score']
    return f"Topic: {top_label}\nConfidence: {top_score:.2f}"

# สร้าง Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="ใส่ข้อความบทความที่ต้องการจัดหมวดหมู่"),
    outputs=gr.Textbox(label="ผลการจำแนก"),
    title="Thai Topic Classifier",
    description="โมเดล fine-tuned จากบทความวิกิพีเดีย"
)

if __name__ == "__main__":
    demo.launch()
