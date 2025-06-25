# schema.py
# This file contains the SCHEMA definition for NLP dataset/prompt generation tasks.

SCHEMA = {
    "task": "str",  # Task name (e.g., Text Classification, Summarization)
    "input": "str",  # Input text or data
    "output": "str",  # Expected output or generated result
    "metadata": {
        "model": "str",  # Model used for the task
        "parameters": "dict",  # Parameters used for the task
        "timestamp": "str",  # Timestamp of execution
        "confidence": "float",  # Confidence score of the result
    },
    "task_specific_fields": {
        "Thai Data Augmentation": {
            "id": "str",
            "original_text": "str",
            "augmented_text": "str",
        },
        "Thai Paraphrase Generation": {
            "id": "str",
            "original_text": "str",
            "paraphrased_text": "str",
        },
        "Thai Emotion Classification": {
            "id": "str",
            "text": "str",
            "emotion": "str",  # เช่น สุข, เศร้า, โกรธ, กลัว, ประหลาดใจ, รังเกียจ
        },
        "FAQ Matching": {
            "id": "str",
            "question": "str",
            "faq_candidates": "list[str]",
            "matched_faq": "str",
        },
        "Thai Grammar Correction": {
            "id": "str",
            "incorrect_text": "str",
            "corrected_text": "str",
        },
        "Thai Spelling Correction": {
            "id": "str",  # ข้อความที่สะกดผิด
            "incorrect_text": "str",
            "corrected_text": "str",  # ข้อความที่แก้ไขถูกต้อง
        },
        "Intent Detection": {
            "id": "str",
            "text": "str",  # ข้อความภาษาไทย
            "intent": "str",  # เช่น สั่งซื้อ, สอบถาม, ร้องเรียน, ทักทาย
        },
        "Thai NER": {
            "id": "str",
            "text": "str",
            "entities": "list[dict]",  # [{"entity": "PERSON", "value": "สมชาย"}, ...]
        },
        "Thai Question Generation": {
            "id": "str",
            "context": "str",  # ข้อความหรือย่อหน้า
            "question": "str",  # คำถามที่ generate
            "answer": "str",  # คำตอบที่เกี่ยวข้อง
        },
        "Thai Toxic Comment Classification": {
            "id": "str",
            "text": "str",
            "label": "str",  # toxic, non-toxic
        },
        "Thai Dialogue Generation": {
            "id": "str",
            "context": "str",  # บทสนทนาเดิม
            "response": "str",  # คำตอบที่ generate
        },
        # Unique tasks only, no duplicates between English and Thai
        "Text Classification": {
            "id": "str",  # รหัสเฉพาะสำหรับข้อความ
            "text": "str",  # ข้อความภาษาไทย
            "label": "str",  # หมวดหมู่ของข้อความ เช่น การเมือง, บันเทิง, การศึกษา, อื่นๆ
            # Removed duplicate 'labels' and 'confidence_scores' fields
        },
        "Token Classification": {
            "id": "str",
            "text": "str",
            "tokens": "list[dict]",  # รายการของ token และ label เช่น LOC, O
        },
        "Table Question Answering": {
            "id": "str",
            "table": "list[list[str]]",
            "question": "str",
            "answer": "str",
        },
        "Question Answering": {
            "id": "str",
            "context": "str",
            "question": "str",
            "answer": "str",
        },
        "Zero-Shot Classification": {
            "id": "str",
            "text": "str",
            "candidate_labels": "list[str]",
            "label": "str",
        },
        "Translation": {
            "id": "str",
            "source_text": "str",
            "source_lang": "str",
            "target_text": "str",
            "target_lang": "str",
        },
        "Summarization": {
            "id": "str",
            "document": "str",
            "summary": "str",
        },
        "Feature Extraction": {
            "id": "str",
            "text": "str",
            "embedding": "list[float]",
        },
        "Text Generation": {
            "id": "str",
            "prompt": "str",
            "generated_text": "str",
        },
        "Text2Text Generation": {
            "id": "str",
            "input_text": "str",
            "output_text": "str",
        },
        "Fill-Mask": {
            "id": "str",
            "sentence": "str",
            "options": "list[str]",
            "answer": "str",
        },
        "Sentence Similarity": {
            "id": "str",
            "sentence1": "str",
            "sentence2": "str",
            "similarity_score": "float",
        },
        "Table to Text": {
            "id": "str",
            "table": "list[list[str]]",
            "generated_text": "str",
        },
        "Multiple Choice": {
            "id": "str",
            "question": "str",
            "options": "list[str]",
            "answer": "str",
        },
        "Text Ranking": {
            "id": "str",
            "query": "str",
            "candidates": "list[dict]",
        },
        "Text Retrieval": {
            "id": "str",
            "query": "str",
            "retrieved_documents": "list[dict]",
        },
        "Thai Dialects Translation": {
            "source_dialect": "str",
            "source_text": "str",
            "target_language": "str",
            "target_text": "str",
            "topic": "str",
            "emotion": "str",
        },
        "Synthetic Persona": {
            "personaId": "str",
            "name": "str",
            "age": "int",
            "gender": "str",
            "background": "str",
            "goals": "str",
            "languageStyle": "str",
            "traits": "list[str]",
            "dialogueSamples": "list[dict]",
        },
        "ThaiSentimentIntentDataset": {
            "id": "str",
            "text": "str",
            "sentiment": "str",
            "intent": "str",
            "domain": "str",
        },
        "Reasoning": {
            "id": "str",
            "context": "str",  # ข้อความหรือสถานการณ์ที่ต้องใช้เหตุผล
            "question": "str",  # คำถามที่ต้องใช้เหตุผลตอบ
            "reasoning_type": "str",  # ประเภท reasoning เช่น multi-hop, commonsense, deductive, abductive
            "reasoning_steps": "list[str]",  # ลำดับขั้นตอนการให้เหตุผล
            "supporting_facts": "list[str]",  # หลักฐานหรือข้อความสนับสนุนแต่ละขั้น
            "chain_of_thought": "str",  # ข้อความอธิบายลำดับเหตุผลแบบต่อเนื่อง
            "evidence_span": "list[dict]",  # ตำแหน่ง span ของข้อความที่ใช้เป็นหลักฐาน เช่น [{"start": 0, "end": 10, "text": "..."}]
            "final_answer": "str",  # คำตอบสุดท้ายที่ได้จากการให้เหตุผล
        },
        "Fact Verification": {
            "id": "str",
            "claim": "str",  # ข้อความอ้างอิง/ข้อเท็จจริง
            "evidence": "list[str]",  # หลักฐานที่ใช้ตรวจสอบ
            "label": "str",  # ผลลัพธ์: SUPPORTED, REFUTED, NOT ENOUGH INFO
            "explanation": "str",  # คำอธิบายเหตุผล
        },
        "Natural Language Inference": {
            "id": "str",
            "premise": "str",  # ข้อความ premise
            "hypothesis": "str",  # ข้อความ hypothesis
            "label": "str",  # entailment, contradiction, neutral
            "explanation": "str",  # คำอธิบายเหตุผล
        },
        "Commonsense QA": {
            "id": "str",
            "question": "str",
            "choices": "list[str]",
            "answer": "str",
            "explanation": "str",
        },
        "Legal Document Classification": {
            "id": "str",
            "text": "str",  # ข้อความหรือเนื้อหาทางกฎหมาย
            "category": "str",  # หมวดหมู่ เช่น คำพิพากษา, สัญญา, กฎหมาย, คำร้อง
            "jurisdiction": "str",  # เขตอำนาจศาล/ประเทศ
            "explanation": "str"  # คำอธิบายเหตุผลการจัดหมวดหมู่
        },
        "Medical Report Summarization": {
            "id": "str",
            "report": "str",  # รายงานทางการแพทย์
            "summary": "str",  # สรุปใจความสำคัญ
            "diagnosis": "str",  # การวินิจฉัย
            "recommendation": "str"  # ข้อแนะนำทางการแพทย์
        },
        "Financial Sentiment Analysis": {
            "id": "str",
            "text": "str",  # ข่าว/บทวิเคราะห์/โพสต์เกี่ยวกับการเงิน
            "sentiment": "str",  # positive, negative, neutral
            "aspect": "str",  # ด้านที่เกี่ยวข้อง เช่น หุ้น, อสังหา, เศรษฐกิจ
            "confidence": "float"  # ความมั่นใจ
        },
        "Product Review Aspect Extraction": {
            "id": "str",
            "review": "str",  # ข้อความรีวิวสินค้า
            "aspects": "list[dict]",  # [{"aspect": "กล้อง", "sentiment": "positive"}, ...]
            "overall_sentiment": "str"  # positive, negative, neutral
        },
        "Academic Paper Title Generation": {
            "id": "str",
            "abstract": "str",  # บทคัดย่อ
            "title": "str"  # ชื่อเรื่องที่ generate
        },
        "Thinking": {
            "id": "str",
            "prompt": "str",  # คำถามหรือสถานการณ์ที่ต้องใช้การคิดวิเคราะห์
            "thought_process": "str",  # ข้อความอธิบายกระบวนการคิด
            "steps": "list[str]",  # ลำดับขั้นตอนการคิด
            "conclusion": "str"  # ข้อสรุปสุดท้าย
        },
    },
}
