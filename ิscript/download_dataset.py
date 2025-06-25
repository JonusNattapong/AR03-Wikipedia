## ตัวอย่างโค้ดโหลด Dataset จาก Hugging Face Hub

from datasets import load_dataset

# ตัวอย่างโหลด dataset ชื่อ "imdb" (review movie)
dataset = load_dataset("imdb")

print(dataset)
print("ตัวอย่างข้อมูลจาก train split:")
print(dataset['train'][0])


## โหลด Dataset ที่คุณอัปโหลดเอง (จาก repo บน Hugging Face)


from datasets import load_dataset

# แทนที่ด้วยชื่อ repo dataset ของคุณ เช่น username/dataset-name
dataset = load_dataset("your-username/your-dataset-repo")

print(dataset)
print(dataset['train'][0])


## โหลดไฟล์ JSONL หรือ CSV ที่อยู่บน Hugging Face Hub


from datasets import load_dataset

# โหลดไฟล์ JSONL จาก repo Dataset
dataset = load_dataset("json", data_files="https://huggingface.co/datasets/your-username/your-dataset/resolve/main/dataset.jsonl")

print(dataset)


## โหลด Dataset ที่บันทึกในไฟล์ JSONL หรือ CSV บนเครื่อง


from datasets import load_dataset

dataset = load_dataset("json", data_files="path/to/your/dataset.jsonl")

print(dataset)

from datasets import load_dataset

# กำหนดโฟลเดอร์สำหรับเก็บไฟล์ cache
cache_directory = "./your_dataset"

ds = load_dataset(
    "your_dataset_name/your_dataset_name",
    "en",
    cache_dir=cache_directory
)

# ลองดูขนาดชุดข้อมูล
print(ds)
