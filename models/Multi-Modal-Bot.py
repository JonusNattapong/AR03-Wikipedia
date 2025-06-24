!pip install transformers pillow

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# โหลดโมเดลภาพ
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# ถามโมเดล
image = Image.open('my_image.png')
inputs = processor(image, "describe this image?", return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
