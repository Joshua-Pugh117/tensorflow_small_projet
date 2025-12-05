# my_data.py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import json

with open('config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def prepare_image(image_path, input_size=640):
    print(f"Loading and resizing image to {input_size}x{input_size}...")
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((input_size, input_size), Image.LANCZOS)
    image_np = np.array(image_resized)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
    return image, input_tensor

# draw_on_image stays exactly as before
def draw_on_image(image, boxes, scores, classes, class_names=data['class_names'], threshold=0.5):
    draw = ImageDraw.Draw(image)
    orig_width, orig_height = image.size

    print(f"Detections above {threshold*100:.0f}%:")
    for i in range(len(scores)):
        if scores[i] < threshold:
            break
        box = boxes[i]
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * orig_width)
        xmax = int(xmax * orig_width)
        ymin = int(ymin * orig_height)
        ymax = int(ymax * orig_height)

        class_name = class_names[int(classes[i] - 1)] if int(classes[i] - 1) < len(class_names) else "unknown"
        score = scores[i]
        print(f"  {class_name}: {score:.2%}")

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        label = f"{class_name} {score:.2%}"
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.rectangle([xmin, ymin - h - 5, xmin + w + 8, ymin], fill="red")
        draw.text((xmin + 4, ymin - h - 3), label, fill="white", font=font)

    return image