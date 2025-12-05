import tensorflow_hub as hub
import json

with open('config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def load_model(model_url = data['model_url']):
    print("Loading model from TensorFlow Hub...")
    try:
        detector = hub.load(model_url)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Try manual cache clear: rmdir /s %USERPROFILE%\.cache\tensorflow_hub")
        exit()
    return detector

def dectect(detector, input_tensor):
    print("Running RCNN object detection...")
    detector_output = detector(input_tensor)

    # Extract results (COCO format)
    boxes = detector_output['detection_boxes'][0].numpy()
    scores = detector_output['detection_scores'][0].numpy()
    classes = detector_output['detection_classes'][0].numpy().astype(int)
    
    return boxes, scores, classes