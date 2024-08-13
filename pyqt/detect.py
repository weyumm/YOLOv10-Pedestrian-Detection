from ultralytics import YOLOv10

# Load a pretrained YOLOv8n model
model = YOLOv10('yolov10n.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('images', save=True, imgsz=640, conf=0.25)
