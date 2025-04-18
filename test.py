import torch
from ultralytics import YOLO
print(torch.__version__)  # Should show a CUDA-enabled version
print(torch.version.cuda)  # Should print 11.4
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print your GPU name

model = YOLO("yolo11n.pt")

results = model.train(data="coco8.yaml",epochs=100, imgsz=640, device=0)