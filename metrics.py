from ultralytics import YOLO

model = YOLO("runs/detect/bestmodel/weights/best.pt")


metrics = model.val(
    data="data.yaml",
    split="test"
)

print(metrics)