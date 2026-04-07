from ultralytics import YOLO

model = YOLO("runs/detect/bestmodel/weights/best.pt")

results = model.train(
    data="new_dataset/data.yaml",
    epochs=15,
    imgsz=640,
    lr0=1e-3,
    pretrained=True,
    project="runs/detect",   # базовая папка
    name="trained_on_new_dataset"      # имя подпапки
)