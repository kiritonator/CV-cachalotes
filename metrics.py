from ultralytics import YOLO

model = YOLO("runs/detect/bestmodel/weights/best.pt")


metrics = model.val(
    data="C:\\Users\Кирилл\\skat_cv\\dataset_(new_images_0)\\data.yaml",
    split="test"
)

print(metrics)