from ultralytics import YOLO
from pathlist import Path

model = YOLO('yolov8n.pt')
results = model.train(
    data="C:\\Users\Кирилл\\skat_cv\\dataset_(new_images_0)\\data.yaml",
    imgsz=640,
    epochs=50,
    batch=16,
    name='bestmodel'
)
run_dir = Path(results.save_dir)
best = run_dir / "weights" / "best.pt"



