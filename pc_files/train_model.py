from ultralytics import YOLO
from pathlist import Path

model = YOLO(r'best.pt')
results = model.train(
    data="\\dataset\\data.yaml",
    imgsz=1080,
    lr0=0.001,
    epochs=30,
    batch=16,
    name='real_model',
    device='cuda',
)
run_dir = Path(results.save_dir)
best = run_dir / "weights" / "best.pt"



