import cv2
from ultralytics import YOLO


model = YOLO('runs/detect/bestmodel/weights/best.pt')
image_path = "C:\\Users\Кирилл\skat_cv\\test\images\card33c-598_png.rf.1f3ca4ae5a0820f9bf04ff784eb8eca5.jpg"
image = cv2.imread(image_path)
dict = {}

results = model.predict(
    source=image,
    conf=0.2
    )

best_cls_id = None
best_conf = -1.0

for r in results:
    boxes = r.boxes
    print("boxes:", len(boxes))
    for box in boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])

        if conf_score > best_conf:
            best_conf = conf_score
            best_cls_id = cls_id

print(model.names[best_cls_id])



