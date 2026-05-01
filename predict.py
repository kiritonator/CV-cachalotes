import cv2
from ultralytics import YOLO


model = YOLO("C:\\Users\\Кирилл\\skat_cv\\runs\\detect\\bestmodel\\weights\\best.pt")
image_path = input("path: ")

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



