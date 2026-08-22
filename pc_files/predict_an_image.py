import cv2
from ultralytics import YOLO


model = YOLO("best.pt")
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
        print(cls_id, conf_score)
        xp, yp, w, h = box.xywhn[0].tolist()

        print(
            f"conf={conf_score:.3f}",
            f"xp={xp:.6f}",
            f"yp={yp:.6f}",
        )



annotated_image = results[0].plot()
cv2.imwrite("prediction_result.png", annotated_image)



