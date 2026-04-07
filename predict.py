import cv2
from ultralytics import YOLO


model = YOLO("C:\\Users\Кирилл\skat_cv\\runs\detect\\runs\detect\\trained_on_new_dataset2\weights\\best.pt")
image_path = "C:\\Users\Кирилл\skat_cv\\new_dataset\\train\images\Снимок экрана 2026-04-07 213703_000_png.rf.3AqIUPAM6j0MIUim8URL.png"

dic = {'0': 1, '1': 10, '10' : 19, '11' : 2, '12' : 20, '13' : 21, '14' : 22, '15' : 23, '16' : 24, '17' : 25, '18' : 26, '19' : 27, '2' : 11, '20' : 28, '22' : 3, '23' : 30, '24' : 31, '25' : 32, '26' : 33, '27' : 34, '28' : 35, '29' : 36, '3' : 12, '30' : 4, '31' : 5, '32' : 6, '33' : 7, '34' : 8, '35' : 9, '4' : 13, '5' : 14, '6' : 15, '7' : 16, '8' : 17, '9' : 18}
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

print(dic[model.names[best_cls_id]])



