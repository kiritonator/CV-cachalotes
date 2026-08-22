from ultralytics import YOLO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import json
from pathlib import Path
from datetime import datetime
from coordinates_calc import coordinates_calc
from csv_logger import LetterCsvLogger
import cv2
import time
import sys
import signal

MODEL_PATH = "/home/raspberrypiuser/best.pt"
SAVE_DIR = Path("/home/raspberrypiuser")
CONF = 0.2

SAVE_DIR.mkdir(parents=True, exist_ok=True)
csv_logger = LetterCsvLogger(
    csv_path=SAVE_DIR / "detected_letters.csv",
    different_letter_distance_m=25.0,
    min_confidence=0.20,
    track_timeout_s=2.0,
    min_frames=3,
)
    
model = YOLO(MODEL_PATH)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
)
picam2.configure(config)

video_dir = Path("/home/raspberrypiuser/videos")
video_dir.mkdir(parents=True, exist_ok=True)
video_path = video_dir / f"video_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"

encoder = H264Encoder(bitrate=8_000_000)
output = FfmpegOutput(str(video_path))

is_recording = True
def finish_recording(signum, frame):
    global is_recording

    if not is_recording:
        return

    is_recording = False

    print("Stopping: saving CSV and video ...", flush=True)

    csv_logger.close()

    try:
        picam2.stop_recording()
    except Exception as error:
        print(f"Video stop error: {error}", flush=True)

    print("Video and CSV saved", flush=True)

    raise SystemExit(0)
    
signal.signal(signal.SIGTERM, finish_recording)
signal.signal(signal.SIGINT, finish_recording)
picam2.start_recording(encoder, output)
print("Recording started", flush=True)

try:
    while True:
        #frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2GRAY)
        frame = picam2.capture_array()
        results = model.predict(
            source=frame,
            conf=CONF,
            verbose=False
        )

        best_cls_id = None
        best_conf = -1.0
        csv_logger.update()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                xp, yp, _, _ = box.xywhn[0].tolist()
                letter = model.names[cls_id]
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    latitude, longitude = coordinates_calc(xp, yp)
                except TypeError:
                    latitude, longitude = 0, 0
                csv_logger.add_detection(
                    letter=letter,
                    confidence=conf_score,
                    latitude=latitude,
                    longitude=longitude,
                )
                print(f"[{ts}] {letter} ({conf_score:.3f})", latitude, longitude, flush=True)

                with open(SAVE_DIR / "result.txt", "a", encoding="utf-8") as f:
                    f.write(f"{ts},{letter},{conf_score:.3f}, {latitude}, {longitude} \n")


        # cv2.imshow("Camera", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # if cv2.waitKey(1) & 0xFF == ord("q"):
            #break
            
        time.sleep(0.02)


except KeyboardInterrupt:
    pass
finally:
    csv_logger.close()
    picam2.stop_recording()
