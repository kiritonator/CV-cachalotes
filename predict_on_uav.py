from ultralytics import YOLO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from pathlib import Path
from datetime import datetime
import cv2
import time
import sys
import signal

MODEL_PATH = "/home/raspberrypiuser/best.pt"
SAVE_DIR = Path("/home/raspberrypiuser")
CONF = 0.2

SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
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
        
    if is_recording:
        is_recording = False
        print("Saving video ...", flush=True)
            
        picam2.stop_recording()
            
        print("video saved <3", flush=True)
    sys.exit(0)
    
signal.signal(signal.SIGTERM, finish_recording)
signal.signal(signal.SIGINT, finish_recording)
picam2.start_recording(encoder, output)
print("Recording started", flush=True)
while is_recording:
    time.sleep(1)


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

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])

                if conf_score > best_conf:
                    best_conf = conf_score
                    best_cls_id = cls_id

        if best_cls_id is not None:
            letter = model.names[best_cls_id]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"[{ts}] {letter} ({best_conf:.3f})", flush=True)

            with open(SAVE_DIR / "result.txt", "a", encoding="utf-8") as f:
                f.write(f"{ts},{letter},{best_conf:.3f}\n")
        
        # cv2.imshow("Camera", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # if cv2.waitKey(1) & 0xFF == ord("q"):
            #break
            
        time.sleep(0.2)
        
        


except KeyboardInterrupt:
    pass
finally:
    picam2.stop_recording()
    #cv2.destroyAllWindows()
