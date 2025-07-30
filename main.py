import cv2
import numpy as np
import torch
from ultralytics import YOLO
from nets.nn import BYTETracker
from utils import util


def draw_box_with_id(image, box, track_id):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"ID: {track_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    video_path = "demo/output_40s.mp4"
    output_path = "demo/output_result.mp4"

    # Model yükle (Ultralytics YOLOv8 formatında)
    model = YOLO("./weights/best.pt")
    print(model.names)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"📛 Video açılamadı: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 VideoWriter oluşturuluyor: width={width}, height={height}, fps={int(fps)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))

    tracker = BYTETracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tahmin
        results = model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()           # (N,4)
        scores = results.boxes.conf.cpu().numpy()          # (N,)
        classes = results.boxes.cls.cpu().numpy().astype(int)  # (N,)

        # Sadece plaka sınıfı (örnek olarak sınıf 0)
        plate_indices = np.where(classes == 0)[0]
        if len(plate_indices) == 0:
            out.write(frame)
            continue

        boxes = boxes[plate_indices]
        scores = scores[plate_indices]
        classes_plate = classes[plate_indices]

        # Tracker update 3 parametre bekliyor: boxes, scores, classes
        track_bbs_ids = tracker.update(boxes, scores, classes_plate)

        for d in track_bbs_ids:
            cls = int(d[6])
            if cls == 0:  # sadece plaka
                x1, y1, x2, y2, track_id = map(int, d[:5])
                draw_box_with_id(frame, (x1, y1, x2, y2), track_id)


        out.write(frame)

    cap.release()
    out.release()
    print("✅ Tamamlandı. Kayıt dosyası:", output_path)


if __name__ == "__main__":
    main()
