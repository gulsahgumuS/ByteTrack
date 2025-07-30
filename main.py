import cv2
import numpy as np
from ultralytics import YOLO
from nets import nn

def draw_box_with_id(image, box, track_id):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"ID: {track_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    video_path = "demo/output_40s.mp4"
    output_path = "demo/output_result.mp4"

    # Video açılır
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"📛 Video açılamadı: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25

    print(f"🎥 VideoWriter oluşturuluyor: width={width}, height={height}, fps={int(fps)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))

    # YOLOv8 modelini yükle (kendi model yolunu yazabilirsin)
    model = YOLO("./weights/best.pt")

    # ByteTrack başlat
    tracker = nn.BYTETracker(frame_rate=fps)

    frame_count = 0
    valid_class_ids = [0]  # sadece plaka sınıfı (örnek: 0)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("📛 Video bitti veya frame alınamadı.")
            break

        frame_count += 1

        # Nesne algılama
        results = model(frame)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # Sadece plaka sınıfını filtrele
        mask = np.isin(classes, valid_class_ids)
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # ByteTrack güncelle
        tracked_objects = tracker.update(boxes, scores, classes)

        # Takip edilen plaka kutularını çiz
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            draw_box_with_id(frame, (x1, y1, x2, y2), track_id)

        out.write(frame)
        print(f"✅ Frame {frame_count} işlendi.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎥 Kayıt tamamlandı: {output_path}")

if __name__ == "__main__":
    main()
