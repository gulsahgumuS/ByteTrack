import cv2
import numpy as np
import torch
<<<<<<< HEAD
from ultralytics import YOLO
from nets.nn import BYTETracker
=======
import warnings
from ultralytics import YOLO  # YOLOv8 modeli için
from nets import nn
>>>>>>> f6b17a5d26f4edb199709f4db0d1e6a3329338e6
from utils import util


<<<<<<< HEAD
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
=======
def draw_line(image, x1, y1, x2, y2, index):
    color = (0, 255, 0)  # Yeşil kutu
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f'ID {index}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    video_path = 'demo/output_40s.mp4'
    save_path = 'demo/output_result.mp4'

    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        print("⚠️ Video açılamadı.")
        return
    else:
        print("✅ Video başarıyla açıldı.")

    fps = reader.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25

    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # YOLOv8 modelini yükle
    model = YOLO('yolov8n.pt')  # Küçük model, istersen değiştirebilirsin

    # ByteTrack takipçisini başlat
    bytetrack = nn.BYTETracker(frame_rate=fps)

    frame_count = 0

    # Sadece bu sınıf id’leri takip edilecek: insan ve bazı araçlar
    valid_class_ids = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

    while True:
        success, frame = reader.read()
        if not success or frame is None:
            print("📛 Video bitti.")
            break
>>>>>>> f6b17a5d26f4edb199709f4db0d1e6a3329338e6

        frame_count += 1

        # YOLO ile nesne tespiti
        results = model(frame)[0]

        # Kutuları, skorları ve sınıfları numpy dizisi olarak al
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        # Sadece istediğimiz sınıfları filtrele
        mask = numpy.isin(class_ids, valid_class_ids)
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # ByteTrack güncelle
        outputs = bytetrack.update(boxes, confidences, class_ids)

        # Takip edilen nesneleri çiz
        for output in outputs:
            x1, y1, x2, y2, track_id = map(int, output[:5])
            draw_line(frame, x1, y1, x2, y2, track_id)

        writer.write(frame)
        print(f"✅ Frame {frame_count}/{total_frames} işlendi.")

    reader.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"🎥 Kayıt tamamlandı: {save_path}")

if __name__ == "__main__":
    main()
