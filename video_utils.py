import cv2
import os

def extract_frames(video_path: str, output_dir: str, interval_seconds: float = 1.0) -> list:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    current_frame = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            timestamp = current_frame / fps
            filename = f"frame_{current_frame}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_frames.append((filename, timestamp))

        current_frame += 1

    cap.release()
    return saved_frames
