"""
Test a YOLOv8 model live on your webcam.

Install:
    pip install ultralytics opencv-python

Run:
    python test_yolov8_camera.py --weights best.pt
    python test_yolov8_camera.py --weights best.pt --camera 1 --conf 0.4

Controls:
    q  quit
    s  save a screenshot of the current frame
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run a YOLOv8 model on a live camera feed.")
    parser.add_argument("--weights", default="/Users/lefanfang/Documents/GitHub/FaceRecognition/models/yolov8_face-2/weights/best.pt", help="Path to your trained .pt file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 = default webcam)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default=None, help="'cpu', '0' for GPU 0, etc. Default: auto")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    print(f"Classes: {model.names}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Try --camera 1, or check that no other app is using it."
        )

    print("Camera opened. Press 'q' to quit, 's' to save a screenshot.")
    prev_time = time.time()
    fps = 0.0
    shot_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            annotated = results[0].plot()  # draws boxes, labels, and scores

            # Smoothed FPS
            now = time.time()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            prev_time = now

            num_dets = len(results[0].boxes)
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}  Detections: {num_dets}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLOv8 Camera Test", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                filename = f"screenshot_{shot_count:03d}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"Saved {filename}")
                shot_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
