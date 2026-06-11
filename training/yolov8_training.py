from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="../data/yolov8_dataset/data.yaml", help="path to the yaml file")
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-b", "--batch", type=int, default=16)
parser.add_argument("-i", "--imgsz", type=int, default=640)
parser.add_argument("-m", "--model", type=str, default="yolov8n.pt", help="pretrained model to fine-tune")
args = parser.parse_args()

model = YOLO(args.model)
model.train(data=args.path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)