import csv
import cv2
import os
from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'../yolov8s-pose.pt', type=str, help='Path to model weight')
    parser.add_argument('--input_video', default=r'../inputs/pushup.mp4', type=str, help='Path to input video')
    parser.add_argument('--data_save_path', default=r'./data_without_resize/pushup/001.csv', type=str, help='Path to save data')
    parser.add_argument('--data_len', default=5, type=int, help='Sequence length')
    args = parser.parse_args()
    return args


def collect_data(args):
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.input_video)
    data = open(args.data_save_path, 'w', newline='')
    writer = csv.writer(data)
    data_row = []
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
            results = model(frame)
            ori_data = results[0].keypoints.data[0, :, 0:2]
            ori_data = ori_data.tolist()
            data_row.append(ori_data)
            if len(data_row) == args.data_len:
                writer.writerow(data_row)
                del data_row[0]

            frame = results[0].plot(boxes=False)

            cv2.imshow("YOLOv8 Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    data.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cfg = parse_args()
    collect_data(cfg)
