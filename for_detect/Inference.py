import torch
import torch.nn as nn
import os
import cv2
import argparse
import json
from ultralytics import YOLO


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device=torch.device('cuda:0')):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        out = nn.functional.softmax(out, dim=1)
        return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pose', default=r'../yolov8s-pose.pt', type=str, help='Path to pose model weight')
    parser.add_argument('--checkpoint', default=r'./checkpoint/', type=str, help='Path to saved checkpoint')
    parser.add_argument('--device', default='cuda:0', type=str, help='Inference device')
    parser.add_argument('--input', default=r'C:\Users\90703\Desktop\produced.mp4', type=str, help='Path to input video')
    args = parser.parse_args()
    return args


def inference(args):

    yolo = YOLO(args.model_pose)
    device = torch.device(args.device)

    detect_model = LSTM(17*2, 8, 2, 3).to(device)
    model_weight = torch.load(os.path.join(args.checkpoint, 'best_model.pt'))
    detect_model.load_state_dict(model_weight)
    with open(os.path.join(args.checkpoint, 'idx_2_category.json'), 'r') as f:
        idx_2_category = json.load(f)
    print(idx_2_category)

    cap = cv2.VideoCapture(args.input)
    pose_key_point_frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Get pose key-points by YOLOv8
            # frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
            pose_results = yolo(frame)
            pose_data = pose_results[0].keypoints.data[0, :, 0:2]
            pose_key_point_frames.append(pose_data.tolist())
            if len(pose_key_point_frames) == 5:
                input_data = torch.tensor(pose_key_point_frames)
                input_data = input_data.reshape(5, 17 * 2)
                x_mean, x_std = torch.mean(input_data), torch.std(input_data)
                input_data = (input_data - x_mean) / x_std
                input_data = input_data.unsqueeze(dim=0)
                input_data = input_data.to(device)
                det_result = detect_model(input_data)
                print(det_result)
                print(idx_2_category[str(det_result.argmax().cpu().item())])
                del pose_key_point_frames[0]

            cv2.imshow("YOLOv8 Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cfg = parse_args()
    inference(cfg)

