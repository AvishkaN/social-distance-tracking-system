import cv2
import torch
import numpy as np
from ultralytics import YOLO, solutions

# Load the video
cap = cv2.VideoCapture('data/testvideo.mp4')

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# custom_model_path = 'data/PeopleDetector.pt'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_model_path, force_reload=True)

# Define desired classes
desired_classes = ['person']
names = model.names

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

def calculate_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5

def isClose(p1, p2):
    dist = calculate_distance(p1, p2)
    calibration = (p1[1] + p2[1]) / 2
    return 0 < dist < 0.25 * calibration

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model(frame)

    tracks = []
    for det in results.xyxy[0]:
        if int(det[5]) in [0]:  # 0 is the class index for 'person' in COCO dataset
            tracks.append(det)

    centerList = []
    statusList = [False] * len(tracks)

    for i, track in enumerate(tracks):
        bbox = track[:4].cpu().numpy()
        center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
        centerList.append(center)

    closePairsList = []

    for i in range(len(centerList)):
        for j in range(len(centerList)):
            if i != j and isClose(centerList[i], centerList[j]):
                closePairsList.append([centerList[i], centerList[j]])
                statusList[i] = True
                statusList[j] = True

    for i, track in enumerate(tracks):
        bbox = track[:4].cpu().numpy()
        if statusList[i]:
            color = (0, 0, 255)  # Red in BGR format
            print(f"Track {i} is close to another object - Coloring Red")  # Debug statement
        else:
            color = (0, 255, 0)  # Green in BGR format
            print(f"Track {i} is not close to another object - Coloring Green")  # Debug statement
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        print(f"Drawing rectangle at: ({int(bbox[0])}, {int(bbox[1])}), ({int(bbox[2]), int(bbox[3])}) with color: {color}")

    for pair in closePairsList:
        cv2.line(frame, tuple(pair[0]), tuple(pair[1]), (0, 0, 255), 2)
        print(f"Drawing line between: {tuple(pair[0])} and {tuple(pair[1])}")

    out.write(frame)
    
    # Display the frame
    cv2.imshow('Social Distance Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
