import cv2 
import numpy as np
from collections import deque
import time

cap = cv2.VideoCapture("traffic_video.mp4")

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

traffic_signal_states = ["RED", "GREEN"]
current_signal_state = "RED"
signal_timer = 11  

vehicle_queue = deque()

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    detected_objects = {"vehicles": [], "humans": [], "animals": []}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = classes[class_id]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                
                if label in ["car", "bus", "truck", "motorbike"]:
                    detected_objects["vehicles"].append((x, y, w, h))
                elif label == "person":
                    detected_objects["humans"].append((x, y, w, h))
                elif label in ["dog", "cat", "cow", "horse", "sheep"]:
                    detected_objects["animals"].append((x, y, w, h))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    return detected_objects, boxes, indexes, class_ids

def update_traffic_signal():
    global current_signal_state, signal_timer
    if len(vehicle_queue) > 10:  
        current_signal_state = "GREEN"
        signal_timer = 30  
    else:
        current_signal_state = "RED"
        signal_timer = 30  

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    detected_objects, boxes, indexes, class_ids = detect_objects(frame)

    if len(indexes) > 0:
        indexes = indexes.flatten()  

    for i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]

        
        color = (0, 255, 0)  
        if label == "person":
            color = (255, 0, 0)  
        elif label in ["dog", "cat", "cow", "horse", "sheep"]:
            color = (0, 165, 255)  

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        
        if label in ["car", "bus", "truck", "motorbike"] and y + h > frame.shape[0] * 0.8:
            vehicle_queue.append((x, y, w, h))

        
        if len(vehicle_queue) > 50:
            vehicle_queue.popleft()

    elapsed_time = time.time() - start_time
    if elapsed_time > signal_timer:
        update_traffic_signal()
        start_time = time.time()

    cv2.putText(frame, f"Signal: {current_signal_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    cv2.imshow("Smart Traffic Management System", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()