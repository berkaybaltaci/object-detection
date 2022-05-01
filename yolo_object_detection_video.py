import cv2
import numpy as np
import os

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#####################
cap = cv2.VideoCapture('london_driving.mp4')
if (cap.isOpened() == False):
    print('FILE NOT FOUND OR WRONG CODEC USED')

while (cap.isOpened()):
    ret, img = cap.read()

    if ret:
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (128, 128), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                # if (label != 'car'):
                #     continue
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x - 5, y - 25), (x + w, y + h), color, 2)
                cv2.putText(img=img, text=label, org=(x,y), fontFace=font, fontScale=2, color=(0,0,0), thickness=4)
                cv2.putText(img=img, text=label, org=(x,y), fontFace=font, fontScale=2, color=color, thickness=1)
                cv2.imshow('Obje algilama', img)

                # hit q to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    os._exit(1)