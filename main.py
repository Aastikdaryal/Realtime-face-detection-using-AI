import cv2
import numpy as np
import time

# Define variables for performance metrics
start_time = time.time()
frame_count = 0

def log_activity(class_name, confidence):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("activity_log.txt", "a") as f:
        f.write(f"{timestamp}: {class_name} detected with confidence {confidence:.2f}%\n")

image_path = 'helmet2.png'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "helmet", "Helmet"]
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)

    net.setInput(blob)
    detected_objects = net.forward()

    # Calculate FPS and processing time
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    processing_time = round((elapsed_time / frame_count) * 1000, 2)  # in milliseconds

    # Display performance metrics on the output frame
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f"Processing Time: {processing_time} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Initialize object count
    object_count = 0

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]

        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])
            class_name = classes[class_index]

            log_activity(class_name, confidence)

            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f"{class_name}: {confidence:.2f}%"
            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
            cv2.putText(image, prediction_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)
            
            object_count += 1

    # Display object count
    cv2.putText(image, f"Objects Detected: {object_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Detected Objects", image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
cap.release()
