import cv2
import numpy as np
import time
import os

# =====================================================
# FIX WORKING DIRECTORY
# =====================================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# FACE + AGE + GENDER MODELS
# =====================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

age_net = cv2.dnn.readNet(
    "models/age_net.caffemodel",
    "models/age_deploy.prototxt"
)

gender_net = cv2.dnn.readNet(
    "models/gender_net.caffemodel",
    "models/gender_deploy.prototxt"
)

age_list = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60+']
gender_list = ['Male','Female']

# =====================================================
# LOG FUNCTION
# =====================================================
def log_person(person_type, age, gender):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("activity_log.txt", "a") as f:
        f.write(f"{timestamp}: {person_type} | {gender} | Age: {age}\n")

# =====================================================
# OBJECT DETECTION (MobileNet-SSD)
# =====================================================
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

min_confidence = 0.2
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
cap = cv2.VideoCapture(0)

# =====================================================
# PERFORMANCE METRICS
# =====================================================
start_time = time.time()
frame_count = 0

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    ret, image = cap.read()
    if not ret:
        break

    height, width = image.shape[:2]

    # Reset counters every frame
    men_count = 0
    women_count = 0
    kid_count = 0

    # ---------- OBJECT DETECTION ----------
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007,
        (300, 300),
        130
    )

    net.setInput(blob)
    detections = net.forward()

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    processing_time = round((elapsed_time / frame_count) * 1000, 2)

    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f"Processing Time: {processing_time} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    object_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            class_index = int(detections[0, 0, i, 1])
            class_name = classes[class_index]

            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            cv2.rectangle(image, (x1, y1), (x2, y2),
                          colors[class_index], 3)

            label = f"{class_name}: {confidence:.2f}%"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colors[class_index], 2)

            object_count += 1

            # ---------- PERSON → FACE → AGE/GENDER ----------
            if class_name == "person":
                person_roi = image[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray_roi, 1.3, 5)

                for (fx, fy, fw, fh) in faces:
                    face = person_roi[fy:fy+fh, fx:fx+fw]
                    if face.size == 0:
                        continue

                    blob = cv2.dnn.blobFromImage(
                        face, 1.0, (227, 227),
                        (78.426, 87.768, 114.895)
                    )

                    gender_net.setInput(blob)
                    gender = gender_list[gender_net.forward()[0].argmax()]

                    age_net.setInput(blob)
                    age = age_list[age_net.forward()[0].argmax()]

                    if age in ['0-2','4-6','8-12']:
                        person_type = "KID"
                        kid_count += 1
                    elif gender == "Male":
                        person_type = "MAN"
                        men_count += 1
                    else:
                        person_type = "WOMAN"
                        women_count += 1

                    cv2.rectangle(
                        image,
                        (x1+fx, y1+fy),
                        (x1+fx+fw, y1+fy+fh),
                        (255, 0, 0), 2
                    )

                    cv2.putText(
                        image,
                        person_type,
                        (x1+fx, y1+fy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

                    log_person(person_type, age, gender)

    # ---------- COUNTS ----------
    cv2.putText(image, f"Men: {men_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f"Women: {women_count}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f"Kids: {kid_count}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(image, f"Objects Detected: {object_count}", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Smart Indoor CCTV – Object + Demographics", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import os

# =====================================================
# FIX WORKING DIRECTORY
# =====================================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# FACE + AGE + GENDER MODELS
# =====================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

age_net = cv2.dnn.readNet(
    "models/age_net.caffemodel",
    "models/age_deploy.prototxt"
)

gender_net = cv2.dnn.readNet(
    "models/gender_net.caffemodel",
    "models/gender_deploy.prototxt"
)

age_list = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60+']
gender_list = ['Male','Female']

# =====================================================
# LOG FUNCTION
# =====================================================
def log_person(person_type, age, gender):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("activity_log.txt", "a") as f:
        f.write(f"{timestamp}: {person_type} | {gender} | Age: {age}\n")

# =====================================================
# OBJECT DETECTION (MobileNet-SSD)
# =====================================================
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

min_confidence = 0.2
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
cap = cv2.VideoCapture(0)

# =====================================================
# PERFORMANCE METRICS
# =====================================================
start_time = time.time()
frame_count = 0

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    ret, image = cap.read()
    if not ret:
        break

    height, width = image.shape[:2]

    # Reset counters every frame
    men_count = 0
    women_count = 0
    kid_count = 0

    # ---------- OBJECT DETECTION ----------
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007,
        (300, 300),
        130
    )

    net.setInput(blob)
    detections = net.forward()

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    processing_time = round((elapsed_time / frame_count) * 1000, 2)

    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f"Processing Time: {processing_time} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    object_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            class_index = int(detections[0, 0, i, 1])
            class_name = classes[class_index]

            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            cv2.rectangle(image, (x1, y1), (x2, y2),
                          colors[class_index], 3)

            label = f"{class_name}: {confidence:.2f}%"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colors[class_index], 2)

            object_count += 1

            # ---------- PERSON → FACE → AGE/GENDER ----------
            if class_name == "person":
                person_roi = image[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray_roi, 1.3, 5)

                for (fx, fy, fw, fh) in faces:
                    face = person_roi[fy:fy+fh, fx:fx+fw]
                    if face.size == 0:
                        continue

                    blob = cv2.dnn.blobFromImage(
                        face, 1.0, (227, 227),
                        (78.426, 87.768, 114.895)
                    )

                    gender_net.setInput(blob)
                    gender = gender_list[gender_net.forward()[0].argmax()]

                    age_net.setInput(blob)
                    age = age_list[age_net.forward()[0].argmax()]

                    if age in ['0-2','4-6','8-12']:
                        person_type = "KID"
                        kid_count += 1
                    elif gender == "Male":
                        person_type = "MAN"
                        men_count += 1
                    else:
                        person_type = "WOMAN"
                        women_count += 1

                    cv2.rectangle(
                        image,
                        (x1+fx, y1+fy),
                        (x1+fx+fw, y1+fy+fh),
                        (255, 0, 0), 2
                    )

                    cv2.putText(
                        image,
                        person_type,
                        (x1+fx, y1+fy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

                    log_person(person_type, age, gender)

    # ---------- COUNTS ----------
    cv2.putText(image, f"Men: {men_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f"Women: {women_count}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f"Kids: {kid_count}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(image, f"Objects Detected: {object_count}", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Smart Indoor CCTV – Object + Demographics", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
