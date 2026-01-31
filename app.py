import streamlit as st
import cv2
import numpy as np
import time
import os

# ================= PATH FIX =================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ================= LOAD MODELS =================
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

net = cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)

age_list = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60+']
gender_list = ['Male','Female']

classes = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus","car",
    "cat","chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

np.random.seed(42)
colors = np.random.uniform(50,255,(len(classes),3))

# ================= STREAMLIT UI =================
st.set_page_config(
    page_title="Smart CCTV Dashboard",
    layout="wide",
)

st.markdown("""
<style>
body {background-color: #0e1117;}
.big-font {font-size:22px !important;}
.metric-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #161b22;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📹 Smart Indoor CCTV – AI Surveillance Dashboard")

col1, col2, col3, col4 = st.columns(4)
men_box = col1.empty()
women_box = col2.empty()
kid_box = col3.empty()
fps_box = col4.empty()

video_col, info_col = st.columns([3, 1])

with video_col:
    frame_window = st.image([])

with info_col:
    st.markdown("### 📊 Live Stats")
    men_box = st.empty()
    women_box = st.empty()
    kid_box = st.empty()
    fps_box = st.empty()

run = st.toggle("▶ Start Camera", value=False)

# ================= VIDEO =================
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

start_time = time.time()
frame_count = 0

# ================= MAIN LOOP =================
while run:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    men = women = kids = 0
    detected_objects_list = []

    # ---------- OBJECT DETECTION ----------
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300,300)),
        0.007, (300,300), 130
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.4:
            class_id = int(detections[0,0,i,1])
            class_name = classes[class_id]

            x1 = int(detections[0,0,i,3] * width)
            y1 = int(detections[0,0,i,4] * height)
            x2 = int(detections[0,0,i,5] * width)
            y2 = int(detections[0,0,i,6] * height)

            detected_objects_list.append(class_name)

            # Draw object box
            cv2.rectangle(
                frame, (x1,y1), (x2,y2),
                colors[class_id], 2
            )

            label = f"{class_name} ({confidence*100:.1f}%)"
            cv2.putText(
                frame, label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colors[class_id], 2
            )

            # ---------- FACE INSIDE PERSON ----------
            if class_name == "person":
                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (fx,fy,fw,fh) in faces:
                    face = roi[fy:fy+fh, fx:fx+fw]
                    if face.size == 0:
                        continue

                    blob_face = cv2.dnn.blobFromImage(
                        face, 1.0, (227,227),
                        (78.426,87.768,114.895)
                    )

                    gender_net.setInput(blob_face)
                    gender = gender_list[gender_net.forward()[0].argmax()]

                    age_net.setInput(blob_face)
                    age = age_list[age_net.forward()[0].argmax()]

                    if age in ['0-2','4-6','8-12']:
                        label_face = "KID"
                        kids += 1
                    elif gender == "Male":
                        label_face = "MAN"
                        men += 1
                    else:
                        label_face = "WOMAN"
                        women += 1

                    cv2.rectangle(
                        frame,
                        (x1+fx, y1+fy),
                        (x1+fx+fw, y1+fy+fh),
                        (255, 0, 0), 2
                    )

                    cv2.putText(
                        frame, label_face,
                        (x1+fx, y1+fy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0), 2
                    )

    # ---------- FPS ----------
    frame_count += 1
    fps = frame_count / (time.time() - start_time)

    # ---------- UI UPDATE ----------
    men_box.markdown(f"<div class='metric-box'>👨 MEN<br><b>{men}</b></div>", unsafe_allow_html=True)
    women_box.markdown(f"<div class='metric-box'>👩 WOMEN<br><b>{women}</b></div>", unsafe_allow_html=True)
    kid_box.markdown(f"<div class='metric-box'>🧒 KIDS<br><b>{kids}</b></div>", unsafe_allow_html=True)
    fps_box.markdown(f"<div class='metric-box'>⚡ FPS<br><b>{fps:.1f}</b></div>", unsafe_allow_html=True)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame, caption="Live CCTV Feed", use_column_width=True)

cap.release()
