import cv2

print("Starting camera scan...")

for i in range(10):
    print(f"Trying camera index {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        print(f"✅ Camera index {i} OPENED successfully")
        cap.release()
    else:
        print(f"❌ Camera index {i} NOT opened")
