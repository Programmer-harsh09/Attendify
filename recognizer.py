import cv2
import pickle

# === LOAD MODEL AND LABEL MAP ===
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

try:
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
except FileNotFoundError:
    print("❌ label_map.pkl not found. Run the training script first.")
    exit()

# === INITIALIZE CAMERA AND CASCADE ===
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("🎥 Starting face recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (200, 200))  # Match training size

        try:
            label, confidence = model.predict(roi_resized)
            name = label_map.get(label, "Unknown")

            if confidence < 70:
                text = f"{name} ({int(confidence)})"
                color = (0, 255, 0)  # Green for known
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Red for unknown

        except:
            text = "Recognition Error"
            color = (0, 0, 255)

        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
