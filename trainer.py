import cv2
import os
import numpy as np
import pickle

# === CONFIGURATION ===
person_name = input("Enter name for this dataset (e.g., champ, friend1): ").strip()
capture_count = 20  # Number of images to capture
data_path = 'dataset'
person_path = os.path.join(data_path, person_name)

# === CREATE FOLDER IF NEEDED ===
os.makedirs(person_path, exist_ok=True)

# === INITIALIZE CAMERA AND CASCADE ===
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print(f"📸 Capturing {capture_count} images for '{person_name}'...")

count = 0
while count < capture_count:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (200, 200))  # ✅ Resize to fixed size
        img_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(img_path, roi_resized)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{count}/{capture_count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Saved {count} images to '{person_path}'")

# === TRAINING PHASE ===
print("🧠 Starting training...")

images, labels = [], []
label_map = {}
current_label = 0

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if not os.path.isdir(folder_path):
        continue

    label_map[current_label] = folder

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipped invalid image: {img_path}")
            continue

        resized_img = cv2.resize(img, (200, 200))  # ✅ Ensure consistent size
        images.append(resized_img)
        labels.append(current_label)

    print(f"✅ Loaded {len(os.listdir(folder_path))} images for '{folder}'")
    current_label += 1

# === FINAL CHECK ===
if len(images) < 2:
    raise ValueError("❌ Not enough training data. Capture at least 2 images total.")

labels = np.array(labels)  # ✅ Only labels need to be NumPy array

# === TRAIN AND SAVE ===
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
model.save("face_model.yml")

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Training complete. Model and label map saved.")
