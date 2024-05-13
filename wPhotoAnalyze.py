import cv2
import numpy as np
from matplotlib import pyplot as plt
from deepface import DeepFace
import face_recognition

def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


image1_path = "C:/Users/Muhammed Kartal/Desktop/photos/deneme22.png"
image2_path = "C:/Users/Muhammed Kartal/Desktop/photos/deneme10.png"

model_name = "VGG-Face"
resp = DeepFace.verify(image1_path, image2_path,model_name,enforce_detection=False)

distance = resp["distance"]
confidence = 1 - distance

print(f"Distance: {distance}")
print(f"VGG - Face Confidence: {confidence:.2f}")

face_image = face_recognition.load_image_file("C:/Users/Muhammed Kartal/Desktop/photos/deneme10.png")
face_image_encoding = face_recognition.face_encodings(face_image)[0]

known_face_encodings = [
    face_image_encoding
]
known_face_names = [
    "berkan",
]

frame = cv2.imread("C:/Users/Muhammed Kartal/Desktop/photos/deneme22.png")

rgb_frame = frame[:, :, ::-1]

face_locations = face_recognition.face_locations(rgb_frame)
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

face_names = []
face_distances = face_recognition.face_distance(np.array(known_face_encodings), np.array(face_encodings))
best_match_index = np.argmin(face_distances)

for face_encoding, face_distance in zip(face_encodings, face_distances):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
    name = "Unknown"
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)
    confidence2 = 1 -face_distance
    print(f"Face-recognation Confidence: {confidence2:.2f}")
    results = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'],enforce_detection=False)
    first_result = results[0]
    age = first_result['age']
    gender = first_result['gender']
    nationality = first_result['dominant_race']
    emotion = first_result['emotion']

    print(f"Age: {age}")
    print(f"Gender: {gender}")
    print(f"Nationality: {nationality}")
    print(f"Emotion: {emotion}")

for (top, right, bottom, left), name, face_distance in zip(face_locations, face_names, face_distances):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 40), font, 0.8, (255, 255, 255), 1)
    confidence_text = f"Confidence: {confidence:.2f}"
    cv2.putText(frame, confidence_text, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

imshow('Face Recognition', frame)
