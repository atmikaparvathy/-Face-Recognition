import face_recognition
import cv2
from google.colab.patches import cv2_imshow

# Load the known image and learn how to recognize it
known_image = face_recognition.load_image_file("known_image.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load the test image where you want to detect the face
test_image = face_recognition.load_image_file("test_image.jpg")

# Find all face locations and face encodings in the test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image)

# Convert the test image to BGR format for OpenCV
test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

# Loop through each face found in the test image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Compare the face encoding with the known encoding
    matches = face_recognition.compare_faces([known_encoding], face_encoding)

    # If a match is found, draw a rectangle around the face
    if matches[0]:
        cv2.rectangle(test_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(test_image_bgr, "Match Found", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.rectangle(test_image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(test_image_bgr, "No Match", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the result
cv2_imshow(test_image_bgr)
