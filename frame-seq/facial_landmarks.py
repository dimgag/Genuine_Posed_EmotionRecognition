import dlib
import cv2
import subprocess

# Run this "rm -rf `find -type d -name .ipynb_checkpoints`" in terminal to avoid .ipynb_checkpoints
# subprocess.run(["rm", "-rf", "`find", "-type", "d", "-name", ".ipynb_checkpoints`"])
# subprocess.run(["rm", "-rf", "`find", "-type", "d", "-name", ".DS_Store`"])


# Initialize the facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('frame-seq/shape_predictor_68_face_landmarks.dat')


# Load an image
img = cv2.imread('Image.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Loop over each face
for face in faces:
    # Detect facial landmarks
    landmarks = predictor(gray, face)

    # Loop over the 68 landmark points
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

# Display the image with facial landmarks
# cv2.imshow('Facial Landmarks', img)
cv2.imwrite('snowbaby.jpg', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()







