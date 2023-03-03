import dlib
import cv2

# Initialize the facial landmark detector
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('/Users/dim__gag/git/Genuine_Posed_EmotionRecognition/frame-seq/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('/Users/dim__gag/git/Genuine_Posed_EmotionRecognition/frame-seq/shape_predictor_68_face_landmarks_GTX.dat')

# Load an image
img = cv2.imread('/Users/dim__gag/Desktop/0_0_S2N2H.MP4Pavel334.jpg')

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
cv2.imshow('Facial Landmarks', img)
cv2.waitKey(0)
cv2.destroyAllWindows()







