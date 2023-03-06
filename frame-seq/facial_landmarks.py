import dlib
import cv2
import os


def get_facial_landmarks(folder):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('frame-seq/shape_predictor_68_face_landmarks.dat')
	data_dir = folder
	images = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

	for image in images:
		img = cv2.imread(image)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector(gray)
        
		for face in faces:
			landmarks = predictor(gray, face)
            
			for n in range(0, 68):
				x = landmarks.part(n).x
				y = landmarks.part(n).y
				cv2.circle(img, (x,y), 2, (0, 255, 0), -1)

		cv2.imwrite(image, img)


def main():
	get_facial_landmarks('data_mtl_withfacialandmarks/train')
	get_facial_landmarks('data_mtl_withfacialandmarks/test')


if __name__ == '__main__':
	main()