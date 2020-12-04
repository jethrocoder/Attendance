# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import pandas as pd
import numpy as np
from tkinter import messagebox
import os

def register(txt,txt2):
	#Init Camera

	cap = cv2.VideoCapture(0)

	# Face Detection
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

	skip = 0
	face_data = []
	dataset_path = './data/'
	name = txt2.get().upper()
	roll_no = txt.get().upper()
	# counter = 10
	while True:
		ret,frame = cap.read()

		if ret==False:
			continue

		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		frame = cv2.rotate(frame,cv2.ROTATE_180)

		faces = face_cascade.detectMultiScale(frame,1.3,5)
		if len(faces)==0:
			# print('your face is not visible \n please get into the frame')
			continue
			
		faces = sorted(faces,key=lambda f:f[2]*f[3])

		# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
		for face in faces[-1:]:
			x,y,w,h = face
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

			#Extract (Crop out the required face) : Region of Interest
			offset = 10
			face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
			face_section = cv2.resize(face_section,(100,100))

			skip += 1
			if skip%5==0:
				face_data.append(face_section)
				''' we can apply PCA here'''
				print(len(face_data))


		cv2.imshow("Frame",frame)
		cv2.imshow("Face Section",face_section)

		key_pressed = cv2.waitKey(1) & 0xFF
		if key_pressed == ord('q') or len(face_data) >= 10:
			break

	cap.release()
	cv2.destroyAllWindows()

	# Convert our face list array into a numpy array
	face_data = np.asarray(face_data)
	face_data = face_data.reshape((face_data.shape[0],-1))
	print(face_data.shape)

	# Save this data into file system
	np.save(dataset_path+roll_no+'.npy',face_data)
	print("Data Successfully save at "+dataset_path+roll_no+'.npy')

	# Registering student in csv file
	
	# if file does not exist write header
	row = np.array([roll_no,name]).reshape((1,2))
	df = pd.DataFrame(row) 
	if not os.path.isfile('student_details.csv'):
	   df.to_csv('student_details.csv', header=['roll','name'],index=False)
	else: # else it exists so append without writing the header
	   df.to_csv('student_details.csv', mode='a', header=False,index=False)
		
	
	messagebox.showinfo("Notification", "You have been registered successfully") 
