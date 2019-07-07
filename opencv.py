import cv2 , os
import numpy as np


detector = cv2.CascadeClassifier('face.xml') 
cam = cv2.VideoCapture(0)

Id = raw_input('enter your id') 
sampleNum = 0 

while(True): 
	
	ret,img = cam.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = detector.detectMultiScale(
		gray,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (35,35)
	)

	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		sampleNum = sampleNum + 1 
		cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
		 
		#cv2.waitKey(100) 
		cv2.imshow('frame',img) 

	if cv2.waitKey(100) & 0xFF == ord('q'): 
		break 
		#cv2.waitKey(1) if(sampleNum > 20): break 
	elif sampleNum>20:
		break
cam.release() 
cv2.destroyAllWindows()