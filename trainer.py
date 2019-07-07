import cv2,os
import numpy as np
from PIL import Image 

path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.createLBPHFaceRecognizer()
cascadePath = path+"face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
dataPath = path+r'/dataSet'

def get_images_and_labels(datapath):
     image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
     # images will contains face images
     print image_paths

     faceSamples=[]

     Ids=[]

     for image_path in image_paths:
        if(os.path.split(image_path)[-1].split(".")[-1]!='jpg'):
            continue

        pilImage = Image.open(image_path).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage,'uint8')
        #getting the Id from the image
        Id = int(os.path.split(image_path)[-1].split(".")[1])
        # extract the face from the training image sample
        #faces = faceCascade.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        
        faceSamples.append(imageNp)
        Ids.append(Id)
        cv2.imshow("training",imageNp)
        cv2.waitKey(10)
     return Ids,faceSamples

Ids,faceSamples = get_images_and_labels(dataPath)
recognizer.train(faceSamples, np.array(Ids))
recognizer.save('trainer/trainer.yml')
cv2.destroyAllWindows()