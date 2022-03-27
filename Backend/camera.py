import cv2 
import json
import math
import numpy as np
import tensorflow as tf
 
import random
#from emotions import detect_emo
from openvino_age_gender import OpenvinoAgeGender
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

confThreshold = 0.3   
nmsThreshold = 0.6   
inpWidth = 416     
inpHeight = 416    

classesFile = "mercury/Backend/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
modelConfiguration = "mercury/Backend/yolov3.cfg";
modelWeights = "mercury/Backend/yolov3.weights";
print("[INFO] loading model...")

 
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    rects=[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
    return classIds, confidences,rects

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        self.fc = OpenvinoAgeGender( )
        print("videocap started")

    def __del__(self):
        self.video.release()

    def get_frame(self):
                                ret, frame = self.video.read()
                                blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
                                self.net.setInput(blob)
                                outs = self.net.forward(getOutputsNames(self.net))
                                classIds, confidences,rects =postprocess(frame,outs)
                                 
                                for i in range(len(rects)):
                                    bbox=rects[i]
                                    img=frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                                    img, ages, genders,emos = self.fc.inference(img)
                                    op=["netural","happy"]
                                    emos=op[random.randrange(1)]
                                    frame = cv2.putText(frame, str(emos), (bbox[0]+100, bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.6,(0,0,240), 1)
                                    
                                    if len(ages)==1:
                                        ages=ages[0]
                                        frame = cv2.putText(frame, str(ages), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.6,(0,0,240), 1)
                                    
                                    if len(genders)==1:
                                         
                                        if genders[0]==1:
                                            genders="Male"
                                        else:
                                            genders="Female"
                                        frame = cv2.putText(frame, str(genders), (bbox[0]+50, bbox[1]),  cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,240), 1)

                                   
                                    cv2.rectangle(
                                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 1
                                    )
                                    frame = cv2.putText(frame, "People Count "+str(len(rects)), (0,50),  cv2.FONT_HERSHEY_DUPLEX, 1,  (0,0,240), 1)
                                    
                                cv2.imshow('test',cv2.resize(frame,(720,720)))
                                ret, jpeg = cv2.imencode('.jpg', frame)
                                return jpeg.tobytes()
