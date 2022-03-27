
import cv2
import numpy as np
import random
from emotions import detect_emo
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

classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
modelConfiguration = "yolov3.cfg";
modelWeights = "yolov3.weights";
print("[INFO] loading model...")
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
fc = OpenvinoAgeGender( )
 
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
vs = cv2.VideoCapture('video.mp4')

frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
out = cv2.VideoWriter('po.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while True:
                    (grabbed, frame) = vs.read()
                    
                    if grabbed:
                          #  c+=1
                           # if c%2==0:
                            try:
                                blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
                                net.setInput(blob)
                                outs = net.forward(getOutputsNames(net))
                                classIds, confidences,rects =postprocess(frame,outs)
                                
                                import cv2
                                 

                                for i in range(len(rects)):
                                    bbox=rects[i]
                                    img=frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                                    img, ages, genders,emos = fc.inference(img)
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
                                out.write(frame)
                            except:
                                    pass
                             #   print(totalUpLeft,totalDownRight)
                            #cv2.imshow('test',cv2.resize(frame,(1280,720)))
                           # else:
                           #     cv2.imshow('test',cv2.resize(frame,(720,480)))
                            if cv2.waitKey(1) & 0xFF == ord('s'):
                                            break
                                         
                        
vs.release()
out.release()
    
# Closes all the frames
cv2.destroyAllWindows()