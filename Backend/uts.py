import sys
import os
import warnings
import cv2
import numpy as np
import copy
import time
from typing import Optional, Tuple
 
from oureyeml.inference.oureye_services.lib.entry_exit.yolo_enex.utils.yolo import YOLO
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

confThreshold = 0.3   
nmsThreshold = 0.5   
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

        # Sets the input to the network

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
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

class YoloEnex():

 
    def __init__(
        self,
 
        vertical: bool = True,
        line_position: int = 540,
        gap_size: int = 30,
        skip_rate: int = 2,
        show_line: bool = True,
        show_bbox: bool = True,
        imgsz: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.5,
         
    ):
        self.device = "cpu"
        self.vertical = vertical
        self.line_position = line_position
        self.gap_size = gap_size
        self.skip_rate = skip_rate
        self.show_line = show_line
        self.show_bbox = show_bbox
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

       

        self.totalUpLeft = 0
        self.totalDownRight = 0
        self.totalFrames = 0
        self.totalInBetween = 0
        self.trackableObjects = {}
        self.trackers = []
        self.total_entry_count = 0
        self.ct = CentroidTracker()

        self.w = 0
        self.h = 0
        self.first_frame = True
        self.classes = None
        self.scores = None
        self.boxes = None

    def inference(
        self,
        frame: np.ndarray,
        line_position=None,
        gap_size=None,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        
        """
            returns: (Up/Left_count,Down/Right_count,frame,classes,scores,boxes, left, right)
            totalcount -> No. of person who have crossed the line till now
            frame -> output frame with detections

        """

        if line_position is None:
            line_position = self.line_position
        if gap_size is None:
            gap_size = self.gap_size
        if iou_thres is None:
            iou_thres = self.iou_thresh
        if conf_thres is None:
            conf_thres = self.conf_thresh

        self.line_position = int(line_position)
        total_entry_count = 0

        if self.first_frame:
            self.w = frame.shape[1]
            self.h = frame.shape[0]
            self.first_frame = False

        if self.totalFrames % self.skip_rate == 0:
            start1 = time.time()
            blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            self.classes, self.scores, self.boxes = postprocess(frame, outs)
            out = []
            for i, box in enumerate(self.boxes):
                out.append(
                    [
                        np.asarray([box[0], box[1], box[2], box[3]]).astype("int"),
                        self.classes[i],
                    ]
                )
            self.trackers = out
            end1 = time.time()
            fps_stat = 1 / (end1 - start1)
        objects = self.ct.update(self.trackers)

        if self.show_line:
            if self.vertical:
                cv2.line(
                    frame,
                    (self.line_position, 0),
                    (self.line_position, self.w),
                    (0, 255, 0),
                    3,
                )
                cv2.line(
                    frame,
                    (self.line_position - gap_size, 0),
                    (self.line_position - gap_size, self.w),
                    (0, 255, 0),
                    3,
                )
            else:
                cv2.line(
                    frame,
                    (0, self.line_position),
                    (self.w, self.line_position),
                    (0, 255, 0),
                    3,
                )
                cv2.line(
                    frame,
                    (0, self.line_position - gap_size),
                    (self.w, self.line_position - gap_size),
                    (0, 255, 0),
                    3,
                )
        left, right = 0, 0
        lb,rb=[],[]
        for (objectID, centroid) in objects.items():


            to = self.trackableObjects.get(objectID, None)

            cent = (centroid[0][0], centroid[0][1])

            bbox = centroid[1]
            if self.show_bbox:
                                    cv2.rectangle(
                                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                                    )


            if to is None:
                to = TrackableObject(objectID, cent)

            else:
                if self.vertical:
                    x = [c[0] for c in to.centroids]
                    direction = cent[0] - np.mean(x)
                    to.centroids.append(cent)
                else:
                    y = [c[1] for c in to.centroids]
                    direction = cent[1] - np.mean(y)
                    to.centroids.append(cent)

                if not to.counted:

                    val = cent[0] if self.vertical else cent[1]

                    if self.line_position - gap_size <= val <= self.line_position:

                        if direction > 0:  # > for Down/Right
                            roi = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                            if self.show_bbox:
                                    cv2.rectangle(
                                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                                    )
                            if np.shape(roi) != ():
                                to.counted = True
                                self.totalDownRight += 1
                                right += 1

                                rb.append(bbox)
                        if direction < 0:  # < for Up/Left
                            roi = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]

                            if np.shape(roi) != ():
                                to.counted = True
                                self.totalUpLeft += 1
                                left += 1
                        
                                lb.append(bbox)

                        if direction == 0:
                            roi = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]

                            if np.shape(roi) != ():
                                to.counted = True
                                self.totalInBetween += 1
                                total_entry_count += 1

            self.trackableObjects[objectID] = to

            cv2.circle(frame, (cent[0], cent[1]), 5, (255, 0, 0), -1)
        self.totalFrames += 1

        return (
            frame,
            self.totalUpLeft,
            self.totalDownRight,
            self.totalInBetween,
            total_entry_count,
            self.classes,
            lb,
            rb,
            left, 
            right,
        )


"""
    parameters:
    vertical -> True for vertical line else False (Default True)
    line_position ->d istance of line from origin (Default 540)
    skip_rate -> no. of frames to skip between detections (Default 2) # Don't go above 5!!
    show_line -> show the line (Default True)
    show_bbox -> show the bounding boxes (Default True)


    USAGE:

            import cv2
            from entry_exit import EntryExit

            vs = cv2.VideoCapture('test.mp4')
            er = EntryExit(vertical=False, line_position=540, skip_rate=2, show_line=True, show_bbox=False)
            while True:
                    (grabbed, frame) = vs.read()
                    if grabbed:
                            left,right,out_frame,_,_,_ = er.infer(frame)
                            print(right)
                            cv2.imshow('test',cv2.resize(out_frame,(720,480)))
                            if cv2.waitKey(25) == ord("q"):
                                    break
            cv2.destroyAllWindows()
            vs.release()

    """
