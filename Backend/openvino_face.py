import os
import sys
import cv2
from openvino.inference_engine import IECore, IENetwork
import numpy as np
from typing import Optional, Tuple


class OpenvinoFace(object):
    def __init__(self,   thresh: float = 0.5):

 
        
        self.thresh = thresh
        self.model_initialize()

    def model_initialize(self):

       
        ie = IECore()
        net = ie.read_network(
           "mercury/Backend/face-detection-retail-0005.xml",
            "mercury/Backend/face-detection-retail-0005.bin",
        )
        self.input_name = next(iter(net.inputs))
        self.input_shape = net.inputs[self.input_name].shape
        self.out_name = next(iter(net.outputs))
        self.out_shape = net.outputs[self.out_name].shape
        self.exec_net = ie.load_network(net, "CPU")
         
    def detect(
        self, img: np.ndarray, thresh: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if thresh is None:
            thresh = self.thresh

        image = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0).astype(np.float32)

        out = self.exec_net.infer(inputs={self.input_name: image})[self.out_name]

        out = out[0, 0]

        to_del = []
        for i, res in enumerate(out):
            if res[2] <= thresh:
                to_del.append(i)
        out = np.delete(out, to_del, axis=0)
        classes = out[:, 1]
        scores = out[:, 2]
        boxes = out[:, 3:]

        boxes[:, 0] = (boxes[:, 0] * 300) * (img.shape[1] / 300)
        boxes[:, 1] = (boxes[:, 1] * 300) * (img.shape[0] / 300)
        boxes[:, 2] = (boxes[:, 2] * 300) * (img.shape[1] / 300)
        boxes[:, 3] = (boxes[:, 3] * 300) * (img.shape[0] / 300)

        return classes, scores, boxes.astype(np.int32)
