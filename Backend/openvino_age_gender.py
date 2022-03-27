import os
import sys
import cv2
from openvino.inference_engine import IECore, IENetwork
import numpy as np
from typing import Optional, Tuple
#from emotions import detect_emo
from openvino_face import OpenvinoFace


class OpenvinoAgeGender(object):
    def __init__(self,   thresh: float = 0.2):
 
        self.thresh = thresh
        self.model_initialize()

    def model_initialize(self):

         
        ie = IECore()
        net = ie.read_network(
            "mercury/Backend/age.xml",
            "mercury/Backend/age.bin",
        )
        self.input_name = next(iter(net.inputs))
        self.input_shape = net.inputs[self.input_name].shape
        self.out_name = next(iter(net.outputs))
        self.out_shape = net.outputs[self.out_name].shape
        self.exec_net = ie.load_network(net, "CPU")
         
        # NOTE: draw is true. Hence no need to draw from our end
        self.facemodel = OpenvinoFace("openvino_face" )

    def inference(
        self, img: np.ndarray, thresh: Optional[float] = None
    ) -> Tuple[np.ndarray, list, list]:

        if thresh is None:
            thresh = self.thresh

        classes, scores, boxes = self.facemodel.detect(
            img, thresh=thresh
        )

        ages = []
        genders = []
        emos=[]
        for box in boxes:

            face = img[box[1] : box[3], box[0] : box[2]]
            x= " "
            if x:
                emos.append(x)
            face = cv2.resize(face, (self.input_shape[3], self.input_shape[2]))
            face = face.transpose(2, 0, 1)
            face = np.expand_dims(face, axis=0).astype(np.float32)

            age_gender = self.exec_net.infer(inputs={self.input_name: face})

            if 'prob' in list(age_gender.keys()):
                gender = list(age_gender['prob'].flatten())
                genders.append(gender.index(max(gender)))

            if 'age_conv3' in list(age_gender.keys()):
                age = int(age_gender[self.out_name] * 100)
                
                if 0 <= age <= 8:
                    age = "Kid"
                elif 9 <= age <= 18:
                    age = "Teen"
                elif 19 <= age <= 26:
                    age = "Youth"
                elif age > 26:
                    age = "Adult"
                
                ages.append(age)
        print(emos)
        return img, ages, genders,emos
