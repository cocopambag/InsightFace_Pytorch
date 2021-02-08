from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import cv2
import numpy as np
import insightface
from PIL import Image


class Detector:
    def __init__(self, thresh=0.9, s=(112, 112)):
        self.net = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.net.prepare(ctx_id=0, nms=0.6)
        self.thresh = thresh
        self.s = s

    def detect(self, original, scale=1 / 2):
        origin = np.array(original)

        image = cv2.resize(origin, None, fx=scale, fy=scale)
        detections, landmark = self.net.detect(image, threshold=self.thresh, scale=1.0)

        # Those are Left Top Right Bottom.
        # Because cv2 is LeftTop is (0, 0). So, well think!
        list_bbox = []
        faces = []
        for i in range(len(detections)):
            if detections[i][-1] > self.thresh:
                bbox_ltrb = detections[i][:4] * (1 / scale)
                list_bbox.append(bbox_ltrb.astype(np.int))
                faces.append(self.make_face(origin, bbox_ltrb.astype(np.int)))

        return np.array(list_bbox), faces

    def make_face(self, img, bbox):
        (l, t, r, b) = bbox
        face = img[t:b+1, l:r+1 :]
        if 0 in face.shape:
            return
        face = cv2.resize(face, self.s)
        return Image.fromarray(face)
