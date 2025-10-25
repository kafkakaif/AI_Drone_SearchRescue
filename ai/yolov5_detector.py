# ai/yolov5_detector.py
import torch
import cv2
import numpy as np

class YOLOv5PersonDetector:
    """
    Simple wrapper around YOLOv5 via torch.hub.
    Returns annotated frame (BGR) and list of detections [{'bbox':[x1,y1,x2,y2], 'conf':f}]
    """

    def __init__(self, model_name="yolov5s", device=None, conf_thres=0.45):
        # device: 'cpu' or 'cuda'
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # load model from torch.hub (ultralytics/yolov5)
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True).to(self.device)
        self.model.conf = conf_thres  # confidence threshold
        # filter only 'person' (COCO class 0)
        self.person_class = 0

    def detect(self, frame_bgr):
        """
        frame_bgr: OpenCV BGR image (H,W,3)
        returns: annotated_frame_bgr, detections_list
        """
        # convert BGR -> RGB for model
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # inference: returns a results object
        results = self.model(img_rgb, size=640)  # size= image size for inference speed/accuracy

        # results.xyxy[0] is a torch.Tensor of shape (n,6): x1,y1,x2,y2,conf,class
        detections = []
        if len(results.xyxy) > 0 and results.xyxy[0].shape[0] > 0:
            arr = results.xyxy[0].cpu().numpy()
            for x1, y1, x2, y2, conf, cls in arr:
                cls = int(cls)
                if cls == self.person_class:
                    detections.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "conf": float(conf)})

        # annotated frame: use results.render() which returns list of RGB images with boxes
        rendered = results.render()  # list of np arrays in RGB
        if len(rendered) > 0:
            annotated_rgb = rendered[0]
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        else:
            annotated_bgr = frame_bgr.copy()

        return annotated_bgr, detections
