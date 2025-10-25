from ultralytics import YOLO
import cv2
import numpy as np

class PersonDetector:
    """
    Simple wrapper around YOLOv8 for *person* detection.
    Uses COCO class 0 (person) and returns annotated frame + detections list.
    """
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.5):
        self.model = YOLO(model_name)  # will download weights on first run
        self.conf = float(conf)
        self.person_class = 0  # COCO class id for "person"

    def detect(self, frame: np.ndarray):
        """
        Args:
            frame: BGR image (OpenCV)
        Returns:
            annotated: frame with boxes
            detections: list of {'bbox': [x1,y1,x2,y2], 'conf': float}
        """
        results = self.model.predict(
            source=frame, conf=self.conf, classes=[self.person_class], verbose=False
        )
        annotated = frame.copy()
        detections = []

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                conf = float(box.conf.cpu().numpy()[0])
                detections.append({"bbox": xyxy.tolist(), "conf": conf})
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"person {conf:.2f}",
                    (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return annotated, detections
