from typing import Optional, List

from cog import BasePredictor, Path, Input, BaseModel
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


class ModelOutput(BaseModel):
    detections: List
    result_image: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = RFDETRBase()

    def predict(self,
                image: Path = Input(description="Input image for prediction"),
                confidence_threshold: float = Input(default=0.5, description="Confidence threshold for predictions"),
                ) -> Path:
        """Run a single prediction on the model"""
        image_pil = Image.open(image)
        detections = self.model.predict(image_pil, threshold=confidence_threshold)

        if len(detections) == 0:
            return ModelOutput(
                detections=[],
                result_image=image,
            )

        detections_labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        annotated_image = image_pil.copy()
        annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels=detections_labels)

        tmp_save_path = "tmp/annotated_image.jpg"
        annotated_image.save(tmp_save_path)

        detection_output = []
        for class_id, confidence, box in zip(detections.class_id, detections.confidence, detections.bbox):
            detection_output.append({
                "class_id": class_id,
                "confidence": confidence,
                "label": COCO_CLASSES[class_id],
                "bbox": box.tolist(),
            })
        return ModelOutput(
            detections=detection_output,
            result_image=Path(tmp_save_path),
        )
