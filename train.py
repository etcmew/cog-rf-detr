# -*- coding: utf-8 -*-

from typing import Optional, List, Any, Dict
import zipfile
import os
import shutil
import json
import warnings
import tempfile

from cog import Input, Path, BaseModel, BasePredictor
from rfdetr import RFDETRBase

try:
    import supervision as sv
    from PIL import Image

    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print(
        "Warning: 'supervision' or 'Pillow' library not found. Annotation features will be disabled."
    )


class TrainingOutput(BaseModel):
    """
    Defines the output structure for the training process.
    It contains the path to the generated trained weights file.
    """

    trained_weights: Path


class PredictionOutput(BaseModel):
    """
    Defines the output structure for the prediction process.
    Includes the annotated image and raw detection data.
    """

    annotated_image: Path
    detections_json: str


def train(
    training_dataset: Path = Input(
        description="Zip file containing the training dataset"
    ),
    pretrain_weights: Optional[Path] = Input(
        description="Optional path to pre-trained weights file (.pth) to start training from",
        default=None,
    ),
    epochs: int = Input(description="Number of training epochs", default=100, ge=1),
    batch_size: int = Input(description="Training batch size", default=4, ge=1),
    learning_rate: float = Input(
        description="Learning rate for the optimizer", default=1e-4, ge=0
    ),
    grad_accum_steps: int = Input(
        description="Gradient accumulation steps", default=16, ge=1
    ),
    device: str = Input(
        description="Device to train on ('cuda' or 'cpu')",
        default="cuda",
        choices=["cuda", "cpu"],
    ),
) -> TrainingOutput:
    """
    Main training function for the RF-DETR model using Cog.

    Args:
        training_dataset (Path): Path to the input zip file containing the dataset.
        pretrain_weights (Optional[Path]): Optional path to pre-trained model weights to initialize training.
        epochs (int): Number of epochs to train for.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate for the AdamW optimizer.
        grad_accum_steps (int): Number of steps to accumulate gradients over.
        device (str): The compute device ('cuda' or 'cpu').

    Returns:
        TrainingOutput: An object containing the path to the trained weights file.

    Raises:
        zipfile.BadZipFile: If the provided training_dataset is not a valid zip file.
        FileNotFoundError: If the expected output weights file is not found after training.
        Other Exceptions: Unexpected errors during file operations or model training will
                          cause the process to terminate.
    """
    print("Starting training process...")
    print(f" - Training Dataset: {training_dataset}")
    print(
        f" - Pre-trained Weights (for init): {pretrain_weights if pretrain_weights else 'None'}"
    )
    print(f" - Epochs: {epochs}")
    print(f" - Batch Size: {batch_size}")
    print(f" - Learning Rate: {learning_rate}")
    print(f" - Grad Accumulation Steps: {grad_accum_steps}")
    print(f" - Device: {device}")

    training_dataset_zip_path = "training_dataset.zip"
    extracted_dataset_dir = "extracted_training_dataset"

    if os.path.exists(extracted_dataset_dir):
        print(f"Removing existing directory: {extracted_dataset_dir}")
        shutil.rmtree(extracted_dataset_dir)

    print(
        f"Copying input dataset from {training_dataset} to {training_dataset_zip_path}"
    )
    shutil.copy(str(training_dataset), training_dataset_zip_path)

    print(f"Extracting {training_dataset_zip_path} to {extracted_dataset_dir}...")
    try:
        with zipfile.ZipFile(training_dataset_zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_dataset_dir)
        print("Dataset extracted successfully.")
    except zipfile.BadZipFile:
        print("Error: Invalid zip file provided.")
        raise

    pretrain_weights_path = str(pretrain_weights) if pretrain_weights else None

    print(
        f"Initializing RFDETRBase model for training with pretrain_weights: {pretrain_weights_path}..."
    )
    model = RFDETRBase(pretrain_weights=pretrain_weights_path)
    print("Model initialized for training.")

    print("Starting model training...")
    model.train(
        dataset_dir=extracted_dataset_dir,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=learning_rate,
    )
    print("Model training completed.")

    trained_weights_path = "output/checkpoint.pth"
    if not os.path.exists(trained_weights_path):
        print(
            f"Error: Expected output weights file not found at {trained_weights_path}"
        )
        output_dir = os.path.dirname(trained_weights_path)
        if os.path.exists(output_dir):
            print(f"Contents of {output_dir}: {os.listdir(output_dir)}")
        else:
            print(f"Output directory {output_dir} does not exist.")
        raise FileNotFoundError(
            f"Trained weights file not found: {trained_weights_path}"
        )

    print(f"Training finished. Weights saved to: {trained_weights_path}")

    print(f"Cleaning up extracted directory: {extracted_dataset_dir}")
    shutil.rmtree(extracted_dataset_dir)
    print(f"Cleaning up copied zip file: {training_dataset_zip_path}")
    os.remove(training_dataset_zip_path)

    return TrainingOutput(
        trained_weights=Path(trained_weights_path),
    )


DEFAULT_WEIGHTS_PATH = "output/checkpoint.pth"


class Predictor(BasePredictor):
    """
    Cog Predictor class for RF-DETR model.
    Loads trained weights (ideally specified by COG_WEIGHTS) during setup
    and runs inference. Allows overriding weights per prediction via input.
    Outputs annotated image and JSON detection data.
    """

    def setup(self):
        """
        Load the default model into memory.
        Uses COG_WEIGHTS environment variable if set for the default model,
        otherwise falls back to DEFAULT_WEIGHTS_PATH.
        """
        print("Setting up predictor...")
        weights_path_for_setup = os.environ.get("COG_WEIGHTS")

        if weights_path_for_setup and os.path.exists(weights_path_for_setup):
            print(f"Loading default model from COG_WEIGHTS: {weights_path_for_setup}")
        elif os.path.exists(DEFAULT_WEIGHTS_PATH):
            weights_path_for_setup = DEFAULT_WEIGHTS_PATH
            print(
                f"COG_WEIGHTS not set or invalid, loading default model from: {weights_path_for_setup}"
            )
        else:
            weights_path_for_setup = None
            print(
                "Warning: No weights specified via COG_WEIGHTS and default weights not found. Default model may be base."
            )

        self.default_model = RFDETRBase(pretrain_weights=weights_path_for_setup)
        print("Predictor setup complete with default model.")

    def predict(
        self,
        image: Path = Input(description="Image file to perform object detection on"),
        pretrained_weights: Optional[Path] = Input(
            description="Optional path to specific weights file (.pth) to use for *this prediction only*. Overrides default model.",
            default=None,
        ),
        confidence_threshold: float = Input(
            description="Confidence threshold for detections", default=0.5, ge=0, le=1.0
        ),
        num_classes: int = Input(
            description="Number of classes the model was trained for (e.g., 1 for 'navel')",
            default=1,
        ),
    ) -> PredictionOutput:
        """
        Run a single prediction on the model. Annotates the image and returns
        both the annotated image and the raw detection data as JSON.

        Args:
            image (Path): Path to the input image file.
            pretrained_weights (Optional[Path]): Path to specific weights to use for this prediction.
            confidence_threshold (float): Minimum confidence score for detected objects.
            num_classes (int): Number of classes expected by the model.

        Returns:
            PredictionOutput: An object containing the path to the annotated image
                              and a JSON string of the detections.
        """
        print(f"Received prediction request for image: {image}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Num classes: {num_classes}")

        if not SUPERVISION_AVAILABLE:
            raise ImportError(
                "Annotation failed: 'supervision' or 'Pillow' library not installed."
            )

        model_to_use = None
        weights_source = "default model loaded during setup"

        if pretrained_weights:
            weights_source = f"specific weights path provided: {pretrained_weights}"
            pretrained_weights_path = str(pretrained_weights)
            if os.path.exists(pretrained_weights_path):
                warnings.warn(
                    "Loading model weights per-prediction is inefficient. "
                    "Use COG_WEIGHTS during setup for better performance."
                )
                print(
                    f"Loading model with specific weights for this prediction: {pretrained_weights_path}"
                )
                # Allow exceptions during temporary model loading to crash
                model_to_use = RFDETRBase(
                    pretrain_weights=pretrained_weights_path, num_classes=num_classes
                )
                print("Temporary model loaded successfully with specific weights.")

            else:
                print(
                    f"Error: Specified pretrained_weights path does not exist: {pretrained_weights_path}"
                )
                raise FileNotFoundError(
                    f"Specified pretrained_weights not found: {pretrained_weights_path}"
                )
        else:
            if not hasattr(self, "default_model") or self.default_model is None:
                print("Error: Default model not loaded during setup.")
                raise RuntimeError("Default model failed to load during setup.")
            model_to_use = self.default_model

        # Allow exceptions during prediction/annotation to crash
        try:
            print(f"Running model prediction using: {weights_source}")
            image_path_str = str(image)
            pil_image = Image.open(image_path_str).convert("RGB")
            detections = model_to_use.predict(pil_image, threshold=confidence_threshold)
            
            print("Model prediction finished.")
            print(f"Raw detections object: {detections}")

            detections_list = []
            if detections and hasattr(detections, "xyxy") and len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    det_data = {
                        "box": detections.xyxy[i].tolist(),
                        "confidence": float(detections.confidence[i])
                        if hasattr(detections, "confidence")
                        else None,
                        "class_id": int(detections.class_id[i])
                        if hasattr(detections, "class_id")
                        else None,
                        "tracker_id": int(detections.tracker_id[i])
                        if hasattr(detections, "tracker_id")
                        and detections.tracker_id is not None
                        else None,
                    }
                    detections_list.append(
                        {k: v for k, v in det_data.items() if v is not None}
                    )

            detections_json = json.dumps(detections_list, indent=2)
            print(f"Detections JSON: {detections_json}")

            annotated_image = pil_image.copy()

            if detections_list:
                CLASS_NAMES = {i: f"class_{i}" for i in range(num_classes)}
                if num_classes == 1:
                    CLASS_NAMES = {0: "navel"}

                labels = [
                    f"{CLASS_NAMES.get(det['class_id'], f'id_{det["class_id"]}')} {det['confidence']:.2f}"
                    for det in detections_list
                    if "class_id" in det and "confidence" in det
                ]

                color = sv.ColorPalette.DEFAULT
                text_scale = sv.calculate_optimal_text_scale(
                    resolution_wh=pil_image.size
                )
                thickness = sv.calculate_optimal_line_thickness(
                    resolution_wh=pil_image.size
                )

                bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
                label_annotator = sv.LabelAnnotator(
                    color=color,
                    text_color=sv.Color.BLACK,
                    text_scale=text_scale,
                    text_thickness=thickness // 2,
                    text_padding=thickness * 2,
                    smart_position=True,
                )

                annotated_image = bbox_annotator.annotate(
                    annotated_image, detections=detections
                )
                annotated_image = label_annotator.annotate(
                    annotated_image, detections=detections, labels=labels
                )
                print("Image annotation applied.")
            else:
                print("No detections found meeting threshold, skipping annotation.")

            output_dir = tempfile.mkdtemp()
            annotated_image_path = os.path.join(output_dir, "annotated_output.png")
            if not isinstance(annotated_image, Image.Image):
                annotated_image = Image.fromarray(annotated_image)
            annotated_image.save(annotated_image_path)
            print(f"Annotated image saved to: {annotated_image_path}")

            return PredictionOutput(
                annotated_image=Path(annotated_image_path),
                detections_json=detections_json,
            )

        except AttributeError as e:
            print(
                f"Error: Attribute error during prediction/annotation. Check model output format and method calls: {e}"
            )
            raise
