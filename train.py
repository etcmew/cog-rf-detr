# -*- coding: utf-8 -*-

from typing import Optional, List, Any
import zipfile
import os
import shutil
import json
import warnings

from cog import Input, Path, BaseModel, BasePredictor
from rfdetr import RFDETRBase


class TrainingOutput(BaseModel):
    """
    Defines the output structure for the training process.
    It contains the path to the generated trained weights file.
    """

    trained_weights: Path


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
    ) -> Any:
        """
        Run a single prediction on the model.

        If `pretrained_weights` is provided, it loads a model with those specific
        weights for this prediction (less efficient). Otherwise, uses the default
        model loaded during setup.

        Args:
            image (Path): Path to the input image file.
            pretrained_weights (Optional[Path]): Path to specific weights to use for this prediction.
            confidence_threshold (float): Minimum confidence score for detected objects.

        Returns:
            Any: The prediction results from the RFDETRBase model.
        """
        print(f"Received prediction request for image: {image}")
        print(f"Confidence threshold: {confidence_threshold}")

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
                try:
                    model_to_use = RFDETRBase(pretrain_weights=pretrained_weights_path)
                    print("Temporary model loaded successfully with specific weights.")
                except Exception as e:
                    print(
                        f"Error loading model with specific weights ({pretrained_weights_path}): {e}"
                    )
                    raise
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

        try:
            print(f"Running model prediction using: {weights_source}")
            image_path_str = str(image)
            prediction_results = model_to_use.predict(
                image_path=image_path_str, confidence_threshold=confidence_threshold
            )
            print("Model prediction finished.")
            return prediction_results

        except AttributeError:
            print(
                f"Error: The selected model ({type(model_to_use)}) does not have a 'predict' method as expected."
            )
            raise
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
