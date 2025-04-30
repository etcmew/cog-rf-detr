# -*- coding: utf-8 -*-

from typing import Optional, List
import zipfile
import os
import shutil

from cog import Input, Path, BaseModel
from rfdetr import RFDETRBase

class TrainingOutput(BaseModel):
    """
    Defines the output structure for the training process.
    It contains the path to the generated trained weights file.
    """
    trained_weights: Path

def train(
    training_dataset: Path = Input(description="Zip file containing the training dataset"),
    pretrain_weights: Optional[Path] = Input(description="Optional path to pre-trained weights file (.pth)", default=None),
    epochs: int = Input(description="Number of training epochs", default=100, ge=1),
    batch_size: int = Input(description="Training batch size", default=4, ge=1),
    learning_rate: float = Input(description="Learning rate for the optimizer", default=1e-4, ge=0),
    grad_accum_steps: int = Input(description="Gradient accumulation steps", default=16, ge=1),
    device: str = Input(description="Device to train on ('cuda' or 'cpu')", default="cuda", choices=["cuda", "cpu"])
) -> TrainingOutput:
    """
    Main training function for the RF-DETR model using Cog.

    Args:
        training_dataset (Path): Path to the input zip file containing the dataset.
        pretrain_weights (Optional[Path]): Optional path to pre-trained model weights.
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
    print(f" - Pre-trained Weights: {pretrain_weights if pretrain_weights else 'None (training from scratch)'}")
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

    print(f"Copying input dataset from {training_dataset} to {training_dataset_zip_path}")
    shutil.copy(str(training_dataset), training_dataset_zip_path)

    print(f"Extracting {training_dataset_zip_path} to {extracted_dataset_dir}...")
    try:
        with zipfile.ZipFile(training_dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dataset_dir)
        print("Dataset extracted successfully.")
    except zipfile.BadZipFile:
        print("Error: Invalid zip file provided.")
        raise

    pretrain_weights_path = str(pretrain_weights) if pretrain_weights else None

    print(f"Initializing RFDETRBase model with pretrain_weights: {pretrain_weights_path}...")
    model = RFDETRBase(pretrain_weights=pretrain_weights_path)
    print("Model initialized.")

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
        print(f"Error: Expected output weights file not found at {trained_weights_path}")
        output_dir = os.path.dirname(trained_weights_path)
        if os.path.exists(output_dir):
             print(f"Contents of {output_dir}: {os.listdir(output_dir)}")
        else:
             print(f"Output directory {output_dir} does not exist.")
        raise FileNotFoundError(f"Trained weights file not found: {trained_weights_path}")

    print(f"Training finished. Weights saved to: {trained_weights_path}")

    print(f"Cleaning up extracted directory: {extracted_dataset_dir}")
    shutil.rmtree(extracted_dataset_dir)
    print(f"Cleaning up copied zip file: {training_dataset_zip_path}")
    os.remove(training_dataset_zip_path)

    return TrainingOutput(
        trained_weights=Path(trained_weights_path),
    )
