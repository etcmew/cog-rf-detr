from typing import Optional
import zipfile
import os

from cog import BasePredictor, Path, Input, BaseModel
from rfdetr import RFDETRLarge

class ModelOutput(BaseModel):
    trained_weights: Path

class Predictor(BasePredictor):
    def setup(self):
        """Initialize the model variable."""
        self.model = None

    def predict(self,
                training_dataset: Path = Input(description="Zip file containing the training dataset"),
                epochs: int = Input(description="Number of training epochs", default=100),
                                ) -> ModelOutput:
        training_dataset_path = "training_dataset.zip"
        training_dataset.rename(training_dataset_path)
        extracted_dataset_dir = "extracted_training_dataset"
        with zipfile.ZipFile(training_dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dataset_dir)

        if self.model is None:
            self.model = RFDETRLarge(resolution=896, pretrain_weights=None)

        history = []

        def callback2(data):
            history.append(data)

        self.model.callbacks["on_fit_epoch_end"].append(callback2)

        self.model.train(
            dataset_dir=extracted_dataset_dir,
            epochs=epochs,
            device="cuda",
            batch_size=4,
            grad_accum_steps=16,
            lr=1e-4,
        )

        trained_weights_path = "output/checkpoint.pth"

        return ModelOutput(
            trained_weights=Path(trained_weights_path),
        )
