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
                model_weights: Optional[Path] = Input(description="Zip file containing the model weights. The file will be unzipped and used for model initialization.", default=None),
                training_dataset: Path = Input(description="Zip file containing the training dataset"),
                epochs: int = Input(description="Number of training epochs", default=100),
                                ) -> ModelOutput:
        model_weights_path = "model_weights.zip"
        extracted_model_weights_dir = "extracted_model_weights"
        pretrain_weights_path = None
        if model_weights is not None:
            model_weights.rename(model_weights_path)
            with zipfile.ZipFile(model_weights_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_model_weights_dir)

            # Ensure the extracted directory contains exactly one file
            extracted_files = [f for f in os.listdir(extracted_model_weights_dir) if os.path.isfile(os.path.join(extracted_model_weights_dir, f))]
            if len(extracted_files) == 1:
                pretrain_weights_path = os.path.join(extracted_model_weights_dir, extracted_files[0])
            else:
                raise ValueError("The model_weights zip file must contain exactly one file.")

        training_dataset_path = "training_dataset.zip"
        training_dataset.rename(training_dataset_path)
        extracted_dataset_dir = "extracted_training_dataset"
        with zipfile.ZipFile(training_dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dataset_dir)

        if self.model is None or (model_weights is not None and self.model.weights_path != pretrain_weights_path):
            self.model = RFDETRLarge(resolution=896, pretrain_weights=pretrain_weights_path if pretrain_weights_path else None)

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
