from typing import Optional
import zipfile

from cog import BasePredictor, Path, Input, BaseModel
from rfdetr import RFDETRLarge

class ModelOutput(BaseModel):
    trained_weights: Path

class Predictor(BasePredictor):
    def setup(self):
        """Initialize the model variable."""
        self.model = None

    def predict(self,
                model_weights: Optional[Path] = Input(description="Binary upload of the model weights file", default=None),
                training_dataset: Path = Input(description="Zip file containing the training dataset"),
                epochs: int = Input(description="Number of training epochs", default=100),
                                ) -> ModelOutput:
        model_weights_path = "model_weights.pth"
        if model_weights is not None:
            model_weights.rename(model_weights_path)

        training_dataset_path = "training_dataset.zip"
        training_dataset.rename(training_dataset_path)
        extracted_dataset_dir = "extracted_training_dataset"
        with zipfile.ZipFile(training_dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dataset_dir)

        if self.model is None or (model_weights is not None and self.model.weights_path != model_weights_path):
            self.model = RFDETRLarge(resolution=896, pretrain_weights=model_weights_path if model_weights is not None else None)

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
