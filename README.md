# RF-DETR-COG

A serverless deployment of Roboflow's state-of-the-art object detection model (RF-DETR) on Replicate.

## Overview

This repository contains the code necessary to deploy the RF-DETR (Roboflow Detection Transformer) model as a serverless endpoint on Replicate. RF-DETR represents one of the most advanced object detection architectures available, combining the strengths of transformer-based models with efficient detection capabilities. Get more information about the model [here](https://blog.roboflow.com/rf-detr/).


## How to Use Custom Models

Customizing the deployment with your own RF-DETR model requires only two simple steps:

1.  Add your model weights file to the root of this repository
2.  Update the model initialization in the code:

```python
# Change this line in predict.py
self.model = RFDETRBase(pretrain_weights="your-custom-model.pth")
```

3.  Follow the [Replicate deployment guide](https://replicate.com/docs/guides/deploy-a-custom-model) to publish your model


## How to use with API:

Get more information about various available api from [here](https://replicate.com/hardikdava/rf-detr/api).

## Local Development and Testing

To test the model locally before deployment:

```bash
# Install cog if you haven't already
pip install cog

# Run a prediction with a local image
cog predict -i image=@/path/to/your/image.jpg

```

## Requirements

-   Python 3.8+
-   PyTorch 1.10+
-   Cog
- rfdetr

## Citation

If you use RF-DETR in your research or applications, please cite the original paper:

```
@software{rf-detr,
  author = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei},
  license = {Apache-2.0},
  title = {RF-DETR},
  howpublished = {\url{https://github.com/roboflow/rf-detr}},
  year = {2025},
  note = {SOTA Real-Time Object Detection Model}
}
```

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](https://github.com/roboflow/rf-detr/blob/main/LICENSE) file for details.
