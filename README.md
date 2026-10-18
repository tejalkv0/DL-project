# Deep Learning — Image Classification (CIFAR-10 subset, ResNet-18 transfer learning)

## Files
| File | Purpose |
|---|---|
| `train_image_classifier.ipynb` | Training notebook: data, augmentation, transfer learning, curves, metrics, model saving |
| `cifar10_resnet18.pth` | Saved model (created when you run the notebook) |
| `predict.py` | Command-line inference script |
| `requirements.txt` | Dependencies |

Notebook outputs: `training_curves.png`, `confusion_matrix.png`, `sample_predictions.png`, `augmentation_preview.png`, `metrics.json`.

## Train (Google Colab)
1. Upload `train_image_classifier.ipynb` to Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all (about 5–10 minutes). CIFAR-10 downloads automatically.
4. Download `cifar10_resnet18.pth` from the Colab file panel (left sidebar → Files).

## Run inference
```bash
pip install -r requirements.txt
python predict.py path/to/image.jpg
```
Example output:
```
dog.jpg
  dog           91.3%
  cat            6.2%
  deer           1.1%
```
Or from Python:
```python
from predict import load_model
model, ckpt = load_model("cifar10_resnet18.pth")
```
Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Works best on simple, centred photos of these objects.

## Method
- **Model:** ImageNet-pretrained ResNet-18, new head (Dropout 0.3 + Linear 512→10).
- **Transfer learning:** Stage 1 trains only the head (4 epochs); Stage 2 also fine-tunes `layer4` at a lower LR (6 epochs).
- **Augmentation:** random crop, horizontal flip, rotation ±10°, colour jitter.
- **Data:** stratified subset, 6,000 train / 1,000 val / 2,000 test, images resized to 128×128.
- **Metrics:** accuracy, per-class precision/recall/F1, confusion matrix, loss/accuracy curves.
