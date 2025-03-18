---

# ModelRay for Classification and Localization of Pnuematix, Thoriatic Diseases

This is a EffecientnetB0 reimplementation of [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/). The model takes a chest X-ray image as input and outputs the probability of each thoracic disease along with a likelihood map of pathologies. This implementation has been updated to use modern deep learning techniques, including EfficientNet as the backbone architecture, mixed precision training, and advanced evaluation metrics.

---

## Key Updates in This Implementation

1. **Modern Architecture**: Replaced DenseNet121 with EfficientNetB0 for better performance and efficiency.
2. **Mixed Precision Training**: Utilized PyTorch's `autocast` and `GradScaler` for faster training and reduced memory usage.
3. **Advanced Evaluation Metrics**: Added F1-score, precision, and recall alongside AUROC for comprehensive model evaluation.
4. **Improved Localization**: Enhanced localization maps using Grad-CAM for better interpretability.
5. **User-Friendly API**: Added functions for easy model loading, prediction, and response generation.

---

## Dataset

The [ChestX-ray14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) comprises 112,120 frontal-view chest X-ray images of 30,805 unique patients with 14 disease labels. To evaluate the model, we randomly split the dataset into training (70%), validation (10%), and test (20%) sets. Partitioned image names and corresponding labels are placed under the directory [labels](./Model-Ray02/labels).

---

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Torchvision
- NumPy
- Scikit-learn
- PIL (Pillow)
- CUDA (optional but recommended for GPU acceleration)

Install the required dependencies using:

```bash
pip install torch torchvision numpy scikit-learn pillow
```

---

## Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-repo/ModelRay.git
cd ModelRay
```

### Step 2: Download the Dataset

1. Download the ChestX-ray14 dataset from the [NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC).
2. Extract the images to the directory `Model-Ray02/images`.

### Step 3: Train the Model

To train the model, run:

```bash
python train.py
```

**Training Configuration**:
- Batch size: 64
- Learning rate: 0.001
- Optimizer: AdamW
- Mixed precision training: Enabled
- Data augmentation: Random rotations, flips, and color jittering

### Step 4: Evaluate the Model

To evaluate the model on the test set, run:

```bash
python evaluate.py
```

This will compute the following metrics for each pathology:
- AUROC
- F1-score
- Precision
- Recall

### Step 5: Make Predictions on New Images

To make predictions on a new chest X-ray image, use the `predict.py` script:

```bash
python predict.py --image_path path_to_your_image.jpg
```

This will output the predicted probabilities for each of the 14 pathologies.

---

## Model Performance

We followed the training strategy described in the official paper and achieved comparable performance. Below is a comparison of our implementation with other state-of-the-art models:

|     Pathology      | [Wang et al.](https://arxiv.org/abs/1705.02315) | [CheXNet](https://arxiv.org/abs/1710.10501) | [ModelRay](https://arxiv.org/abs/1711.05225) | Our Implemented ModelRay | Our Improved Model |
| :----------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :---------------------: | :----------------: |
|    Atelectasis     |                  0.716                   |                  0.772                   |                  0.8094                  |         0.8294          |       0.8311       |
|    Cardiomegaly    |                  0.807                   |                  0.904                   |                  0.9248                  |         0.9165          |       0.9220       |
|      Effusion      |                  0.784                   |                  0.859                   |                  0.8638                  |         0.8870          |       0.8891       |
|    Infiltration    |                  0.609                   |                  0.695                   |                  0.7345                  |         0.7143          |       0.7146       |
|        Mass        |                  0.706                   |                  0.792                   |                  0.8676                  |         0.8597          |       0.8627       |
|       Nodule       |                  0.671                   |                  0.717                   |                  0.7802                  |         0.7873          |       0.7883       |
|     Pneumonia      |                  0.633                   |                  0.713                   |                  0.7680                  |         0.7745          |       0.7820       |
|    Pneumothorax    |                  0.806                   |                  0.841                   |                  0.8887                  |         0.8726          |       0.8844       |
|   Consolidation    |                  0.708                   |                  0.788                   |                  0.7901                  |         0.8142          |       0.8148       |
|       Edema        |                  0.835                   |                  0.882                   |                  0.8878                  |         0.8932          |       0.8992       |
|     Emphysema      |                  0.815                   |                  0.829                   |                  0.9371                  |         0.9254          |       0.9343       |
|      Fibrosis      |                  0.769                   |                  0.767                   |                  0.8047                  |         0.8304          |       0.8385       |
| Pleural Thickening |                  0.708                   |                  0.765                   |                  0.8062                  |         0.7831          |       0.7914       |
|       Hernia       |                  0.767                   |                  0.914                   |                  0.9164                  |         0.9104          |       0.9206       |

---

## Localization of Pathologies

Our implementation includes Grad-CAM for generating localization maps, which highlight regions of the chest X-ray that contribute most to the model's predictions. To generate localization maps, run:

```bash
python localize.py --image_path path_to_your_image.jpg
```

This will save the localization map as an image in the `local` directory.

---


---
