# Facial Emotion Recognition using CNN

A deep learning project that looks at a face image and predicts the emotion shown. It is built with a Convolutional Neural Network (CNN) in TensorFlow/Keras and trained on 48x48 grayscale face images.

**Emotions detected (7 classes):** `angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Demo: Sample Predictions](#demo-sample-predictions)
3. [Dataset](#dataset)
4. [How It Works](#how-it-works)
5. [Model Architecture](#model-architecture)
6. [Tech Stack](#tech-stack)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Results](#results)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)
13. [Author](#author)

---

## Project Overview

Humans read emotions from faces without thinking. This project teaches a computer to do the same.

The model takes a small grayscale photo of a face, passes it through several convolutional layers that learn visual patterns (eyes, mouth shape, eyebrows, wrinkles), and outputs a probability for each of the 7 emotions. The emotion with the highest probability is the prediction.

**Possible use cases:** customer feedback analysis, driver monitoring, mental-health tools, human-computer interaction, and smart classrooms.

---

## Demo: Sample Predictions

The trained model correctly classified these test images:

| Actual | Predicted | Result |
|--------|-----------|--------|
| surprise | surprise | Correct |
| happy | happy | Correct |
| disgust | disgust | Correct |

> Add your screenshots to an `assets/` folder and uncomment the lines below so they show on GitHub:
>
> <!-- ![Surprise prediction](assets/surprise.png) -->
> <!-- ![Happy prediction](assets/happy.png) -->
> <!-- ![Disgust prediction](assets/disgust.png) -->

---

## Dataset

- **Type:** Facial expression images, 48x48 pixels, grayscale (FER-2013 style dataset from Kaggle)
- **Classes:** 7 emotions
- **Training images:** 28,821
- **Test images:** 7,066

Expected folder layout (one sub-folder per emotion):

```
archive/
└── images/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        └── (same 7 folders)
```

The dataset is **not included** in this repository because of its size. Download it from Kaggle and extract it into the `archive/` folder.

---

## How It Works

The notebook follows these steps:

1. **Load data:** walk through the train/test folders and build a table of image paths and their emotion labels.
2. **Extract features:** open each image in grayscale and convert it into a 48x48 NumPy array.
3. **Normalize:** divide pixel values by 255 so they fall between 0 and 1. This makes training faster and more stable.
4. **Encode labels:** convert text labels (`happy`, `sad`, ...) into numbers with `LabelEncoder`, then into one-hot vectors with `to_categorical`.
5. **Build the CNN:** stack convolution, pooling, and dropout layers, followed by dense layers.
6. **Train:** fit the model on the training set while checking accuracy on the test set after every epoch.
7. **Predict:** load a single image, preprocess it the same way, and print the predicted emotion.

---

## Model Architecture

| Block | Layers |
|-------|--------|
| Conv Block 1 | Conv2D (128, 3x3, ReLU) → MaxPool (2x2) → Dropout 0.4 |
| Conv Block 2 | Conv2D (256, 3x3, ReLU) → MaxPool (2x2) → Dropout 0.4 |
| Conv Block 3 | Conv2D (512, 3x3, ReLU) → MaxPool (2x2) → Dropout 0.4 |
| Conv Block 4 | Conv2D (512, 3x3, ReLU) → MaxPool (2x2) → Dropout 0.4 |
| Classifier | Flatten → Dense 512 (ReLU) → Dropout 0.4 → Dense 256 (ReLU) → Dropout 0.3 |
| Output | Dense 7 (Softmax) |

- **Input shape:** (48, 48, 1)
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Metric:** Accuracy
- **Batch size:** 128
- **Epochs:** up to 100

**Why these choices?**
- *Conv + MaxPool* layers find patterns in the image and shrink it step by step.
- *Dropout* randomly switches off neurons during training so the model does not just memorize the training images (overfitting).
- *Softmax* at the end gives a probability for each of the 7 emotions that adds up to 1.

---

## Tech Stack

- Python
- TensorFlow / Keras (model building and training)
- NumPy, Pandas (data handling)
- scikit-learn (label encoding)
- Matplotlib (showing images)
- tqdm (progress bars)
- Jupyter Notebook

---

## Project Structure

```
.
├── face_emotion_recog.ipynb   # Full pipeline: data loading, training, prediction
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── assets/                    # Screenshots used in this README
└── archive/                   # Dataset (download separately)
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/suryaantarvedi/<your-repo-name>.git
cd <your-repo-name>
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the dataset** from Kaggle and extract it so the folders match the layout shown in the [Dataset](#dataset) section.

---

## Usage

**Train the model**

```bash
jupyter notebook
```

Open `face_emotion_recog.ipynb` and run the cells from top to bottom. Training on a CPU is slow (roughly 6 minutes per epoch on a laptop). A GPU or Google Colab speeds it up a lot.

**Predict on a single image**

```python
image = 'archive/images/train/happy/7.jpg'
img = ef(image)                       # preprocess the image
pred = model.predict(img)             # get probabilities
print(label[pred.argmax()])           # -> 'happy'
```

Where `label = ['angry','disgust','fear','happy','neutral','sad','surprise']`.

---

## Results

The model correctly identifies emotions on sample images (see [Demo](#demo-sample-predictions)).

| Metric | Value |
|--------|-------|
| Final training accuracy | _fill in after training_ |
| Final validation accuracy | _fill in after training_ |

Emotion recognition on this kind of data is hard: images are tiny, blurry, and some emotions (like fear vs. sad, or angry vs. disgust) look alike even to humans. The `disgust` class also has far fewer images than the others.

---

## Limitations

- Works only on **48x48 grayscale, face-cropped** images. Full photos need face detection first.
- **Class imbalance:** some emotions have many more images than others, which can bias predictions.
- Predicts one emotion per image; real expressions are often mixed.
- Not suitable for high-stakes decisions. Accuracy varies across lighting, angles, age, and skin tone.

---

## Future Improvements

- Add `BatchNormalization` and data augmentation (flip, rotate, zoom) to improve accuracy.
- Add early stopping and model checkpoints so training can be stopped and resumed safely.
- Save the trained model (`model.save('emotion_model.keras')`) so it can be reused without retraining.
- Try transfer learning (e.g., MobileNet, ResNet) for higher accuracy.
- Add a confusion matrix and per-class precision/recall.
- Real-time emotion detection from a webcam using OpenCV.
- Deploy as a web app (Streamlit or Flask).

---

## Author

**Surya**
B.Tech Computer Science, SRKR Engineering College
GitHub: [github.com/suryaantarvedi](https://github.com/suryaantarvedi)

If you found this project useful, consider giving it a star.
