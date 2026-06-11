# 🦴 Bone Fracture Detection — Fine-Tuned VGG19 on X-Ray Images

> A deep learning project that classifies bone X-rays as **Fractured** or **Healthy** using a custom CNN and transfer learning with a fine-tuned **VGG19** model.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-VGG19-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

This project tackles a **binary image classification** problem in the medical imaging domain. Given an X-ray image of a bone, the model predicts whether the bone is **fractured** or **healthy**.

Two approaches are compared:
1. **Custom CNN** — a lightweight convolutional network trained from scratch
2. **Fine-Tuned VGG19** ⭐ — transfer learning using ImageNet-pretrained VGG19 with frozen layers and a custom output head

---

## 🖼️ Sample X-Ray Images

Here are examples of the two classes the model learns to distinguish:

### 🟢 Healthy Bones

| | | |
|:---:|:---:|:---:|
| ![healthy1](https://raw.githubusercontent.com/trisha2103/Bone-Fracture-Detection/main/healthy%20(1).jpg) | ![healthy3](https://raw.githubusercontent.com/trisha2103/Bone-Fracture-Detection/main/healthy%20(3).jpg) | ![healthy4](https://raw.githubusercontent.com/trisha2103/Bone-Fracture-Detection/main/healthy%20(4).jpg) |
| Healthy (1) | Healthy (3) | Healthy (4) |

### 🔴 Fractured Bones

| | | |
|:---:|:---:|:---:|
| ![fractured48](https://raw.githubusercontent.com/trisha2103/Bone-Fracture-Detection/main/fractured%20(48).jpg) | ![fractured49](https://raw.githubusercontent.com/trisha2103/Bone-Fracture-Detection/main/fractured%20(49).jpg) | ![fractured56](https://raw.githubusercontent.com/trisha2103/Bone-Fracture-Detection/main/fractured%20(56).jpg) |
| Fractured (48) | Fractured (49) | Fractured (56) |

> All images are **224×224 JPG** X-rays preprocessed with VGG19's built-in normalization.

---

## 🚀 Features

- ✅ Automated **dataset splitting** into `train / valid / test` directories
- ✅ **VGG19 transfer learning** with frozen pretrained layers
- ✅ Custom CNN baseline for performance comparison
- ✅ **Confusion matrix** visualization for model evaluation
- ✅ VGG19-specific **image preprocessing** pipeline
- ✅ Trained on **1,500 images per class** (3,000 total)

---

## 🗂️ Dataset Structure

```
xray_dataset/
├── train/
│   ├── fractured/     # 750 images
│   └── healthy/       # 750 images
├── valid/
│   ├── fractured/     # 250 images
│   └── healthy/       # 250 images
└── test/
    ├── fractured/     # 50 images
    └── healthy/       # 50 images
```

---

## 🧠 Models

### Model 1 — Custom CNN (Baseline)

| Layer | Config |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU, same padding |
| MaxPool2D | 2×2, stride 2 |
| Conv2D | 64 filters, 3×3, ReLU, same padding |
| MaxPool2D | 2×2, stride 2 |
| Flatten | — |
| Dense (output) | 2 units, Softmax |

### Model 2 — Fine-Tuned VGG19 ⭐

| Property | Details |
|---|---|
| **Base Model** | `VGG19` pretrained on ImageNet |
| **Frozen Layers** | All VGG19 layers (weights locked) |
| **Custom Head** | `Dense(2, activation='softmax')` |
| **Input Shape** | `(224, 224, 3)` |
| **Classes** | `fractured`, `healthy` |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/trisha2103/Bone-Fracture-Detection.git
cd Bone-Fracture-Detection

# 2. Install dependencies
pip install tensorflow scikit-learn matplotlib numpy
```

> 💡 A GPU is recommended for faster training. TensorFlow will fall back to CPU automatically.

---

## 📖 How It Works

### Step 1 — Organize the Dataset

```python
for jpgfile in random.sample(glob.glob('healthy*.jpg'), 750):
    shutil.move(jpgfile, 'train/healthy')
for jpgfile in random.sample(glob.glob('fractured*.jpg'), 750):
    shutil.move(jpgfile, 'train/fractured')
# repeated for valid (250) and test (50) splits
```

### Step 2 — Create Image Data Generators

```python
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input
).flow_from_directory(train_path, target_size=(224, 224),
                      classes=['fractured', 'healthy'], batch_size=40)
```

### Step 3 — Build & Train the Fine-Tuned VGG19

```python
vgg19_model = tf.keras.applications.vgg19.VGG19()

model = Sequential()
for layer in vgg19_model.layers[:-1]:  # remove original output layer
    model.add(layer)

for layer in model.layers:
    layer.trainable = False            # freeze all VGG19 weights

model.add(Dense(units=2, activation='softmax'))  # custom head
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)
```

### Step 4 — Evaluate with Confusion Matrix

```python
predictions = model.predict(x=test_batches, verbose=0)
cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=1))
plot_confusion_matrix(cm=cm, classes=['fractured', 'healthy'],
                      title='Confusion Matrix')
```

---

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Categorical Crossentropy |
| Batch Size | 40 |
| Epochs | 10 |
| Image Size | 224 × 224 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🧠 `TensorFlow / Keras` | Model building & training |
| 🏛️ `VGG19` | Pretrained feature extractor |
| 📊 `scikit-learn` | Confusion matrix |
| 🖼️ `ImageDataGenerator` | Data loading & preprocessing |
| 📈 `matplotlib` | Visualization |

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👩‍💻 Author

**Trisha** — [@trisha2103](https://github.com/trisha2103)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using TensorFlow & Transfer Learning</p>
