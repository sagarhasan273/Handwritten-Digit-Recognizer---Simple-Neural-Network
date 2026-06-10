
---

## 📁 Project 2: Handwritten Digit Recognizer

### README.md

```markdown
# Handwritten Digit Recognizer 🔢

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)](https://matplotlib.org/)

## 📌 Overview

A deep learning model that recognizes handwritten digits (0-9) using a neural network built with TensorFlow/Keras. Trained on the famous MNIST dataset of 70,000 handwritten digits.

**Test Accuracy:** 97.8% 🔥

## 🎯 Problem Statement

Given a 28x28 pixel grayscale image of a handwritten digit, classify it into one of 10 classes (0-9). This is the "Hello World" of computer vision and deep learning.

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | TensorFlow, Keras |
| **Numerical Computing** | NumPy |
| **Visualization** | Matplotlib |
| **Development** | Jupyter Notebook / Google Colab |
| **Language** | Python 3.8+ |

## 🧠 Neural Network Architecture
┌─────────────────────────────────────────┐
│ Input Layer (28x28 image = 784 pixels) │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ Flatten Layer (784 → 784) │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ Hidden Layer: Dense(128, ReLU) │
│ 128 neurons with ReLU activation │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ Dropout Layer (20%) │
│ Prevents overfitting │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ Output Layer: Dense(10, Softmax) │
│ 10 neurons (digits 0-9) │
└─────────────────────────────────────────┘

Total Parameters: 101,770
Trainable Parameters: 101,770



## 📊 Dataset: MNIST

### Source
**The MNIST Database** - Yann LeCun, NYU

### Dataset Details
| Property | Value |
|----------|-------|
| **Total Images** | 70,000 |
| **Training Images** | 60,000 |
| **Test Images** | 10,000 |
| **Image Size** | 28x28 pixels (grayscale) |
| **Classes** | 10 digits (0-9) |
| **File Size** | ~11 MB (compressed) |
| **Year Created** | 1998 |

### Sample Distribution
| Digit | Training Count |
|-------|----------------|
| 0 | 5,923 |
| 1 | 6,742 |
| 2 | 5,958 |
| 3 | 6,131 |
| 4 | 5,842 |
| 5 | 5,421 |
| 6 | 5,985 |
| 7 | 6,265 |
| 8 | 5,851 |
| 9 | 5,942 |

## 🚀 How to Run

### Option 1: Google Colab (Recommended - No Setup)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Just click the badge above and run all cells!

### Option 2: Local Machine

```bash
# 1. Clone repository
git clone https://github.com/yourusername/digit-recognizer.git
cd digit-recognizer

# 2. Install dependencies
pip install tensorflow numpy matplotlib jupyter

# 3. Run notebook
jupyter notebook digit_recognizer.ipynb
