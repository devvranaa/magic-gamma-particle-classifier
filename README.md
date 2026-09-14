
# MAGIC Gamma Telescope Particle Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devvranaa/magic-gamma-particle-classifier/blob/main/magic_gamma.ipynb)

A supervised machine learning and deep learning project to differentiate high-energy primary **gamma rays (signal)** from background **hadron cosmic rays (noise)** using simulated Cherenkov shower images from the Major Atmospheric Gamma Imaging Cherenkov (MAGIC) Telescope.

---

## 📌 Dataset Overview

* **Source:** [UCI Machine Learning Repository - MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
* **Instances:** 19,020 observations
* **Attributes:** 10 continuous geometric features
* **Target:** `class` (Binary: `1` for Gamma Signal, `0` for Hadron Background Noise)

### Key Features (Hillas Parameters)
| Feature | Description |
| :--- | :--- |
| `fLength` | Major axis of the light ellipse [mm] |
| `fWidth` | Minor axis of the light ellipse [mm] |
| `fSize` | 10-log of sum of content of all pixels [photons] |
| `fConc` | Ratio of sum of two highest pixels over total |
| `fConc1` | Ratio of highest pixel over total |
| `fAsym` | Distance from highest pixel to center along major axis |
| `fM3Long` | 3rd root of 3rd moment along major axis |
| `fM3Trans` | 3rd root of 3rd moment along transverse axis |
| `fAlpha` | Angle of major axis with vector to center [deg] |
| `fDist` | Distance from image center to ellipse center [mm] |

---

## ⚙️ Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA):** Plotted normalized probability density histograms across all 10 features to analyze class distributions and feature separability.
2. **Data Partitioning:** Split data into **Train (60%)**, **Validation (20%)**, and **Test (20%)** sets.
3. **Class Imbalance Resolution:** Applied `RandomOverSampler` (`imblearn`) to the training set to achieve a balanced 50/50 split between gamma and hadron samples, preventing majority-class bias.
4. **Feature Standardization:** Standardized all inputs using `StandardScaler` ($Z = \frac{x - \mu}{\sigma}$) to ensure distance-based models and neural network gradient updates treat all features equitably.

---

## 🧪 Models Evaluated

### Traditional Machine Learning (Baselines)
* **k-Nearest Neighbors (k-NN)**
* **Naive Bayes**
* **Logistic Regression**
* **Support Vector Machine (SVM)** (RBF Kernel)

### Deep Learning Architecture & Hyperparameter Tuning
Built a fully connected feedforward Neural Network using **TensorFlow / Keras**:
* **Architecture:** 
  * Input layer: 10 features
  * Hidden Layers: `Dense` layers with `ReLU` activations and `Dropout` regularization
  * Output layer: Single neuron with `Sigmoid` activation (outputs probability $[0, 1]$)
* **Optimization:** `Adam` optimizer paired with `binary_crossentropy` loss.
* **Grid Search Hyperparameter Tuning:**
  * **Nodes:** `[16, 32, 64]`
  * **Dropout Rates:** `[0, 0.2]`
  * **Learning Rates:** `[0.01, 0.005, 0.001]`
  * **Batch Sizes:** `[32, 64, 128]`
  * Evaluated across **100 epochs** per combination, tracking `val_loss` to prevent overfitting.

---

## 📊 Results & Performance Comparison

| Model | Accuracy | F1-Score (Signal) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Naive Bayes** | ~72% | ~0.77 | Fast baseline; struggles with non-independent features. |
| **Logistic Regression** | ~78% | ~0.83 | Good linear baseline, but cannot capture complex decision boundaries. |
| **k-Nearest Neighbors** | ~81% | ~0.85 | Solid non-linear metric; sensitive to high-dimensional distance density. |
| **Support Vector Machine (SVM)** | ~85% | ~0.88 | Strong non-linear separation with RBF kernel. |
| **🏆 Tuned Neural Network** | **88%** | **0.91** | **Best overall performance; highest recall and precision balance.** |

---

### Final Neural Network Classification Report (Test Set)

```text
              precision    recall  f1-score   support

           0       0.89      0.76      0.82      1348
           1       0.88      0.95      0.91      2456

    accuracy                           0.88      3804
   macro avg       0.88      0.85      0.86      3804
weighted avg       0.88      0.88      0.88      3804 Imbalanced-Learn
* **Environment:** Google Colab / Jupyter Notebook
