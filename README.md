
# MAGIC Gamma Telescope Particle Classification

This project builds a supervised binary classification pipeline to differentiate high-energy **primary gamma rays (signal)** from background **hadron cosmic rays (noise)** using observational data from the Major Atmospheric Gamma Imaging Cherenkov (MAGIC) Telescope.

---

## 📌 Dataset Overview
* **Source:** UCI Machine Learning Repository
* **Dataset Size:** 19,020 instances, 10 continuous attributes
* **Target Variable:** `class` (`1` for Gamma rays, `0` for Hadron cosmic background)
* **Key Features:**
  * `fLength`: Major axis of ellipse [mm]
  * `fWidth`: Minor axis of ellipse [mm]
  * `fSize`: 10-log of sum of content of all pixels [in photons]
  * `fConc`: Ratio of sum of two highest pixels over total
  * `fConc1`: Ratio of highest pixel over total
  * `fAsym`: Distance from highest pixel to center, projected onto major axis
  * `fM3Long` & `fM3Trans`: 3rd root of 3rd moment along major/transverse axis
  * `fAlpha`: Angle of major axis with vector to center [deg]
  * `fDist`: Distance from center of image to center of ellipse [mm]

---

## ⚙️ Machine Learning Pipeline
1. **Data Exploration & Visualization:** Plotted overlapping probability density distribution histograms across all continuous features using Matplotlib to assess feature separability.
2. **Data Splitting:** Partitioned data into Training, Validation, and Testing sets.
3. **Class Imbalance Resolution:** Applied `RandomOverSampler` (`imblearn`) on the training set to equalize gamma and hadron sample distributions (50/50 balance).
4. **Feature Standardization:** Standardized 10 geometric features via `StandardScaler` (Z-score scaling) to ensure equal feature weighting for distance-based models.
5. **Model Training & Evaluation:**
   * **K-Nearest Neighbors (KNN)**
   * **Logistic Regression**
   * **Naive Bayes**
   * **Support Vector Machines (SVM)**
   * Evaluated model accuracy, precision, recall, and F1-scores using `classification_report`.

---

## 🛠️ Technologies & Libraries Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Imbalanced-Learn
* **Environment:** Google Colab / Jupyter Notebook
