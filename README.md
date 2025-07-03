# 📊 Support Vector Machine (SVM) Classifier — Breast Cancer Detection

## 📌 Objective:
To use **Support Vector Machines (SVM)** for **linear** and **non-linear (RBF kernel)** classification to detect breast cancer tumors as **Malignant (M)** or **Benign (B)** based on selected features.

---

## 📦 Tools & Libraries:
- **Python 3.x**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

---

## 📁 Dataset:
- **File:** `data.csv`
- **Source:** Breast Cancer Wisconsin dataset  
- **Target Column:** `diagnosis`
  - `M` → Malignant (1)
  - `B` → Benign (0)

- **Features used for visualization:**  
  - `radius_mean`  
  - `texture_mean`  

*(selected for 2D decision boundary plots)*

---

## 📊 Task Steps:
1. Load and preprocess the dataset:
   - Drop unnecessary columns (`id`, `Unnamed: 32`).
   - Encode target variable.
2. Select 2 key features for 2D visualization.
3. Split data into **training** and **testing** sets.
4. Standardize features.
5. Train two SVM models:
   - **Linear Kernel**
   - **RBF Kernel**
6. Visualize decision boundaries.
7. Tune hyperparameters `C` and `gamma` for RBF kernel using cross-validation.
8. Display cross-validation accuracy results.

---

## 📈 Performance Summary:
- **Linear SVM:**  
  - Simple, straight-line boundary

- **RBF Kernel SVM:**  
  - Flexible, non-linear boundary
  - **Best Hyperparameter Config:**  
    `C=10`, `gamma=0.1` with highest cross-validation accuracy ~0.901

---

## 📊 How to Run:
1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   
---

**##output**:
**Hyperparameter Tuning Results:**

C=0.1, gamma=0.01, Mean CV Accuracy: 0.675

C=0.1, gamma=0.1, Mean CV Accuracy: 0.890

C=0.1, gamma=1, Mean CV Accuracy: 0.899

C=1, gamma=0.01, Mean CV Accuracy: 0.884

C=1, gamma=0.1, Mean CV Accuracy: 0.899

C=1, gamma=1, Mean CV Accuracy: 0.897

C=10, gamma=0.01, Mean CV Accuracy: 0.890

C=10, gamma=0.1, Mean CV Accuracy: 0.901

C=10, gamma=1, Mean CV Accuracy: 0.895
