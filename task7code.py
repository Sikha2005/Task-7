# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("E:\data.csv")  # adjust path if needed

# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Encode target variable (M=1, B=0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Select 2 features for 2D visualization
X = df[['radius_mean', 'texture_mean']].values
y = df['diagnosis'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM with Linear Kernel
svm_linear = SVC(kernel='linear', C=1.0, probability=True)
svm_linear.fit(X_train_scaled, y_train)

# Train SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_rbf.fit(X_train_scaled, y_train)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('radius_mean')
    plt.ylabel('texture_mean')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Plot decision boundaries
plot_decision_boundary(svm_linear, X_train_scaled, y_train, 'SVM with Linear Kernel')
plot_decision_boundary(svm_rbf, X_train_scaled, y_train, 'SVM with RBF Kernel')

# Hyperparameter tuning using cross-validation
print("\nHyperparameter Tuning Results:")
for C in [0.1, 1, 10]:
    for gamma in [0.01, 0.1, 1]:
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
        print(f'C={C}, gamma={gamma}, Mean CV Accuracy: {scores.mean():.3f}')
