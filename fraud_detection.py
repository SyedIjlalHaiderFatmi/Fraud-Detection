import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from imblearn.over_sampling import SMOTE

# 1. Load the dataset
# Ensure you have 'creditcard.csv' in your working directory
data = pd.read_csv('creditcard.csv')

# 2. Preprocessing 
# Scaling the 'Amount' and 'Time' as they are not scaled like V1-V28
scaler = StandardScaler()
data['std_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['std_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop original unscaled columns
data.drop(['Time', 'Amount'], axis=1, inplace=True)

# 3. Define Features and Target
X = data.drop('Class', axis=1)
y = data['Class']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Handle Class Imbalance 
# Using SMOTE to oversample the minority (fraud) class
print("Original class distribution:", np.bincount(y_train))
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("Resampled class distribution:", np.bincount(y_train_res))

# 5. Train the Classification Model 
# Random Forest is robust for high-dimensional anonymized variables [cite: 7]
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

# 6. Evaluation 
y_pred = model.predict(X_test)
auprc = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

print("\n--- Model Evaluation ---")
print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Fraud vs Legitimate')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
