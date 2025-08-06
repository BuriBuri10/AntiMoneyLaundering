import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    log_loss
)
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

print("Libraries imported successfully__>>>>>>>>>>>")

data = pd.read_csv("C://0//Evrything//TrainLikeHell//SAML-D.csv//SAML-D.csv")
df = pd.DataFrame(data)
df = pd.concat([df], ignore_index=True)

## NOTE: Prep  & Feature Engineering

# Combining Date and Time into a single datetime column
df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df = df.drop(columns=['Date', 'Time'])

# Sorting by time for potential sequential analysis
df = df.sort_values(by='Timestamp').reset_index(drop=True)

print("Data loaded and initial preparation done.")
print("Starting feature engineering...")

# Time-based Features
df['hour_of_day'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Transactional Features
df['is_foreign_exchange'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
df['is_cross_border'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)
df['log_amount'] = np.log1p(df['Amount']) # Use log of amount to handle skewed distributions

# Relational / Behavioral Features
df['sender_transaction_count'] = df.groupby('Sender_account')['Sender_account'].transform('count')
df['sender_total_amount'] = df.groupby('Sender_account')['Amount'].transform('sum')
df['receiver_transaction_count'] = df.groupby('Receiver_account')['Receiver_account'].transform('count')
df['receiver_total_amount'] = df.groupby('Receiver_account')['Amount'].transform('sum')
df['sender_fan_out'] = df.groupby('Sender_account')['Receiver_account'].transform('nunique')
df['receiver_fan_in'] = df.groupby('Receiver_account')['Sender_account'].transform('nunique')

print("Feature engineering complete.")
print("Engineered DataFrame sample:")
print(df.head())


X = df.drop(columns=['Is_laundering', 'Laundering_type', 'Timestamp', 'Sender_account', 'Receiver_account'])
y = df['Is_laundering']

# One-Hot Encoding all categorical features
X = pd.get_dummies(X, columns=[
    'Payment_currency', 'Received_currency',
    'Sender_bank_location', 'Receiver_bank_location', 'Payment_type'
], drop_first=True)

print("\nFinal feature set shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Calculating scale_pos_weight to handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale Pos Weight: {scale_pos_weight:.2f}") # This value will be high, which is expected

# Initializing and Training the XGBoost Classifier
model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='aucpr',  # Area Under Precision-Recall Curve, good for imbalanced data
    n_estimators=200,     # Number of trees
    learning_rate=0.1,    # Step size shrinkage
    max_depth=4,          # Max depth of a tree
    subsample=0.8,        # Subsample ratio of the training instance
    colsample_bytree=0.8, # Subsample ratio of columns when constructing each tree
    random_state=42
)

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Laundering', 'Laundering']))

auprc = average_precision_score(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)
print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
print(f"Area Under ROC Curve (AUC-ROC): {roc_auc:.4f}")
print(f"Log Loss & AUC: {log_loss}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Laundering', 'Predicted Laundering'],
            yticklabels=['Actual Not Laundering', 'Actual Laundering'])
plt.title('Confusion Matrix')
plt.show()

fig, ax = plt.subplots(figsize=(12, 10))

sorted_idx = model.feature_importances_.argsort()
N = 20 
plt.barh(X.columns[sorted_idx][-N:], model.feature_importances_[sorted_idx][-N:])

plt.xlabel("XGBoost Feature Importance (Gain)")
plt.title(f"Top {N} Feature Importances")
plt.show()

print("\nSaving Model and Columns for Deployment")

# Saving the trained model to a file
joblib.dump(model, 'aml_xgb_model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')

print("Model and column list saved successfully.")
print("Files 'aml_xgb_model.pkl' and 'model_columns.pkl' are ready for deployment")