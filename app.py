import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load the data
print("🔄 Loading dataset (this might take a few seconds)...")
df = pd.read_csv('creditcard.csv')

# 2. Scale the 'Amount' feature to normalize values
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)

# 3. Separate features and targets
X = df.drop('Class', axis=1)
y = df['Class']

# Split into training and testing sets while preserving target balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Original training shapes: {X_train.shape}, Frauds in train: {sum(y_train)}")

# 4. Apply SMOTE to fix the massive imbalance in training data
print("⚖️ Applying SMOTE to balance data classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Balanced training shapes: {X_train_res.shape}, Frauds after SMOTE: {sum(y_train_res)}")

# 5. Train the Random Forest Model
print("🤖 Training the classifier (Random Forest)...")
model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)  # reduced estimators for speed
model.fit(X_train_res, y_train_res)

# 6. Evaluate model performance
print("\n📈 Model Evaluation Results:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))