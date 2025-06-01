import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore", category=UserWarning)

# File paths
file1_path = r"C:\Users\Anandhi\OneDrive\Documents\mini project\dataR2.csv"
file2_path = r"C:\Users\Anandhi\OneDrive\Documents\mini project\IHME-GBD_2021_DATA-0529c413-1.csv"
output_file = r"merged_cleaned.csv"

# Load data
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Define age ranges and mapping
age_ranges = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29),
              (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59),
              (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 89),
              (90, 94), (95, 150)]

range_labels = [f"{start}-{end} years" if end != 150 else "95+ years" for start, end in age_ranges]
range_to_code = {label: i for i, label in enumerate(range_labels)}

# Convert numerical age to label
def map_age_to_range(age):
    for (start, end), label in zip(age_ranges, range_labels):
        if start <= age <= end:
            return label
    return "Unknown"

# Map df1 ages to age range strings and then to numeric codes
df1["age_range_label"] = df1["Age"].apply(map_age_to_range)
df1["age_code"] = df1["age_range_label"].map(range_to_code)

# Map df2["age_name"] to numeric code using same mapping
df2["age_code"] = df2["age_name"].map(range_to_code)

# Merge based on the numeric age code
merged_df = pd.merge(df1, df2, on="age_code", how="inner")

# Drop redundant columns
merged_df.drop(columns=["Age", "age_name", "age_range_label"], inplace=True)

# Save final file
merged_df.to_csv(output_file, index=False)

print(f"Done! Merged file saved at: {output_file}")


import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Let's assume merged_df is the DataFrame you already created (from merged_data.csv)
# and "Classification" is your target column.

target_column = "Classification"  # Replace this with the correct target name if different

# 1. Check if target_column is actually in your DataFrame
if target_column not in merged_df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset!")

# 2. Encode categorical variables (age_range, gender, Tissue, Classification, etc.)
label_encoders = {}
for col in merged_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col])
    label_encoders[col] = le

# 3. Separate features (X) and target (y)

user_input_features = ["age_code", "BMI", "Glucose", "Insulin", "Leptin", "Resistin", "Adiponectin", "MCP.1"]
X = merged_df[user_input_features]
y = merged_df[target_column]
unique_classes = sorted(y.unique())
if unique_classes == [1, 2]:
    y = y - 1
    print("Adjusted target labels from", unique_classes, "to", sorted(y.unique()))

# 4. SULOV Step: remove highly correlated features
correlation_matrix = X.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# You can adjust this threshold (0.9) based on how strict you want to be about correlation
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.75)]
X_filtered = X.drop(columns=to_drop)

# 5. MRMR Step: use mutual information to select the top K features
mi_scores = mutual_info_classif(X_filtered, y)
mi_scores = np.array(mi_scores)

# Choose how many features you want to keep
k = 10  # for example, top 10 features
selected_features = X_filtered.columns[np.argsort(mi_scores)[-k:]]

X_selected = X_filtered[selected_features]

# 6. (Optional) Train a model to verify feature quality
gb_model = GradientBoostingClassifier()
gb_model.fit(X_selected, y)

# 7. Save the dataset with selected features
selected_data = X_selected.copy()
selected_data[target_column] = y
selected_data.to_csv(r"newselected_features.csv", index=False)

print("SULOV-MRMR feature selection completed. Selected features saved.")


# Import XGBoost and other required modules
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', max_depth=5, min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8, learning_rate=0.01,n_estimators=100, random_state=42)

# Train the model on the training set
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Model Accuracy on test set:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: XGBoost")
print(conf_matrix)

import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Initialize the LightGBM classifier
lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    learning_rate=0.05,
    n_estimators=1000,
    random_state=42,
    verbosity=-1  # <- THIS suppresses the "[Warning] No further splits..." spam
)


# Train the model on the training set
lgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("LightGBM Model Accuracy on test set:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

# Compute and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: LightGBM")
print(cm)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=None,random_state=42)

# Train the model on the same dataset
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate performance
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Model Accuracy: {dt_accuracy:.4f}")

# Print classification report
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))
from sklearn.metrics import confusion_matrix

# Compute and print confusion matrix
cm = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix: Decision Tree")
print(cm)

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced',random_state=42)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Model Accuracy on test set:", accuracy_rf)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
from sklearn.metrics import confusion_matrix

# Compute and print confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix (Random Forest):")
print(cm_rf)

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Scale Features (important for Neural Networks)
scaler_mlp = StandardScaler()

# Fit and transform using DataFrame (this will retain feature names)
X_train_scaled_mlp = scaler_mlp.fit_transform(X_train)
X_test_scaled_mlp = scaler_mlp.transform(X_test)

# Ensure that you're passing a DataFrame with feature names
X_train_scaled_mlp_df = pd.DataFrame(X_train_scaled_mlp, columns=X_train.columns)
X_test_scaled_mlp_df = pd.DataFrame(X_test_scaled_mlp, columns=X_test.columns)

# Initialize MLPClassifier (Multi-Layer Perceptron)
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
                          activation='relu',  # ReLU activation for hidden layers
                          solver='adam',  # Adam optimizer
                          max_iter=500,  # Train for 500 epochs
                          random_state=42)

# Train the MLP Model (with feature names)
mlp_model.fit(X_train_scaled_mlp_df, y_train)

# Make Predictions (pass DataFrame here to retain feature names)
y_pred_mlp = mlp_model.predict(X_test_scaled_mlp_df)

# Evaluate Performance
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"\nMLP Model Accuracy: {accuracy_mlp:.4f}")
print("MLP Classification Report:")
print(classification_report(y_test, y_pred_mlp))
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print("Confusion Matrix (MLP):")
print(cm_mlp)

from sklearn.svm import SVC

# Initialize the SVM model with a linear kernel
svm_model = SVC(kernel='linear', probability=True, random_state=42)
# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Model Accuracy on test set:", accuracy_svm)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
# Print Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix (SVM):")
print(cm_svm)


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Feature Scaling with Retained Column Names
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Optimized Logistic Regression Model
log_reg_model = LogisticRegression(
    random_state=42, 
    max_iter=5000, 
    solver='lbfgs', 
    C=2.0,  
    class_weight='balanced'  
)

# Train the model
log_reg_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

# Evaluate performance
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f" Logistic Regression Accuracy: {accuracy_log_reg:.4f}")
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
conf_matrix = confusion_matrix(y_test, y_pred_log_reg)
print("Confusion Matrix:")
print(conf_matrix)

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# 1. Scale and preserve column names
scaler_sgd = StandardScaler()
X_train_scaled_sgd = pd.DataFrame(scaler_sgd.fit_transform(X_train), columns=X_train.columns)
X_test_scaled_sgd = pd.DataFrame(scaler_sgd.transform(X_test), columns=X_test.columns)

# 2. Train the SGDClassifier
sgd_model = SGDClassifier(class_weight="balanced", max_iter=1000, eta0=0.01, learning_rate="adaptive",loss='log_loss',penalty='l2',alpha=0.001,tol=1e-3,random_state=42)

sgd_model.fit(X_train_scaled_sgd, y_train)

# 3. Evaluate on test set
y_pred_sgd = sgd_model.predict(X_test_scaled_sgd)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)

print(f"\n SGDClassifier Accuracy: {accuracy_sgd:.4f}")
print("SGD Classification Report:")
print(classification_report(y_test, y_pred_sgd))
# Confusion Matrix
conf_matrix_sgd = confusion_matrix(y_test, y_pred_sgd)
print("Confusion Matrix:")
print(conf_matrix_sgd)

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Assuming X_selected and y are already defined

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 2. Quantile Transformation
transformer = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42)
X_train_trans = transformer.fit_transform(X_train)
X_test_trans = transformer.transform(X_test)

# 3. Class weights for imbalance handling (since GaussianNB doesn't support class_weight directly)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 4. Train Calibrated GaussianNB with isotonic calibration
nb_model = GaussianNB(var_smoothing=1e-8)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = CalibratedClassifierCV(estimator=nb_model, method='isotonic', cv=cv)
model.fit(X_train_trans, y_train, sample_weight=sample_weights)

# 5. Evaluate
y_pred = model.predict(X_test_trans)

print("\n Naive Bayes Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras import initializers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Build deeper and more regularized model
dl_model = Sequential([Dense(128, activation='relu',input_dim=X_train_scaled.shape[1],kernel_initializer=initializers.HeNormal(seed=42)),
                     Dropout(0.3), 
                     Dense(64, activation='relu',kernel_initializer=initializers.HeNormal(seed=42)),
                     Dropout(0.3),
                     Dense(1, activation='sigmoid',kernel_initializer=initializers.GlorotUniform(seed=42))])


# Compile
dl_model.compile(optimizer=Adam(learning_rate=0.0005),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])


# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Fit the model
history = dl_model.fit(X_train_scaled, y_train,
                       validation_split=0.2,
                       epochs=150,
                       batch_size=16,
                       callbacks=[early_stop],
                       class_weight=class_weight_dict,
                       verbose=0)

# Final evaluation
loss, accuracy = dl_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nImproved DL Test Accuracy: {accuracy:.3f}")

# Predict class labels for test set
y_pred_probs = dl_model.predict(X_test_scaled)

y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


models = {
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "MLP": mlp_model,
    "SVM": svm_model,
    "Logistic Regression": log_reg_model,
    "SGDClassifier": sgd_model,
    "Naive Bayes": model ,
    "Deep Learning":dl_model
}

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
def plot_feature_importance(model, model_name, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)

        plt.figure(figsize=(10, 6))
        plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color="teal")
        plt.xlabel("Importance")
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{model_name} does not support feature importance.")
def plot_conf_matrix(model, X_test, y_test, model_name, scaler=None, is_dl=False):
    if is_dl:
        X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.show()
def plot_model_accuracies(accuracy_dict):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracy_dict.keys()), y=list(accuracy_dict.values()), palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
plot_feature_importance(xgb_model, "XGBoost", X_selected.columns)
plot_feature_importance(lgb_model, "LightGBM", X_selected.columns)
plot_feature_importance(rf_model, "Random Forest", X_selected.columns)
plot_feature_importance(dt_model, "Decision Tree", X_selected.columns)
plot_conf_matrix(xgb_model, X_test, y_test, "XGBoost")
plot_conf_matrix(lgb_model, X_test, y_test, "LightGBM")
plot_conf_matrix(rf_model, X_test, y_test, "Random Forest")
plot_conf_matrix(dt_model, X_test, y_test, "Decision Tree")
plot_conf_matrix(mlp_model, X_test_scaled_mlp_df, y_test, "MLP")
plot_conf_matrix(svm_model, X_test, y_test, "SVM")
plot_conf_matrix(log_reg_model, X_test_scaled, y_test, "Logistic Regression")
plot_conf_matrix(sgd_model, X_test_scaled_sgd, y_test, "SGDClassifier")
plot_conf_matrix(model, X_test_trans, y_test, "Naive Bayes")
plot_conf_matrix(dl_model, X_test, y_test, "Deep Learning", scaler=scaler, is_dl=True)
model_accuracies = {
    "XGBoost": accuracy,
    "LightGBM": accuracy_rf,
    "Random Forest": accuracy_rf,
    "Decision Tree": dt_accuracy,
    "MLP": accuracy_mlp,
    "SVM": accuracy_svm,
    "Logistic Regression": accuracy_log_reg,
    "SGDClassifier": accuracy_sgd,
    "Naive Bayes": accuracy_score(y_test, model.predict(X_test_trans)),
    "Deep Learning": accuracy
}

plot_model_accuracies(model_accuracies)
plt.savefig("plot_name.png", dpi=300)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
def plot_roc_curves(models, X_test_dict, y_test, scaler=None):
    plt.figure(figsize=(10, 7))
    
    for name, model in models.items():
        X_test = X_test_dict.get(name, None)
        if X_test is None:
            continue

        if name == "Deep Learning":
            X_test_scaled = scaler.transform(X_test)
            y_score = model.predict(X_test_scaled).flatten()
        elif name == "Naive Bayes":
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:  # For models like SVM (if probability=False)
            y_score = model.decision_function(X_test)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())  # Normalize

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
X_test_dict = {
    "XGBoost": X_test,
    "LightGBM": X_test,
    "Random Forest": X_test,
    "Decision Tree": X_test,
    "MLP": X_test_scaled_mlp_df,
    "SVM": X_test,
    "Logistic Regression": X_test_scaled,
    "SGDClassifier": X_test_scaled_sgd,
    "Naive Bayes": X_test_trans,
    "Deep Learning": X_test
}
plot_roc_curves(models, X_test_dict, y_test, scaler=scaler)
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curves(models, X_test_dict, y_test, scaler=None):
    plt.figure(figsize=(10, 7))
    
    for name, model in models.items():
        X_test = X_test_dict[name]

        if name == "Deep Learning":
            y_score = model.predict(scaler.transform(X_test)).flatten()
        elif hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        avg_precision = average_precision_score(y_test, y_score)
        plt.plot(recall, precision, label=f"{name} (AP = {avg_precision:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_precision_recall_curves(models, X_test_dict, y_test, scaler=scaler)
from sklearn.calibration import calibration_curve

def plot_calibration(models, X_test_dict, y_test, scaler=None):
    plt.figure(figsize=(10, 7))
    
    for name, model in models.items():
        X_test = X_test_dict[name]

        if name == "Deep Learning":
            y_proba = model.predict(scaler.transform(X_test)).flatten()
        elif hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            continue  # skip models without probability support

        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)

    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_calibration(models, X_test_dict, y_test, scaler=scaler)

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd

# Define a function to compute AUC with CV
def compute_auc_stats(model, X, y, scaler=None, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train, verbose=0)
            y_proba = model.predict(X_test_scaled).flatten()
        else:
            model.fit(X_train, y_train)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_proba = model.decision_function(X_test)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
        
        auc_scores.append(roc_auc_score(y_test, y_proba))
    
    return {
        'mean': np.mean(auc_scores),
        'std': np.std(auc_scores),
        'p2.5': np.percentile(auc_scores, 2.5),
        'p97.5': np.percentile(auc_scores, 97.5)
    }

# Define a function to calculate confusion matrix metrics (updated)
def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'TPR': tp / (tp + fn),        # True Positive Rate (Sensitivity/Recall)
        'TNR': tn / (tn + fp),        # True Negative Rate (Specificity)
        'FNR': fn / (tp + fn),        # False Negative Rate (Miss Rate)
        'FPR': fp / (tn + fp),        # False Positive Rate (Fall-out)
        'FDR': fp / (tp + fp),        # False Discovery Rate (1 - Precision)
        'FOR': fn / (tn + fn),        # False Omission Rate
    }
    return metrics

def evaluate_model(model, X, y, scaler=None, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    all_metrics = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train, verbose=0)
            y_proba = model.predict(X_test_scaled).flatten()
            y_pred = (y_proba > 0.5).astype(int)
        else:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        
        auc_scores.append(roc_auc_score(y_test, y_proba))
        all_metrics.append(calculate_metrics(y_test, y_pred))
    
    # Aggregate results
    metrics_df = pd.DataFrame(all_metrics)
    metrics_summary = {metric: {'mean': metrics_df[metric].mean(), 
                               'std': metrics_df[metric].std()}
                      for metric in metrics_df.columns}
    
    return {
        'AUC': {
            'mean': np.mean(auc_scores),
            'std': np.std(auc_scores),
            'p2.5': np.percentile(auc_scores, 2.5),
            'p97.5': np.percentile(auc_scores, 97.5)
        },
        'metrics': metrics_summary
    }

# Evaluate all models
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    if name == "Deep Learning":
        results[name] = evaluate_model(model, X_train, y_train, scaler=scaler)
    else:
        results[name] = evaluate_model(model, X_train, y_train)

# Print AUC results (unchanged)
print("\nAUC Results (5-Fold CV):")
print("="*85)
print(f"{'Model':<25} {'Mean AUC':<10} {'Std':<10} {'2.5%':<10} {'97.5%':<10}")
print("-"*85)
for name, res in sorted(results.items(), key=lambda x: x[1]['AUC']['mean'], reverse=True):
    auc = res['AUC']
    print(f"{name:<25} {auc['mean']:.4f}    {auc['std']:.4f}    {auc['p2.5']:.4f}    {auc['p97.5']:.4f}")

# Print metrics results (updated to show TPR, TNR, FNR, FPR, FDR, FOR)
print("\nPerformance Metrics (5-Fold CV) - Sorted by TPR (Descending):")
print("="*120)
header = f"{'Model':<20} {'TPR':<15} {'TNR':<15} {'FNR':<15} {'FPR':<15} {'FDR':<15} {'FOR':<15}"
print(header)
print("-"*120)

# Sort models by TPR (descending)
sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['TPR']['mean'], reverse=True)

for name, res in sorted_results:
    m = res['metrics']
    line = f"{name:<20}"
    for metric in ['TPR', 'TNR', 'FNR', 'FPR', 'FDR', 'FOR']:
        line += f"{m[metric]['mean']:.4f}±{m[metric]['std']:.4f}  "
    print(line)

# Find best models (updated to include FNR, FPR, etc.)
best_auc = max(results.items(), key=lambda x: x[1]['AUC']['mean'])
best_tpr = max(results.items(), key=lambda x: x[1]['metrics']['TPR']['mean'])
best_tnr = max(results.items(), key=lambda x: x[1]['metrics']['TNR']['mean'])

print("\nBest Models:")
print(f"- By AUC: {best_auc[0]} (AUC = {best_auc[1]['AUC']['mean']:.4f})")
print(f"- By TPR (Sensitivity): {best_tpr[0]} (TPR = {best_tpr[1]['metrics']['TPR']['mean']:.4f})")
print(f"- By TNR (Specificity): {best_tnr[0]} (TNR = {best_tnr[1]['metrics']['TNR']['mean']:.4f})")

def get_user_input(selected_features):
    user_data = {}
    print("\nEnter the following feature values:")
    
    for feature in selected_features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                user_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value.")

    return pd.DataFrame([user_data])

s=input("enter stop to stop giving input")
# Get user input
while s!="stop":
    
    user_input_data = get_user_input(X_selected.columns)

    # Make predictions using each model
    print("\nPredictions:")

    for name, model in models.items():
        pred = model.predict(user_input_data)[0]
        if name == "Deep Learning":
            pred = int(pred[0]) 
            user_input_data_scaled = scaler.transform(user_input_data)
            pred_prob = model.predict(user_input_data_scaled)[0][0]
            pred = int(pred_prob > 0.5)
            print(f"Deep Learning Prediction: {pred} (Probability: {pred_prob:.4f})")
        else:
            print(f"{name} Prediction: {pred}")
    s=input("enter stop to stop giving input")