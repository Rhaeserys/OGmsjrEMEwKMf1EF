import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandarScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
# Importing file directly from the google drive using gdown
import gdown

# Data Downloading
gdrive_file_id = '1KWE3J0uU_sFIJnZ74Id3FDBcejELI7FD'
gdrive_url = f'https://drive.google.com/uc?id={gdrive_file_id}'
output_csv = 'customer_survey_data.csv'

# File Downloading Steps
try:
    gdown.download(grdrive_url, output_csv, quiet=False)
    print (f"File downloaded successfully as {output_csv}")
except Exception as e:
    print(f"An error occurred during download: {e}")
    print("Please ensure the Google Drive link is shareable or download the file manually and place it in your working directory as 'customer_survey_data.csv'")

# Import Dataset
try:
    df = pd.read_csv(output_csv)
    print("\nData loaded successfully. Here's a preview:")
    print(df.head())
except FileNotFoundError:
    print(f"\nError: The file {output_csv} was not found. Please check the download ste or place the file manually.")

    # Exit if Loading Fails to Execute
    exit()
except Exception as e:
    print(f"\nAn error occurred while loading the CSV: {e}")
    exit()

print("\nData Infrmation:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

# Exploratory Data Analysis
print("\n--- Exploratory Data Analysis ---")


# Response Variable Distribution
plt.figure(figssize-(6, 4))
sns.countplot(x= 'Y', data=df)
plt.title('Distribution of Customer Happiness (Y)')
plt.xlabel('Happiness (0 = Unhappy, 1 = Happy)')
plt.ylabel ('Count')
plt.show()
print("\nTarget Variable Value Counts:")
print(df['Y'].value_counts(normalize-True))

#Feature Administration
features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
for i, feature in enumerate(features):
    plt.figure(figsize=(15, 10))
    sns.countplot(x=features, data=df, hue='y', palette='viridis')
    plt.title(f'Distribution of {featured} by Happiness')
    plt.xlabel(f'Score for {feature}')
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrixc of Features and Target')
plt.show()

print("\nCorrelation with Target Variable (Y):")
print(correlation_matrix['Y'].sort_values(ascending=False))

# Processing Data
print("\n--- Data Processing ---")

# Describe what features (x) and target (y)
X = df.drop('Y', axis=1)
y = df['Y']

#Data splits into testing and training sets of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTraining set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test.shape")

#Logitistic Regression Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Cunstruction and Training
print("\n--- MOdel Building and Training ---")

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

results = {}
kf = StratifiedFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    if model_name == "Logistic Regression":
        # Data is scaled for Logistic Regression
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=kf, scoring='accuracy')
    else:
        # Default data for tree models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"{model_name} Test Accuracy: {accuracy:.4f}")
    print(f"{model_name} Cross-Validation Mean Accuracy: {cv_scores.mean():.rf} (+/- {cv_scores.std():.4f})")
    print("Classification Report:");
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:");
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

print("\n--- Model Comparison ---")
best_model_name = max(results, key=results.get)
print(f"Best performing model based on Test Accuracy: {best_model_name} with Accuracy: {results[best_model_name]:.4f}")

# Save top model for the feature importance
best_model = models[best_model_name]
if best_model_name == "Logistic Regression":
    best_model.fit(X_train_scaled, y_train)
else:
    best_model.fit(X_train, y_train)

# --- Feature Importance and Selection ---
print("\n--- Feature Importance and Selection ---")

# Get feature importances from the best model (if it's tree-based)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title(f'Feature Importances from {best_model_name}')
    plt.show()
    print("\nFeature Importances:")
    print(feature_importance_df)

    # Identify features to potentially remove (e.g., those with very low importance)
    # This is subjective and depends on the distribution of importances
    threshold = 0.05 # Example threshold: features contributing less than 5%
    less_important_features = feature_importance_df[feature_importance_df['importance'] < threshold]['feature'].tolist()
    if less_important_features:
        print(f"\nFeatures with importance less than {threshold*100}%: {less_important_features}")
        print("These could be candidates for removal in future surveys, but consider business context.")
    else:
        print("\nAll features have importance above the threshold or no clear low-importance features identified.")

elif best_model_name == "Logistic Regression":
    # For Logistic Regression, we look at the coefficients
    # Ensure the model was fit on scaled data if interpreting coefficients directly
    if 'X_train_scaled' in globals(): # Check if scaled data was used
        coefficients = best_model.coef_[0]
        feature_names = X.columns
        # Use absolute values for magnitude of importance
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'coefficient_abs': np.abs(coefficients)})
        feature_importance_df = feature_importance_df.sort_values(by='coefficient_abs', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='coefficient_abs', y='feature', data=feature_importance_df)
        plt.title(f'Absolute Coefficient Values from {best_model_name} (on scaled data)')
        plt.show()
        print("\nFeature Importance (Absolute Coefficient Values for Logistic Regression):")
        print(feature_importance_df)
    else:
        print("\nCannot reliably show feature importance for Logistic Regression without scaled data context.")


# --- Recursive Feature Elimination (RFE) with Random Forest ---
# (This is is an alternative or complementary approach for feature selection)
print("\n--- Recursive Feature Elimination (RFE) ---")
rfe_model = RandomForestClassifier(random_state=42)

# Goal is smallest amount of the set.
n_features_to_select = 3
rfe = RFE(estimator=rfe_model, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train) # Use original X_train for tree-based RFE

selected_features = X.columns[rfe.support_]
print(f"\nTop {n_features_to_select} features selected by RFE: {selected_features.tolist()}")

# Train a model with only RFE selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

model_with_rfe_features = RandomForestClassifier(random_state=42)
model_with_rfe_features.fit(X_train_rfe, y_train)
y_pred_rfe = model_with_rfe_features.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)

print(f"\nAccuracy of Random Forest with RFE selected features ({len(selected_features)} features): {accuracy_rfe:.4f}")
print("Classification Report for model with RFE selected features:")
print(classification_report(y_test, y_pred_rfe))

# Contrast the model accuracy.
if accuracy_rfe >= results[best_model_name] * 0.98: # e.g., if it preserves 98% of the best model's accuracy
    print(f"\nThe RFE-selected features ({selected_features.tolist()}) provide good accuracy ({accuracy_rfe:.4f}) compared to the full feature set.")
    print("This suggests that these questions are the most critical.")
    print("Questions that can potentially be removed (those not in RFE list):")
    removable_questions = [q for q in X.columns if q not in selected_features.tolist()]
    print(removable_questions)
else:
    print(f"\nUsing only RFE selected features ({selected_features.tolist()}) leads to a notable drop in accuracy ({accuracy_rfe:.4f}).")
    print("It might be better to keep more features for optimal prediction.")
