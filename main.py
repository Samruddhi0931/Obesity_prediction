# Course: CS 584 003
# Team : Surbhi Kharche , Deepika Naik, Shreyas Patil , Samruddhi Deshmukh
# Date : 11/27/2024

# Project : Predictive Modeling for Obesity Risk Classification

# Importing libraries for Data manipulation and preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing libraries for Machine learning models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Loading the training dataset
df = pd.read_csv('train.csv')

cat_columns = df[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']]
num_columns = df[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']]

for col in cat_columns:
    plt.figure(figsize=[12, 7])
    sns.countplot(x=col, data=df).set(title=f'{col} Value Distribution')
    plt.show()

for col in num_columns:
    plt.figure(figsize=[10, 7])
    sns.histplot(df[col], kde=True).set(title=f'{col} Histogram')
    plt.axvline(df[col].mean(), color='r', label='Mean')
    plt.axvline(df[col].median(), color='y', linestyle='--', label='Median')
    plt.legend()
    plt.show()

# Step 1: Encoding 'Gender' column (Binary encoding)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Step 2: Encoding 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC' (Binary encoding)
binary_columns = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in binary_columns:
    df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)

# Step 3: Ordinal encoding for 'FCVC', 'NCP', 'FAF', 'TUE'
ordinal_columns = ['FCVC', 'NCP', 'FAF', 'TUE']
df[ordinal_columns] = df[ordinal_columns].astype(int)

# Step 4: One-hot encoding 'CAEC', 'CALC', 'MTRANS' (One-hot encoding for categorical with multiple levels)
categorical_columns = ['CAEC', 'CALC', 'MTRANS']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Step 5: Feature Engineering - Calculating BMI
df['BMI'] = df['Weight'] / ((df['Height']) ** 2)

# Relationship between BMI and Obesity Risk Classes across males and females
plt.figure(figsize=(15, 5))
sns.lineplot(data=df, x='NObeyesdad', y='BMI', hue='Gender').set(title= ' BMI vs NObeyesdad')

# Distribution of Obesity Risk Classes
plt.figure(figsize=[15,10])
sns.countplot(df,x=df['NObeyesdad']).set(title= 'NObeyesdad Value Distribution')
plt.show()

# Step 6: Normalizing or standardizing continuous features (Age, Height, Weight, CH2O, BMI)
scaler = StandardScaler()
continuous_columns = ['Age', 'Height', 'Weight', 'CH2O', 'BMI']
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# Step 7: Encoding target variable 'NObeyesdad'
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])

# Displaying  label mapping for 'NObeyesdad'
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding Mapping for 'NObeyesdad':", label_mapping)

selected_columns = ['BMI', 'Weight', 'Age', 'family_history_with_overweight', 
                    'FCVC', 'CH2O', 'Gender', 'Height', 'FAF', 'NObeyesdad']

# Create a DataFrame with the selected columns
df_selected = df[selected_columns]

# Compute the correlation matrix
correlation_matrix = df_selected.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()

# Step 8: Splitting into features and target variable
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Step 9: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Defining the models to be used
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Step 11: Training and evaluating each model
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Evaluate using accuracy, recall, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store the results
    results[model_name] = {
        'Accuracy': accuracy,
        'Recall': recall,
        'F1 Score': f1
    }

# Step 12: Displaying results
results_df = pd.DataFrame(results).T
print(results_df)

# Step 13: Identifying the best model based on F1 Score from the results
best_model_name = results_df['F1 Score'].idxmax()
best_model = models[best_model_name]
print(f"Best model based on F1 Score: {best_model_name}")

# Loading the testing dataset
test_df = pd.read_csv('test.csv')

# Applying the same preprocessing steps to test_df
# Encode 'Gender'
test_df['Gender'] = test_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Encoding binary columns
for col in binary_columns:
    test_df[col] = test_df[col].apply(lambda x: 1 if x == 'yes' else 0)

# Ensuring ordinal columns are integers
test_df[ordinal_columns] = test_df[ordinal_columns].astype(int)

# One-hot encoding categorical columns
test_df = pd.get_dummies(test_df, columns=categorical_columns, drop_first=True)

# Calculating BMI
test_df['BMI'] = test_df['Weight'] / (test_df['Height'] ** 2)

# Normalizing or standardizing continuous features
test_df[continuous_columns] = scaler.transform(test_df[continuous_columns])

# Encoding target variable (if it exists in test.csv)
if 'NObeyesdad' in test_df.columns:
    test_df['NObeyesdad'] = label_encoder.transform(test_df['NObeyesdad'])
    X_test_final = test_df.drop('NObeyesdad', axis=1)
    y_test_final = test_df['NObeyesdad']
else:
    X_test_final = test_df  # if there’s no target column

print(test_df)

# Step 14: Aligning test set columns with training set columns
X_test_final = X_test_final.reindex(columns=X_train.columns, fill_value=0)

# Step 15: Evaluating the best model on the test set
y_pred_final = best_model.predict(X_test_final)

# Calculate metrics (if test set has labels)
if 'NObeyesdad' in test_df.columns:
    accuracy_final = accuracy_score(y_test_final, y_pred_final)
    recall_final = recall_score(y_test_final, y_pred_final, average='weighted')
    f1_final = f1_score(y_test_final, y_pred_final, average='weighted')

    print(f"Performance of the best model ({best_model_name}) on test.csv:")
    print(f"Accuracy: {accuracy_final:.4f}")
    print(f"Recall: {recall_final:.4f}")
    print(f"F1 Score: {f1_final:.4f}")
else:
    print("Predictions on test.csv (no labels provided):")
    print(y_pred_final)

# Step 17: Preparing the output predictions file
prediction = test_df[['id']]
prediction['NObeyesdad'] = y_pred_final
prediction['NObeyesdad'] = prediction['NObeyesdad'].map({0:'Insufficient_Weight', 1:'Normal_Weight',2:'Obesity_Type_I',3:'Obesity_Type_II',4:'Obesity_Type_III',5:'Overweight_Level_I',6:'Overweight_Level_II'})
prediction.to_csv('prediction.csv', index=False)

# Computing feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    palette = sns.color_palette("viridis", len(feature_importances))

    sns.barplot(x=feature_importances, y=feature_importances.index, palette=palette)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

from sklearn.linear_model import LogisticRegression
import numpy as np

# Training the Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

log_reg.fit(X_train, y_train)

coefficients = log_reg.coef_[0]
log_reg_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(log_reg_df)

