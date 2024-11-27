#!/usr/bin/env python
# coding: utf-8
# Import necessary libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("/Users/jeevandeep/Downloads/myenv/Employee Data.csv")
print(data.head())
print(data.columns)


### Question 1: Scaling and Encoding
# 1. Scale interval variables
scaler = MinMaxScaler()
interval_vars = ['JoiningYear', 'Age', 'ExperienceInCurrentDomain']
data[interval_vars] = scaler.fit_transform(data[interval_vars])
# 2. Perform dummy encoding for categorical variables, convert booleans to integers
data_encoded = pd.get_dummies(
    data, columns=['Education', 'City', 'Gender', 'EverBenched', 'PaymentTier'], drop_first=True
)
# Ensure all boolean columns are converted to integers
data_encoded = data_encoded.astype({col: 'int' for col in data_encoded.select_dtypes(include=['bool']).columns})
# Total number of columns after encoding (including target variable)
total_columns_after_encoding = data_encoded.shape[1]
print("Total columns after encoding:", total_columns_after_encoding)
# Save the encoded dataset (optional)
#data_encoded.to_csv('Employee_Data_Encoded.csv', index=False)


### Question 2: Correlation Matrix
correlation_matrix = data_encoded.corr()
# Plot the heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Matrix of Encoded Variables")
plt.show()
# Extract correlations of the specified variables with the given options
variables_to_check = ['Gender_Male', 'City_Pune', 'PaymentTier_3', 'JoiningYear']
given_options = ['Education_Masters', 'PaymentTier_2', 'City_New Delhi', 'LeaveOrNot']
# Calculate correlations
correlation_with_given_options = correlation_matrix.loc[variables_to_check, given_options]
# Find the most correlated variable from the given options for each specified variable
most_correlated = correlation_with_given_options.idxmax(axis=1)
correlation_values = correlation_with_given_options.max(axis=1)
# Print the results
print("Most correlated variables from the given options:")
for var, corr_var, corr_value in zip(variables_to_check, most_correlated, correlation_values):
    print(f"{var} -> {corr_var} (Correlation: {corr_value:.2f})")


### Question 3: Variance Inflation Factor (VIF)
# Drop the dependent variable ('LeaveOrNot') for VIF calculation
independent_vars = data_encoded.drop(columns=['LeaveOrNot'])
# Calculate VIF
vif_data = pd.DataFrame({
    'Variable': independent_vars.columns,
    'VIF': [variance_inflation_factor(independent_vars.values, i) for i in range(independent_vars.shape[1])]
}).sort_values(by='VIF', ascending=False)
print(vif_data)


### Question 4: Removing High VIF Variables Iteratively
# Iterate until all VIF values are <= 10
threshold_vif = 10
while vif_data["VIF"].iloc[0] > threshold_vif:
    # Remove the variable with the highest VIF
    variable_to_remove = vif_data["Variable"].iloc[0]
    print(f"Removing variable with high VIF: {variable_to_remove}")
    independent_vars = independent_vars.drop(columns=[variable_to_remove])
    
    # Recalculate VIF
    vif_data = pd.DataFrame({
        'Variable': independent_vars.columns,
        'VIF': [variance_inflation_factor(independent_vars.values, i) for i in range(independent_vars.shape[1])]
    }).sort_values(by='VIF', ascending=False)
highest_vif_after_removal = round(vif_data["VIF"].iloc[0], 3)
print("Highest remaining VIF:", highest_vif_after_removal)


### Question 5: Proportion of 1's in Target Variable
target_proportion = data_encoded['LeaveOrNot'].mean()
print(f"Proportion of 1's in target variable: {target_proportion:.3f}")


### Question 7: Variables Most Correlated with Target
variables_of_interest = ['JoiningYear', 'Age', 'EverBenched_Yes', 'ExperienceInCurrentDomain']
correlations_with_target = correlation_matrix['LeaveOrNot'][variables_of_interest].sort_values(ascending=False)
print("Variables in descending order of correlation with LeaveOrNot:")
print(correlations_with_target)


## Question 8: Model 1 with the variable most correlated with the target
#Define the target variable and identify the variable most correlated with it
target = 'LeaveOrNot'
correlation_matrix = data_encoded.corr()
correlations_with_target = correlation_matrix[target].sort_values(ascending=False)
# Step 1: Identify the variable most correlated with the target variable
highest_corr_variable = correlations_with_target.index[1]  # Skip the target variable itself
print(f"The variable most correlated with '{target}' is: {highest_corr_variable}")
# Step 2: Perform a 70/30 training/validation split
X_model1 = data_encoded[[highest_corr_variable]]
y_model1 = data_encoded[target]
X_train_model1, X_val_model1, y_train_model1, y_val_model1 = train_test_split(
    X_model1, y_model1, test_size=0.3, random_state=0
)
# Step 3: Train logistic regression model using statsmodels
X_train_const_model1 = sm.add_constant(X_train_model1)  # Add intercept for statsmodels
model_1 = sm.Logit(y_train_model1, X_train_const_model1).fit()
# Display the model summary
print("Model 1 Summary:")
print(model_1.summary())
# Step 4: Make predictions on the validation set
X_val_const_model1 = sm.add_constant(X_val_model1)  # Add intercept for validation set
y_pred_model1 = model_1.predict(X_val_const_model1) > 0.5  # Use 0.5 threshold
# Step 5: Plot Confusion Matrix
cm_model1 = confusion_matrix(y_val_model1, y_pred_model1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_model1, display_labels=["Not Leaving", "Leaving"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Model 1 (One Variable)")
plt.show()
# Step 6: Evaluate Model 1
accuracy_model1 = accuracy_score(y_val_model1, y_pred_model1)
f1_model1 = f1_score(y_val_model1, y_pred_model1)
roc_auc_model1 = roc_auc_score(y_val_model1, model_1.predict(X_val_const_model1))
# Display metrics
print(f"Model 1 Accuracy: {accuracy_model1:.3f}")
print(f"Model 1 F1 Score: {f1_model1:.3f}")
print(f"Model 1 ROC AUC Score: {roc_auc_model1:.3f}")


### Question 9: Logistic Regression (Model 2) with Only Significant Variables (pseudo R^2 concept ardham kaale)
# Step 1: Prepare the dataset with all independent variables
X_model2 = data_encoded.drop(columns=[target])
y_model2 = data_encoded[target]
# Step 2: Perform a 70/30 training/validation split
X_train_model2, X_val_model2, y_train_model2, y_val_model2 = train_test_split(
    X_model2, y_model2, test_size=0.3, random_state=0
)
# Step 3: Train logistic regression model with all variables
X_train_const_model2 = sm.add_constant(X_train_model2)  # Add intercept
model_2 = sm.Logit(y_train_model2, X_train_const_model2).fit()
# Display the model summary
print("Model 2 Summary:")
print(model_2.summary())
# Step 4: Categorize variables into significant and non-significant
p_values = model_2.pvalues
significant_vars = p_values[p_values < 0.05].index.tolist()
non_significant_vars = p_values[p_values >= 0.05].index.tolist()
# Display significant and non-significant variables
print("Significant Variables:", significant_vars)
print("Non-Significant Variables:", non_significant_vars)


### Question 10:
# 
# # Step 1: Identify significant variables from the previous model
significant_vars = [var for var in model_2.pvalues.index if model_2.pvalues[var] < 0.05]
# Check if the intercept ('const') is significant
if 'const' in significant_vars:
    print("Intercept (const) is significant and will be retained.")
else:
    print("Intercept (const) is not significant and will be excluded.")
    significant_vars = [var for var in significant_vars if var != 'const']  # Exclude intercept if not significant
# Step 2: Prepare training and validation sets with only significant variables
X_train_significant = X_train_model2[significant_vars]
X_val_significant = X_val_model2[significant_vars]
# Explicitly skip adding the constant if the intercept is not significant
if 'const' in significant_vars:
    X_train_significant_const = sm.add_constant(X_train_significant, has_constant='add')
    X_val_significant_const = sm.add_constant(X_val_significant, has_constant='add')
    print("Constant added to the model.")
else:
    X_train_significant_const = X_train_significant
    X_val_significant_const = X_val_significant
    print("Constant excluded from the model.")
# Step 3: Train a logistic regression model with only significant variables
model_significant = sm.Logit(y_train_model2, X_train_significant_const).fit()
# Display the summary of the reduced model
print("Reduced Model Summary:")
print(model_significant.summary())
# Step 4: Compute the pseudo R-squared value
# Null model: Log-likelihood of a model predicting the mean of the target variable
null_model_loglik = sm.Logit(y_val_model2, np.ones(len(y_val_model2))).fit(disp=False).llf
# Full model: Log-likelihood of the reduced model on validation data
reduced_model_loglik = model_significant.llf
# Compute pseudo R-squared
pseudo_r2 = round(1 - (reduced_model_loglik / null_model_loglik), 3)
print(f"Pseudo R-squared for the reduced model on validation set: {pseudo_r2}")


###Question 13:
# Calculate training accuracy for Scikit-Learn model
y_pred_train_sklearn = log_reg_sklearn.predict(X_train_significant)
accuracy_train = round(accuracy_score(y_train_model2, y_pred_train_sklearn), 3)
print(f"Training Set Accuracy: {accuracy_train}")
print(f"Validation Set Accuracy: {accuracy_val}")


### Question 14:
from sklearn.metrics import precision_score, recall_score
# Compute precision, recall, and F1 score
precision_val = round(precision_score(y_val_model2, y_pred_val_sklearn), 3)
recall_val = round(recall_score(y_val_model2, y_pred_val_sklearn), 3)
f1_val = round(f1_score(y_val_model2, y_pred_val_sklearn), 3)
# Display metrics
print(f"Precision: {precision_val}")
print(f"Recall: {recall_val}")
print(f"F1-Score: {f1_val}")
print(f"ROC-AUC Score: {roc_auc_model1:.3f}")


### Question 15:
from sklearn.model_selection import train_test_split
# Define the features for Model 3
features_model3 = ['PaymentTier_3', 'Age', 'Gender_Male']
# Prepare the data
X_model3 = data_encoded[features_model3]
y_model3 = data_encoded['LeaveOrNot']
# Perform a 70/30 train-validation split
X_train_model3, X_val_model3, y_train_model3, y_val_model3 = train_test_split(
    X_model3, y_model3, test_size=0.3, random_state=0
)
# Perform logistic regression using statsmodels on the training set (without the intercept)
model_3_no_const = sm.Logit(y_train_model3, X_train_model3).fit()
# Display the summary of the model
print("Model 3 Summary (Without Intercept):")
print(model_3_no_const.summary())
# Extract p-values and find the maximum
p_values_model3_no_const = model_3_no_const.pvalues
max_p_value_model3_no_const = round(p_values_model3_no_const.max(), 3)
print(f"Maximum p-value for Model 3 (Without Intercept): {max_p_value_model3_no_const}")
# Evaluate the model on the validation set
y_pred_model3_no_const = model_3_no_const.predict(X_val_model3) > 0.5  # Classify as 1 if probability > 0.5
# Generate the confusion matrix for the validation set
cm_model3_no_const = confusion_matrix(y_val_model3, y_pred_model3_no_const)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_model3_no_const, display_labels=["Not Leave", "Leave"])
# Plot the confusion matrix
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Model 3 (Validation Set, Without Intercept)")
plt.show()


### Question 16: TNR for All Models
# Define a function to calculate TNR from the confusion matrix
def calculate_tnr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)
# Model 1 TNR
tnr_model1 = calculate_tnr(y_val_model1, y_pred_model1)
# Model 2 TNR
y_pred_model2 = model_significant.predict(X_val_significant_const) > 0.5
tnr_model2 = calculate_tnr(y_val_model2, y_pred_model2)
# Model 3 TNR
y_pred_model3 = model_3.predict(X_model3_const) > 0.5
tnr_model3 = calculate_tnr(y_model3, y_pred_model3)
# Compare TNRs
print(f"Model 1 TNR: {tnr_model1:.3f}")
print(f"Model 2 TNR: {tnr_model2:.3f}")
print(f"Model 3 TNR: {tnr_model3:.3f}")
# Select the best model based on TNR
if tnr_model1 > tnr_model2 and tnr_model1 > tnr_model3:
    best_model = "Model 1"
elif tnr_model2 > tnr_model1 and tnr_model2 > tnr_model3:
    best_model = "Model 2"
else:
    best_model = "Model 3"

print(f"The best model based on TNR is: {best_model}")

