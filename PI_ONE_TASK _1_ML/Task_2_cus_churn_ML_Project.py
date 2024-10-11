#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries and Loading Data
# 
# 
# 
# 

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'E:\PI_ONE_TASK _1_ML\csv\Telco-Customer-Churn.csv'
cus_churn = pd.read_csv(file_path)


# ## Data Exploration and Preprocessing
# 
# 

# In[2]:


# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(cus_churn.head())

# Check for missing values
print("\nMissing values in each column:")
print(cus_churn.isnull().sum())

# Drop the 'customerID' column and encode the target variable
cus_churn.drop(columns=['customerID'], inplace=True)
le = LabelEncoder()
cus_churn['Churn'] = le.fit_transform(cus_churn['Churn'])

# Encode categorical variables
cus_churn = pd.get_dummies(cus_churn, drop_first=True)


# ## Feature Engineering and Splitting Data
# 
# 

# In[3]:


# Define features and target variable
features = cus_churn.drop('Churn', axis=1)  # Features
target = cus_churn['Churn']                # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ## Feature Scaling

# In[4]:


# Initialize a StandardScaler object
scaler = StandardScaler()

# Fit and transform the training features
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing features
X_test_scaled = scaler.transform(X_test)


# ## Model Training and Evaluation

# In[5]:


# Initialize a DecisionTreeClassifier object
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = dt_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nDecision Tree Model Accuracy: {accuracy:.2f}")

# Print confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ## Visualization

# In[8]:


# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# ### If you want to use Logistic Regression instead, you can replace the DecisionTreeClassifier with LogisticRegression from sklearn.linear_model:

# In[7]:


# from sklearn.linear_model import LogisticRegression

# # Initialize a LogisticRegression object
# log_model = LogisticRegression(random_state=42)

# # Train the model
# log_model.fit(X_train_scaled, y_train)

# # Make predictions
# y_pred = log_model.predict(X_test_scaled)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nLogistic Regression Model Accuracy: {accuracy:.2f}")

# # Print confusion matrix and classification report
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))


# In[ ]:




