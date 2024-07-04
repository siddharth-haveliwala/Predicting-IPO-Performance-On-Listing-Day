#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Import necessary libraries
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import boxcox
from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.cluster.hierarchy import dendrogram, linkage
from bayes_opt import BayesianOptimization
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import pickle


# In[22]:


# Read the input data from csv
df = pd.read_csv('/Users/siddharthhaveliwala/dataglacierrepos/week-4/IPO_Prediction_week3/data/ipo_report_listing_day_gain.csv')


# In[23]:


# Summary of dataframe df
df.info()


# In[24]:


df.head(5)


# In[25]:


df.columns, df.shape


# In[26]:


df.isnull().sum()


# In[27]:


# Remove unnecessary columns and save new dataframe udf
udf = df.drop(['P/E Ratio', 'EMP', 'Open Price', 'Low Price', 'High Price', 'Close Price', 'Issuer Company', 'Listing Date'], axis=1)

# Remove NA or missing values
udf = udf.dropna()


# In[28]:


udf.info()


# In[29]:


udf.describe()


# In[30]:


# Remove bad data from dataframe udf
bad_record = 'R'
rows_to_remove = udf[udf['QIB'] == bad_record].index
print(rows_to_remove) # No bad data


# In[31]:


# Standardize the dataset 
scaler = StandardScaler()
numerical_cols = udf.select_dtypes(include=['int64', 'float64']).columns

udf[numerical_cols] = scaler.fit_transform(udf[numerical_cols])


# In[32]:


#type(udf['% Change'])


# In[33]:


# Generate the Response variable (IPO Listing positive (1) or negative(0))
udf['target'] = udf['% Change'].apply(lambda x: 1 if x > 0 else 0)

# Drop the '% Change' column
udf = udf.drop(['% Change'], axis=1)


# In[34]:


# Compute correlations with target variable
correlations = udf.corr()
target_correlation = correlations['target'].drop('target')

# Select features with significant correlation
high_corr_features = target_correlation[abs(target_correlation) > 0.2].index.tolist()


# In[35]:


# Separate features and target
X = udf.drop('target', axis=1)
y = udf['target']

# Train a random forest regressor to get feature importance
model = RandomForestRegressor()
model.fit(X, y)

importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
feature_importances


# In[36]:


# Plot histograms for each feature
features = udf.drop('target', axis = 1)
features = features.columns
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(len(features)//3 + 1, 3, i+1)
    sns.histplot(udf[feature], kde=True)
    plt.title(f'Histogram for : {feature}')
    plt.tight_layout()

plt.show()


# In[37]:


plt.figure(figsize=(20, 15))

for i, feature in enumerate(features):
    plt.subplot(len(features)//3 + 1, 3, i+1)
    sns.boxplot(x=udf[feature])
    plt.title(f'Boxplot for: {feature}')
    plt.tight_layout()

plt.show()


# In[38]:


# Load the udf dataframe into final fd dataframe
fd = udf.copy()

# List of skewed columns based on the histograms to apply transoformations
skewed_cols = ['Issue Price', 'Lot Size', 'Issue Price (Rs Cr)', 'QIB', 'NII', 'RII', 'TOTAL']

for col in skewed_cols:
    if (fd[col] <= 0).any():
        # Apply a log transformation with an offset to handle zero and negative values
        # The offset of 1 is used to ensure all values are positive
        fd[col] = np.log1p(fd[col] + 1)
    else:
        # Apply the Box-Cox transformation since all values are positive
        fd[col], fitted_lambda = boxcox(fd[col])


# In[39]:


# Separate features and target
X = fd.drop('target', axis=1)
y = fd['target']

# Train a random forest regressor to get feature importance
model = RandomForestRegressor()
model.fit(X, y)
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
#print(feature_importances)


# In[40]:


# Convert the target column into Factor type
fd['target'] = fd['target'].factorize()[0]

# Check for available target values
#print(fd['target'].value_counts())


# In[41]:


# Visualizing the distribution of the target
sns.countplot(x='target', data=fd)
plt.title('Distribution of Target Classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


# In[42]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42, test_size=0.20, shuffle=True)


# In[56]:


X_train.shape, X_test.shape, X_train.columns


# In[64]:


X_train[0:1]


# In[44]:


# Fit a Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, Y_train)

# Fit Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, Y_train)

# Fit SVM Classifier
svm_model = SVC(kernel='linear', random_state=42, max_iter=100)
svm_model.fit(X_train, Y_train)

decision_tree_predictions = decision_tree.predict(X_test)
random_forest_predictions = random_forest.predict(X_test)
svm_predictions = svm_model.predict(X_test)


# In[45]:


# Plot the Accuracy for fitted models (without Bayesian Optimization)
decision_tree_accuracy = accuracy_score(Y_test, decision_tree_predictions)
random_forest_accuracy = accuracy_score(Y_test, random_forest_predictions)
svm_accuracy = accuracy_score(Y_test, svm_predictions)

accuracies = {
    'Decision Tree': decision_tree_accuracy,
    'Random Forest': random_forest_accuracy,
    'SVM': svm_accuracy
}

#print(accuracies)
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim([0, 1])  # Accuracy ranges between 0 and 1
plt.show()


# In[46]:


# Create dtree_cv function for Decision Tree
def dtree_cv(max_depth, min_samples_split, min_samples_leaf):
    estimator = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    # Implement 5-fold cross-validation on the dataset
    cval = cross_val_score(estimator, X_train, Y_train, scoring='accuracy', cv=5)
    return cval.mean()

# Define Hyperparameter space
params = {
    'max_depth': (3, 10),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}

# Implement Bayesian Optimization
optimizer = BayesianOptimization(f=dtree_cv, pbounds=params, random_state=42)
optimizer.maximize(init_points=10, n_iter=25)

# Extract the best hyperparameters
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

# Train Decision Tree classifier with selected best hyperparameters
best_tree = DecisionTreeClassifier(**best_params, random_state=42)
best_tree.fit(X_train, Y_train)

predictions = best_tree.predict(X_test)

accuracy_dtree = accuracy_score(Y_test, predictions)
#print(f"Accuracy of the best Decision Tree: {accuracy_dtree:.4f}")


# In[47]:


# Define rf_cv function to be maximized and include hyperparameters for tuning for Random Forest
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Apply 5-fold cross-validation
    cval = cross_val_score(estimator, X_train, Y_train, scoring='accuracy', cv=5)

    return cval.mean()

params = {
    'n_estimators': (10, 250),
    'max_depth': (5, 30),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=rf_cv, pbounds=params, random_state=42)
optimizer.maximize(init_points=10, n_iter=25)

# Extract the best hyperparameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

# Train the model with selected best hyperparameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, Y_train)

predictions = best_rf.predict(X_test)

accuracy_rf = accuracy_score(Y_test, predictions)
#print(f"Accuracy of the best RandomForest: {accuracy_rf:.4f}")


# In[48]:


# Initialize the AdaBoost Classifier
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)

ada_clf.fit(X_train, Y_train)
ada_predictions = ada_clf.predict(X_test)

ada_accuracy = accuracy_score(Y_test, ada_predictions)

# Initialize the CatBoost Classifier
cat_clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, verbose=False, random_state=42)

cat_clf.fit(X_train, Y_train)
cat_predictions = cat_clf.predict(X_test)

cat_accuracy = accuracy_score(Y_test, cat_predictions)

# Initialize the XGBoost Classifier
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

xgb_clf.fit(X_train, Y_train)
xgb_predictions = xgb_clf.predict(X_test)

xgb_accuracy = accuracy_score(Y_test, xgb_predictions)

# Print the Boositng methods accuracies
#print(f"Accuracy of AdaBoost Classifier: {ada_accuracy:.4f}")
#print(f"Accuracy of CatBoost Classifier: {cat_accuracy:.4f}")
#print(f"Accuracy of XGBoost Classifier: {xgb_accuracy:.4f}")


# In[49]:


# Define logistic_cv function for Logistic Regression
def logistic_cv(C, l1_ratio):

    estimator = LogisticRegression(
        C=C,
        penalty='elasticnet',
        l1_ratio=l1_ratio,
        solver='saga',
        max_iter=1000,
        random_state=42
    )

    cval = cross_val_score(estimator, X_train, Y_train, scoring='accuracy', cv=5)
    return cval.mean()


params = {
    'C': (0.001, 10),  # Regularization parameter
    'l1_ratio': (0, 1)  # Balance of L1 and L2 regularization
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=logistic_cv, pbounds=params, random_state=42)
optimizer.maximize(init_points=10, n_iter=25)

# Extract the best hyperparameters
best_params = optimizer.max['params']

# Train the model with the best hyperparameters
best_logistic = LogisticRegression(
    C=best_params['C'],
    penalty='elasticnet',
    l1_ratio=best_params['l1_ratio'],
    solver='saga',
    max_iter=1000,
    random_state=42
)
best_logistic.fit(X_train, Y_train)

# Predictions
predictions = best_logistic.predict(X_test)

# Evaluation
accuracy_log = accuracy_score(Y_test, predictions)
#print(f"Accuracy of the best Logistic Regression: {accuracy_log:.4f}")


# In[50]:


# Define svm_cv function to be maximized and include hyperparameters for tuning for SVM model
def svm_cv(C, gamma):
    estimator = SVC(
        C=C,
        gamma=gamma,
        random_state=42
    )
    # Apply 5-fold cross-validation
    cval = cross_val_score(estimator, X_train, Y_train, scoring='accuracy', cv=5)
    return cval.mean()

params = {
    'C': (0.001, 100),
    'gamma': (0.0001, 5)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=svm_cv, pbounds=params, random_state=42)
optimizer.maximize(init_points=10, n_iter=25)

# Extract the best hyperparameters
best_params = optimizer.max['params']

# Train SVM model with selected best hyperparameters
best_svm = SVC(**best_params, random_state=42)
best_svm.fit(X_train, Y_train)

predictions = best_svm.predict(X_test)

accuracy_svm = accuracy_score(Y_test, predictions)
#print(f"Accuracy of the best SVM: {accuracy_svm:.4f}")


# In[51]:


# Fit Neural Network model
nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                         alpha=0.01, batch_size=32, learning_rate='constant',
                         learning_rate_init=0.01, max_iter=300, random_state=42)

nn_model.fit(X_train, Y_train)

nn_predictions = nn_model.predict(X_test)


accuracy_nn = accuracy_score(Y_test, nn_predictions)
#print(f"Accuracy of the Neural Network: {accuracy_nn:.4f}")


# In[52]:


accuracies_b = {
    'Model': ['Logistic Regression','Decision Tree Classifier','Random Forest','AdaBoost Classifier', 'CatBoost Classifier', 'XGBoost Classifier', 'SVM Classifier', 'Neural Network Model'],
    'Accuracy': [accuracy_log, accuracy_dtree, accuracy_rf, ada_accuracy, cat_accuracy, xgb_accuracy, accuracy_svm, accuracy_nn]
}

df_acc = pd.DataFrame(accuracies_b)

print(df_acc)


# In[54]:


pickle.dump(best_logistic, open('model.pkl', 'wb'))


# In[55]:


best_model = pickle.load(open('model.pkl', 'rb'))


# In[80]:


validate_record = [[195, 72, 171, 15.2, 23.7, 68.5, 118]]
test_web_results = best_model.predict(validate_record)
if test_web_results == 1:
    print("Success.")
else:
    print("Failure.")


# In[ ]:




