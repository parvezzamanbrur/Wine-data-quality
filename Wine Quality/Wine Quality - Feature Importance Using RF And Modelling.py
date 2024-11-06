import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,  precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


df= pd.read_csv("Data/WineQT.csv")
df=df.drop(['Id'], axis=1)
df['quality'] = df['quality']-3

df_category=df.copy()
df_category=df_category.sort_values(by='quality', ascending=True)

df_category["Quality Category"]=df_category["quality"]
df_category.replace({"Quality Category": {0: "Terrible", 1: "Very Poor", 2: "Poor", 3: "Good", 4: "Very Good", 5: "Excellent"}}, inplace=True)

# Feature importance
X = df.drop(['quality'], axis=1)
y = df['quality']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Training the Random Forest model
forest = RandomForestClassifier(n_estimators=500, random_state=0)
forest.fit(X_train, y_train)

# Making predictions
y_pred = forest.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy score with 500 decision trees: {accuracy:.4f}') #precision of four decimal places


feature_scores = pd.Series(forest.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(feature_scores)
feature_scores_perc = [per * 100 for per in feature_scores]


# Assuming you have trained RandomForest and other models, and you have their predictions

model_comparison={}
rf_feature_imp={}


# Random Forest performance
rf_accuracy = accuracy_score(y_test, y_pred)
rf_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, y_pred, average='weighted')

# Storing Random Forest metrics in model_comparison
model_comparison['Random Forest'] = {'Accuracy': rf_accuracy,'Precision': rf_precision,'Recall': rf_recall}

# Storing Random Forest feature importances in rf_feature_imp
rf_feature_imp = dict(zip(feature_scores.index, feature_scores_perc))

# Example printout
print("Model Comparison:", model_comparison)
print("\nRandom Forest Feature Importances:", rf_feature_imp)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_drop_quality=df.drop(['quality'], axis=1)
normal_df = scaler.fit_transform(df_drop_quality)
normal_df = pd.DataFrame(normal_df, columns = df_drop_quality.columns)
normal_df['quality']=df['quality']
X1 = normal_df.drop(['quality'], axis=1)
y1 = normal_df['quality']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

cv = KFold(n_splits=10, random_state=0, shuffle=True)

# random_state=0, n_estimators default=100
forest = RandomForestClassifier(random_state=0).fit(X_train1, y_train1)
y_pred = forest.predict(X_test1)
print('Model accuracy score : {0:0.3f}%'. format(accuracy_score(y_test1, y_pred)*100))

# random_state=0, n_estimators default=100
forest = RandomForestClassifier(random_state=0).fit(X_train1, y_train1)
y_pred = forest.predict(X_test1)
print('Model accuracy score : {0:0.4f}%'. format(accuracy_score(y_test1, y_pred)*100))

scores = cross_val_score(forest, X1, y1, scoring='accuracy', cv=cv, n_jobs=-1)
print("Mean Accuracy: %.2f%%, Standard Deviation: (%.2f%%)" % (scores.mean()*100.0, scores.std()*100.0))


X = df.drop(['quality'], axis=1)

y = df['quality']
# test_size=0.2 => %20 test, %80 train
# random_state=0 provides to have same results
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Cross-validation setup
cv = KFold(n_splits=5, random_state=0, shuffle=True)

# Parameter grid
parameters = {'min_samples_split': [2, 5], 'max_features': [1, 5], 'max_depth': [14, 24]}

# RandomForestClassifier
rf = RandomForestClassifier()

# Grid Search
clf = GridSearchCV(rf, parameters, cv=cv, n_jobs=-1)
clf.fit(X_train, y_train)

# Output best parameters
print(f'Best Hyperparameters: {clf.best_params_}')

# random_state=0, n_estimators default=100
forest = RandomForestClassifier(random_state=0).fit(X_train, y_train)
y_pred = forest.predict(X_test)
print('Model accuracy score : {0:0.2f}%'. format(accuracy_score(y_test, y_pred)*100))

scores = cross_val_score(forest, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Mean Accuracy: %.2f%%, Standard Deviation: (%.2f%%)" % (scores.mean()*100.0, scores.std()*100.0))

# random_state=0, n_estimators=500, max_depth=14
forest = RandomForestClassifier(n_estimators=500,max_depth=14, random_state=0).fit(X_train, y_train)
y_pred = forest.predict(X_test)
print('Model accuracy score : {0:0.2f}%'. format(accuracy_score(y_test, y_pred)*100))

scores = cross_val_score(forest, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Mean Accuracy: %.2f%%, Standard Deviation: (%.2f%%)" % (scores.mean()*100.0, scores.std()*100.0))

# random_state=0, n_estimators=500, max_depth=14, min_samples_split=3, max_features=1
forest = RandomForestClassifier(n_estimators=500,max_depth=14, random_state=0, min_samples_split=2, max_features=5).fit(X_train, y_train)
y_pred = forest.predict(X_test)
print('Model accuracy score : {0:0.2f}%'. format(accuracy_score(y_test, y_pred)*100))

scores = cross_val_score(forest, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Mean Accuracy: %.2f%%, Standard Deviation: (%.2f%%)" % (scores.mean()*100.0, scores.std()*100.0))

model_comparison['Random Forest']=[scores.mean()*100.0,scores.std()*100.0]
rf_feature_imp['Random Forest']=[scores.mean()*100.0,scores.std()*100.0]
print("Model Performance Comparison:", model_comparison)
print("Random Forest Feature Importances:", rf_feature_imp)

# Finding best parameters to tune randomforestclassfier() Decision tree

parameters = {'min_samples_split': [2,5], 'max_features':[1,5], 'max_depth':[14,24]}

rf = DecisionTreeClassifier()

print('Paramaters:', rf.get_params())

clf = GridSearchCV(rf, parameters, cv=10).fit(X_train, y_train)

print(f'Best Hyperparameters: {clf.best_params_}')

decision_tree=DecisionTreeClassifier(random_state=0).fit(X_train,y_train)
y_pred = decision_tree.predict(X_test)
print('Model accuracy score : {0:0.3f}%'. format(accuracy_score(y_test, y_pred)*100))

scores = cross_val_score(decision_tree, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Mean Accuracy: %.2f%%, Standard Deviation: (%.2f%%)" % (scores.mean()*100.0, scores.std()*100.0))


decision_tree=DecisionTreeClassifier(random_state=0, max_depth=14, max_features=5, min_samples_split=2).fit(X_train,y_train)
y_pred = decision_tree.predict(X_test)
print('Model accuracy score : {0:0.4f}%'. format(accuracy_score(y_test, y_pred)*100))

scores = cross_val_score(decision_tree, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Mean Accuracy: %.6f%%, Standard Deviation: (%.6f%%)" % (scores.mean()*100.0, scores.std()*100.0))

model_comparison['Decision Tree']=[scores.mean()*100.0,scores.std()*100.0]

# Initial model training and evaluation
svc = svm.SVC(random_state=0).fit(X_train, y_train)
test_accuracy = svc.score(X_test, y_test) * 100
print(f'Model accuracy score: {test_accuracy:.2f}%')


# Cross-validation setup
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(svc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

mean_accuracy = scores.mean() * 100
std_deviation = scores.std() * 100
print(f"Mean Accuracy: {mean_accuracy:.6f}%, Standard Deviation: ({std_deviation:.6f}%)")

# Generate predictions on the test set
y_pred = svc.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(cm.shape[1]), yticklabels=range(cm.shape[0]))
plt.title('Multi-Class Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Unique values in y_test:", set(y_test))
print("Unique values in y_pred:", set(y_pred))

## REF Code
# Separate features (X) and target variable (y)
#X = df.drop(columns=["quality"], axis = 1)
#y = df["quality"]

# Split the data into training and testing sets for binary classification
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

# Standardization process
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Initialize and train the model
#svc_model = SVC()
#svc_model.fit(X_train, y_train)

# Make predictions on the test set
#y_pred = svc_model.predict(X_test)


# Define parameter grid for SVC
#param_grid = {
#   'C': [0.1, 1, 10, 100],
#   'kernel': ['linear', 'rbf', 'poly'],
#   'gamma': ['scale', 'auto'],
#   'degree': [2, 3, 4]  # Only relevant for 'poly' kernel
#}

# GridSearchCV to find the best hyperparameters
#grid_search = GridSearchCV(svm.SVC(random_state=0), param_grid, scoring='accuracy', cv=cv, n_jobs=-1)

# Fit grid search
#grid_search.fit(X_train, y_train)

# Best parameters and model accuracy
#best_params = grid_search.best_params_
#print(f"Best Hyperparameters: {best_params}")
#best_score = grid_search.best_score_ * 100
#print(f"Best Cross-Validated Accuracy: {best_score:.2f}%")
###----###

# Train SVC model
svc = svm.SVC(random_state=0).fit(X_train, y_train)

# Accuracy on the test set
test_accuracy = svc.score(X_test, y_test) * 100
print(f'Model accuracy score: {test_accuracy:.2f}%')

# Cross-validation setup (if not already defined)
cv = StratifiedKFold(n_splits=6, random_state=0, shuffle=True)

# Perform cross-validation
scores = cross_val_score(svc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# Print cross-validation results
mean_accuracy = scores.mean() * 100
std_deviation = scores.std() * 100
print(f"Mean Accuracy: {mean_accuracy:.2f}%, Standard Deviation: ({std_deviation:.2f}%)")

# Train SVM with linear kernel
svc = svm.SVC(kernel='linear', random_state=42).fit(X_train, y_train)

# Accuracy on test set
test_accuracy = svc.score(X_test, y_test) * 100
print(f'Model accuracy score: {test_accuracy:.2f}%')

# Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

# Cross-validation scores
scores = cross_val_score(svc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# Mean and standard deviation of accuracy
mean_accuracy = scores.mean() * 100
std_deviation = scores.std() * 100
print(f"Mean Accuracy: {mean_accuracy:.2f}%, Standard Deviation: ({std_deviation:.2f}%)")

model_comparison['SVC']=[scores.mean()*100.0,scores.std()*100.0]

# Assume model_comparison is already defined and filled with model results
df_comparison = pd.DataFrame.from_dict(model_comparison).T
df_comparison.columns = ['Mean Accuracy', 'Standard Deviation']
df_comparison = df_comparison.sort_values('Mean Accuracy', ascending=True)

# Display with gradient in Jupyter
df_comparison.style.background_gradient(cmap='Blues')

# Print the DataFrame
print(df_comparison)

#INCREASING ACCURACY - DROP LEAST IMPORTANT FEATURES
# Define cross-validation
cv = KFold(n_splits=10, random_state=0, shuffle=True)

# First feature removal: 'residual sugar'
X_train1 = X_train.drop(['residual sugar'], axis=1)
X_test1 = X_test.drop(['residual sugar'], axis=1)
scores = cross_val_score(RandomForestClassifier(n_estimators=500, max_depth=14,
                                                random_state=0, min_samples_split=2,
                                                max_features=1),
                         X_train1, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print("Without 'residual sugar' - Mean Accuracy: %.2f%%, Standard Deviation: (%.2f%%)" % (scores.mean()*100.0, scores.std()*100.0))

# Second feature removal: 'free sulfur dioxide'
X_train1 = X_train1.drop(['free sulfur dioxide'], axis=1)
X_test1 = X_test1.drop(['free sulfur dioxide'], axis=1)
scores = cross_val_score(RandomForestClassifier(n_estimators=500, max_depth=14,
                                                random_state=0, min_samples_split=2,
                                                max_features=1),
                         X_train1, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print("Without 'residual sugar' & 'free sulfur dioxide' - Mean Accuracy: %.2f%%, Standard Deviation: (%.2f%%)" % (scores.mean()*100.0, scores.std()*100.0))
