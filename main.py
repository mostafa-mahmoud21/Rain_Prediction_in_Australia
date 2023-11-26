# Import the required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, \
    log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

# Importing the Dataset
df = pd.read_csv("Weather_Data.csv")

# Data preprocessing
# one hot encoding (convert categorical variables to binary variables)
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0, 1], inplace=True)

# Training Data and Test Data
df_sydney_processed.drop('Date', axis=1, inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
x = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
y = df_sydney_processed['RainTomorrow']

# Linear Regression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)
y_pred = LinearReg.predict(x_test)
predictions = LinearReg.predict(x_test)
print("predictions:", predictions)
# Convert predictions to binary classes based on the threshold
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)
binary_y_test = (y_test > threshold).astype(int)
print(binary_predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)
laccuracy = accuracy_score(y_test, binary_predictions)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r_squared}")
print(f"accuracy: {laccuracy}")

# KNN
KNN = KNeighborsRegressor(n_neighbors=4)
KNN.fit(x_train, y_train)
knn_predictions = KNN.predict(x_test)

KNN_Accuracy_Score = accuracy_score(binary_y_test, binary_predictions)
KNN_Jaccard = jaccard_score(binary_y_test, binary_predictions)
KNN_F1_Score = f1_score(binary_y_test, binary_predictions)
knn_accuracy = accuracy_score(binary_y_test, binary_predictions)
print(f"KNN Accuracy Score: {KNN_Accuracy_Score}")
print(f"KNN Jaccard Index: {KNN_Jaccard}")
print(f"KNN F1 Score: {KNN_F1_Score}")
print(f"KNN accuracy: {knn_accuracy}")

# Decision Tree
Tree = DecisionTreeRegressor()
Tree.fit(x_train, y_train)
tree_predictions = Tree.predict(x_test)
print(tree_predictions)
tree_Jaccard = jaccard_score(y_test, tree_predictions)
tree_F1_Score = f1_score(y_test, tree_predictions)
tree_accuracy = accuracy_score(y_test, tree_predictions)
print(f"tree Jaccard Index: {tree_Jaccard}")
print(f"tree F1 Score: {tree_F1_Score}")
print(f"tree_accuracy: {tree_accuracy}")

# Logistic regression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
LR = LogisticRegression(solver='liblinear')
LR.fit(x_train, y_train)
lr_predictions = LR.predict(x_test)
# Use the trained Logistic Regression model to get probability estimates for each class
predict_proba = LR.predict_proba(x_test)
print("predictions:", lr_predictions)
print("predictions prob:", predict_proba)
# Convert probabilities to binary predictions using a threshold
threshold_lr = 0.5
binary_predictions_lr = (predict_proba[:, 1] > threshold_lr).astype(int)
LR_Accuracy_Score = accuracy_score(y_test, binary_predictions_lr)
LR_JaccardIndex = jaccard_score(y_test, binary_predictions_lr)
LR_F1_Score = f1_score(y_test, binary_predictions_lr)
LR_Log_Loss = log_loss(y_test, predict_proba)
print(f"Logistic Regression Accuracy Score: {LR_Accuracy_Score}")
print(f"Logistic Regression Jaccard Index: {LR_JaccardIndex}")
print(f"Logistic Regression F1 Score: {LR_F1_Score}")
print(f"Logistic Regression Log Loss: {LR_Log_Loss}")

# SVM
SVM = SVC()
SVM.fit(x_train, y_train)
svm_predictions = SVM.predict(x_test)
print("svm_predictions:", svm_predictions)
SVM_Accuracy_Score = accuracy_score(y_test, svm_predictions)
SVM_JaccardIndex = jaccard_score(y_test, svm_predictions)
SVM_F1_Score = f1_score(y_test, svm_predictions)
print(f"SVM Accuracy Score: {SVM_Accuracy_Score}")
print(f"SVM Jaccard Index: {SVM_JaccardIndex}")
print(f"SVM F1 Score: {SVM_F1_Score}")

results = {
    'Model': ['Linear Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Logistic Regression',
              'Support Vector Machine'],
    'Accuracy': [laccuracy, knn_accuracy, tree_accuracy, LR_Accuracy_Score, SVM_Accuracy_Score],
    'Jaccard Index': [0, KNN_Jaccard, tree_Jaccard, LR_JaccardIndex, SVM_JaccardIndex],
    'F1 Score': [0, KNN_F1_Score, tree_F1_Score, LR_F1_Score, SVM_F1_Score],
    'Log Loss': [0, 0, 0, LR_Log_Loss, 0]
}
results_df = pd.DataFrame(results)
print(results_df)
