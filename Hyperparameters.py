import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
 
df = pd.read_csv("path")   # replace with actual path 
 
label_encoders = {} 
for column in df.columns: 
    le = LabelEncoder() 
    df[column] = le.fit_transform(df[column]) 
    label_encoders[column] = le 
 
X = df.drop('Play', axis=1) 
y = df['Play'] 
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
) 
 
hyperparameters = [ 
    {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}, 
    {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 1}, 
    {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 2}, 
    {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 4}, 
] 
 
best_accuracy = 0 
best_params = None 
best_tree = None 
 
for params in hyperparameters: 
    tree = DecisionTreeClassifier(**params, random_state=42) 
    tree.fit(X_train, y_train) 
    y_pred = tree.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    print(f"Parameters: {params}, Accuracy: {accuracy:.4f}") 
    if accuracy > best_accuracy: 
        best_accuracy = accuracy 
        best_params = params 
        best_tree = tree 
 
print(f"\nBest Parameters: {best_params}, Best Accuracy: {best_accuracy:.4f}") 
 
plt.figure(figsize=(12, 8)) 
plot_tree( 
    best_tree, 
    filled=True, 
    feature_names=X.columns, 
    class_names=[str(cls) for cls in label_encoders['Play'].classes_], 
    rounded=True 
) 
plt.title("Best Decision Tree") 
plt.show() 
 
y_pred_best = best_tree.predict(X_test) 
 
print("\nBest Decision Tree - Classification Report:") 
print(classification_report(y_test, y_pred_best)) 
 
print("Best Decision Tree - Confusion Matrix:") 
print(confusion_matrix(y_test, y_pred_best)) 
 
print("Best Decision Tree - Accuracy Score:") 
print(accuracy_score(y_test, y_pred_best))
