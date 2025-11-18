from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
import pandas as pd 
 
ds = pd.DataFrame(load_iris().data, columns=load_iris().feature_names) 
data = load_iris() 
 
X = data.data 
y = data.target 
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
) 
 
clf = SVC(kernel="linear") 
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 
 
print(f"Accuracy: {accuracy_score(y_test, y_pred)}") 
print("Classification Report:", classification_report(y_test, y_pred)) 
print("Confusion Matrix:", confusion_matrix(y_test, y_pred)) 
 
predicted_Class = clf.predict([[5.9, 3.0, 5.1, 1.8]]) 
print("Predicted flower type:", data.target_names[predicted_Class[0]])
