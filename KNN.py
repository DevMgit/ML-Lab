from sklearn.datasets import load_iris 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
 
iris = load_iris() 
data = iris.data 
target = iris.target 
 
X_train, X_test, Y_train, Y_test = train_test_split(
    data, target, test_size=0.3
) 
 
clf = KNeighborsClassifier(n_neighbors=5) 
clf.fit(X_train, Y_train) 
 
predict = clf.predict(X_test) 
 
print(f"Scikit-learn KNN classifier accuracy: {accuracy_score(Y_test, predict)}") 
print(classification_report(Y_test, predict))
