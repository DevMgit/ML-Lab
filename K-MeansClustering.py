import pandas as pd  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
import seaborn as sns  
 
data = pd.read_csv(r"D:\ML lab\iris.csv")  
x = data.drop(columns=["species"])  
y = data["species"]  
 
scaler = StandardScaler()  
x_scaled = scaler.fit_transform(x)  
 
kmeans = KMeans(n_clusters=3, random_state=42)  
kmeans.fit(x_scaled)  
labels = kmeans.labels_  
print("Cluster Labels:", labels)  
 
label_map = {}  
for cluster in range(3):  
    species_in_cluster = y[labels == cluster]  
    most_common_species = species_in_cluster.mode()[0]  
    label_map[cluster] = most_common_species  
 
print("Cluster to Species Mapping:", label_map)  
predicted_species = [label_map[label] for label in labels]  
correct_labels = sum(y == predicted_species)  
 
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))  
print("Accuracy score: {0:0.2f}".format(correct_labels / float(y.size)))
