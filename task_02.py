import pandas as pd

# Dataset URL
dataset_url = 'https://www.kaggle.com/datasets/vjchoudhary/customer-segmentation-tutorial-in-python'

# Reading data from a CSV file
data = pd.read_csv('Mall_Customers.csv')

# Displaying the first few rows of the dataset
data.head()

# Checking for missing values
print(data.isnull().sum())

# Dropping the 'CustomerID' column
data = data.drop(columns=['CustomerID'])

# Converting values in the 'Gender' column to numeric
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Normalizing the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Converting back to DataFrame with the original columns
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Importing libraries for clustering and visualization
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determining inertia for different values of k
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the elbow method to determine the optimal number of clusters
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choosing the number of clusters
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data_scaled)

# Assigning cluster labels to the data
data['Cluster'] = kmeans.labels_

# Displaying the first few rows of data with clusters
data.head()

# Visualizing the clusters
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title('Customer Clusters')
plt.show()

# Analyzing the clusters by calculating the mean of the features
cluster_analysis = data.groupby('Cluster').mean()
print(cluster_analysis)
