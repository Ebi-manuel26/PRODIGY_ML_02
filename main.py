import data as data
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your customer purchase history data into a pandas DataFrame
# Replace 'customer_data.csv' with your actual data file
data = pd.read_csv('C:/Users/RABI/Desktop/custer/Mall_Customers.csv')


# Assuming you have columns like 'CustomerID', 'AmountSpent', 'Frequency', 'Recency', etc.
# Select the relevant features for clustering (e.g., AmountSpent, Frequency, Recency)
X = data[['CustomerID','Age','Annual Income','Scor']]

# Standardize the data to have mean = 0 and variance = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (K) - you can use techniques like Elbow Method to find the optimal K
k = 4  # Example, you can change this based on your analysis

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add the cluster labels back to the original DataFrame
data['Cluster'] = clusters

# View the resulting clusters
print(data[['CustomerID', 'Cluster']])
