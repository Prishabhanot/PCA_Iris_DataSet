from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Iris.csv")
print(df.head())

labels = df['Species']
X = df.drop(["Id", "Species"], axis=1)

# Standardize the features
X_std = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_transform = pca.fit_transform(X_std)

print(pca.explained_variance_ratio_)
print(X_transform)

# Unpack the transformed features
pca1, pca2 = zip(*X_transform)

# Define the color dictionary
color_dict = {
    "Iris-setosa": "green",
    "Iris-versicolor": "red",
    "Iris-virginica": "blue"
}

# Plot the first two principal components
for i, label in enumerate(labels):
    plt.scatter(pca1[i], pca2[i], color=color_dict[label])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()
