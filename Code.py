from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

df=pd.read_csv("Iris.csv")
print (df.head())

labels = df['Species']

X  = df.drop(["Id", "Species"], axis=1)

#Center all xi around orgin 
X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)

X_transform = pca.fit_transform(X_std)

print (pca.explained_variance_ratio_)
print(X_transform)

pca1 = zip(*X_transform)[0]
pca2 = zip(*X_transform)[1]

color_dict={}
color_dict["Iris-sentosa"]="green"
color_dict["Iris-versicolor"]="red"
color_dict["Iris-verginica"]="blue"

for label in labels:
    plt.scatter(pca1[i],pca2[i],color=color_dict)

plt.show()

