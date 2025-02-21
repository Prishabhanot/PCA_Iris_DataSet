from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

df=pd.read_csv("Iris.csv")
print (df.head())

labels = df['Species']

x = df.drop(["Id", "Species"], axis=1)

#Center all xi around orgin 