# 1
# Getting Required Packages and tools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 2
# Uploading the data
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")

# PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
Seg = pd.DataFrame(data=principalComponents, columns=['dimension_1', 'dimension_2'])
print('PCA on Complete & Non-Normalize Data')
print(pca.explained_variance_ratio_)
# [0.76926041 0.10788044]

# Normalization
Scale = StandardScaler()
XTT = Scale.fit_transform(X)
XTT = pd.DataFrame.from_records(XTT)

# PCA
pca_0 = PCA(n_components=2)
principalComponents = pca_0.fit_transform(XTT)
Seg_0 = pd.DataFrame(data=principalComponents, columns=['dimension_1', 'dimension_2'])
print('PCA on Complete & Normalize Data')
print(pca_0.explained_variance_ratio_)
# [0.14207747 0.05266416]

# Scatter plot
Y = Y.replace((0, 1, 2), ("acquired", "closed", "operating"))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(' Dimension1', fontsize=15)
ax.set_ylabel(' Dimension 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = ['operating', 'closed', 'acquired']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = Y['Y'] == target
    ax.scatter(Seg.loc[indicesToKeep, 'dimension_1'], Seg.loc[indicesToKeep, 'dimension_2'], c=color, s=50)
ax.legend(targets)
ax.grid()

# Segmented PCA
pca_1 = PCA(n_components=2)
principalComponents = pca_1.fit_transform(XTT.iloc[:, 0:3])
Seg_1 = pd.DataFrame(data=principalComponents, columns=['1', '2'])
print('PCA on Selected Features')
print(pca_1.explained_variance_ratio_)

pca_2 = PCA(n_components=2)
principalComponents = pca_2.fit_transform(XTT.iloc[:, -22:-1])
Seg_2 = pd.DataFrame(data=principalComponents, columns=['3', '4'])
print('PCA on Selected Features')
print(pca_2.explained_variance_ratio_)
