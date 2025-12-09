#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
from sklearn.cluster import KMeans  
from sklearn.metrics import davies_bouldin_score  
from sklearn.decomposition import PCA  

RANDOM_STATE = 2025

print("# 1st exercise")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

print("Number of records:", X.shape[0])
print("Number of attributes:", X.shape[1])
print("Number of classes:", len(np.unique(y)))
print()

print("# 2nd exercise")
i_mean = list(feature_names).index('mean area')
i_se = list(feature_names).index('area error')

plt.scatter(X[:, i_mean], X[:, i_se], c=y, cmap='coolwarm', alpha=0.7)
plt.title("Scatterplot of area")
plt.xlabel('mean')
plt.ylabel('standart error')
plt.show()

print("# 3rd exercise")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
)

print("# 4th exercise")
models = []

for depth in [1, 2, 3, 4, 5]:
    dt = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    models.append((f"DT_depth_{depth}", dt, dt.score(X_test, y_test)))

lr = LogisticRegression(solver='liblinear', max_iter=1000)
lr.fit(X_train, y_train)
models.append(("LogReg", lr, lr.score(X_test, y_test)))

mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', max_iter=5000, random_state=RANDOM_STATE)
mlp.fit(X_train, y_train)
models.append(("MLP_5_logistic", mlp, mlp.score(X_test, y_test)))

best_name, best_model, best_score = max(models, key=lambda x: x[2])
print("Best model:", best_name)
print("Test score:", best_score)
print()

print("# 5th exercise")
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

print("# 6th exercise")
ConfusionMatrixDisplay(cm, display_labels=cancer.target_names).plot(cmap='Blues')
plt.title(best_name)
plt.show()

print("# 7th exercise")
ks = range(2, 31)
db_scores = [davies_bouldin_score(X, KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)) for k in ks]

best_k = ks[int(np.argmin(db_scores))]
print("Optimal K:", best_k)
print()

print("# 8th exercise")
labels4 = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
X_pca = PCA(n_components=2).fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels4, cmap='tab10', alpha=0.7)
plt.title("K-means (k=4) with PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
