#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:07:23 2022

@author: gabriel
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

data = pd.read_csv('/users/gabriel/desktop/health_first/covid_hospitalization_data.csv')

# Encode categorical variables
data['Case_Severity_Enc'] = data['Case_Severity']
data['Case_Severity_Enc'] = data['Case_Severity_Enc'].replace(['Mild', 'Moderate', 'Severe'] , [0, 1, 2])

# Features & label
all_features = data[['Available_Rooms', 'Bed_Grade', 'Registered_Visitors', 'Patient_Age', 'Case_Severity_Enc', 'Initial_Amount_Billed']]
label = data['Stay_Over_30_Days']

# Normalize features
norm_features = pd.DataFrame(preprocessing.normalize(all_features, axis = 0))
norm_features.columns = all_features.columns

# Set random seed
seed = 9

# Model with all features
x_train, x_test, y_train, y_test = train_test_split(norm_features, label, test_size = 0.3, random_state = seed)
knn_model = KNeighborsClassifier(n_neighbors = 7)
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)
starting_acc = metrics.accuracy_score(y_test, y_pred)
print('All features        |  Acc.  | F', '\n------------------------------------')
print(''.ljust(22), round(metrics.accuracy_score(y_test, y_pred), 3), ' ', round(metrics.f1_score(y_test, y_pred), 3), '\n')

# Automate feature selection by backwards elimination using accuracy
print('Feature dropped     |  Acc.  | F', '\n------------------------------------')
feature_count = len(all_features.columns)

for j in range(feature_count):
    acc_lst = []
    for i in range(feature_count):
        column_to_drop = all_features.columns[i]
        feature_subset = all_features.drop(columns = column_to_drop)
        x_train, x_test, y_train, y_test = train_test_split(feature_subset, label, test_size = 0.3, random_state = seed)
        knn_model = KNeighborsClassifier(n_neighbors = 7)
        knn_model.fit(x_train, y_train)
        y_pred = knn_model.predict(x_test)
        column_to_drop_spaced = column_to_drop.ljust(22)
        acc_lst.append(metrics.accuracy_score(y_test, y_pred))
        max_acc = max(acc_lst)
        print(column_to_drop_spaced, round(metrics.accuracy_score(y_test, y_pred), 3), ' ', round(metrics.f1_score(y_test, y_pred), 3))
    print('\n')
    max_acc_index = acc_lst.index(max_acc)
    if max_acc > starting_acc:
        acc_improvement = round(max_acc - starting_acc, 4)
        column_to_drop = all_features.columns[max_acc_index]
        print('Round ', j + 1, ': ', 'Drop', column_to_drop, 'to improve acc. by', acc_improvement, '\n')
        all_features = all_features.drop(columns = column_to_drop)
        starting_acc = max_acc
        feature_count -= 1
    else:
        selected_features = all_features.columns
        print('Selected features: ', selected_features)
        break

x_train, x_test, y_train, y_test = train_test_split(all_features[selected_features], label, test_size = 0.3, random_state = seed)

# Hyperparameter optimization
ks = []
accs = []
f1s = []
for i in range(1, 40):
    ks.append(i)
    knn_model = KNeighborsClassifier(n_neighbors = i)
    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_test)
    accs.append(metrics.accuracy_score(y_test, y_pred))
    f1s.append(metrics.f1_score(y_test, y_pred))
max_acc = max(accs)
max_acc_ind = accs.index(max_acc)
optimal_k = max_acc_ind + 1
print('\nOptimal K:', optimal_k)
plt.scatter(ks, accs, color = 'blue', label = 'Accuracy')
plt.scatter(ks, f1s, color = 'red', label = 'f1')
plt.xlabel('# of Neighbors (K)')
plt.ylabel('Performance')
plt.legend(loc = "lower right")

# Evaluate final model using confusion matrix
knn_model = KNeighborsClassifier(n_neighbors = optimal_k)
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)
matrix = plot_confusion_matrix(knn_model, x_test, y_test, cmap = plt.cm.Blues, normalize='true')
plt.title('KNN Confusion Matrix')
plt.show(matrix)
plt.show()
print('\nFinal model')
print('Accuracy:'.ljust(20), round(metrics.accuracy_score(y_test, y_pred), 3))
print('Precision:'.ljust(20), round(metrics.precision_score(y_test, y_pred), 3))
print('Recall:'.ljust(20), round(metrics.recall_score(y_test, y_pred), 3))
print('F:'.ljust(20), round(metrics.f1_score(y_test, y_pred), 3))




