import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def NaiveBayes_result(train_x, train_y, df_train):
    m = GaussianNB()
    # print(np.average(cross_val_score(m2,train_x,train_y.reshape(-1,), cv=10)))
    predicted = cross_val_predict(m, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print("NB accuracy", metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    # print(len(df_train.iloc[error_index]),'\n')
    # error_df = df_train.iloc[error_index].sort_values(by=['FL_DATE'])
    # error_df.to_csv('./output/Error_nb.csv', index=False)

    f1 = f1_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("f1 score: ", f1)

    auc = roc_auc_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("auc score: ", auc)
    return [str(k) for k in ['Naive Bayes', metrics.accuracy_score(train_y.reshape(-1, ), predicted),f1,auc]]

def knnVoting(train_x, train_y, df_train):
    m = KNeighborsClassifier(5)
    predicted = cross_val_predict(m, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print("KNN accuracy", metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    # print(len(df_train.iloc[error_index]),'\n')
    # error_df = df_train.iloc[error_index].sort_values(by=['FL_DATE'])
    # error_df.to_csv('./output/Error_knn.csv', index=False)

    f1 = f1_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("f1 score: ", f1)

    auc = roc_auc_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("auc score: ", auc)
    return [str(k) for k in ['kNN', metrics.accuracy_score(train_y.reshape(-1, ), predicted), f1, auc]]


def NeuralNetwork(train_x, train_y, df_train):
    m = MLPClassifier(alpha=1)
    predicted = cross_val_predict(m, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print("Neural Network accuracy", metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    # print(len(df_train.iloc[error_index]),'\n')
    # error_df = df_train.iloc[error_index].sort_values(by=['FL_DATE'])
    # error_df.to_csv('./output/Error_nn.csv', index=False)

    f1 = f1_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("f1 score: ", f1)

    auc = roc_auc_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("auc score: ", auc)
    return [str(k) for k in ['Neural Network', metrics.accuracy_score(train_y.reshape(-1, ), predicted), f1, auc]]


def LogisticRegressionModel(train_x, train_y, df_train):
    m = LogisticRegression()
    predicted = cross_val_predict(m, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print("LR accuracy", metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    # print(len(df_train.iloc[error_index]),'\n')
    # error_df = df_train.iloc[error_index].sort_values(by=['FL_DATE'])
    # error_df.to_csv('./output/Error_lr.csv', index=False)

    f1 = f1_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("f1 score: ", f1)

    auc = roc_auc_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("auc score: ", auc)
    return [str(k) for k in ['Logistic Regression', metrics.accuracy_score(train_y.reshape(-1, ), predicted),f1,auc]]