import pandas as pd
import numpy as np
from Model_1 import preprocess_features
from Model_1 import SVM_result
from Model_1 import random_forests_model
from Model_2 import NaiveBayes_result
from Model_2 import knnVoting
from Model_2 import NeuralNetwork
from Model_2 import LogisticRegressionModel


# path = './1_weather included.csv'
path = './Jan-data-updated.csv'
prediction = ['DEP_DELAY_NEW']
numeric = ['CRS_DEP_TIME','aver_temp','total_snow','aver_wind_speed','precipitation','aver_visibility','HOLIDAY']
bool = []
# string = ["FL_DATE","OP_UNIQUE_CARRIER","OP_CARRIER_FL_NUM","ORIGIN","DEST"]
string = ["FL_DATE","OP_UNIQUE_CARRIER","ORIGIN","DEST"]

train_x, train_y, df_train = preprocess_features(path, numeric, bool, string, prediction)
res_svm = SVM_result(train_x, train_y, df_train)
res_rf = random_forests_model(train_x, train_y, df_train)
res_nb = NaiveBayes_result(train_x, train_y, df_train)
res_knn = knnVoting(train_x, train_y, df_train)
res_nn = NeuralNetwork(train_x, train_y, df_train)
res_lr = LogisticRegressionModel(train_x, train_y, df_train)
# 'SVM','RandomForest','NaiveBayes','kNN','NeuralNetwork','LogisticRegression'
with open('./output/Metrics.txt', 'a') as f:
    f.write('\t'.join(['Numeric Features'] + numeric))
    f.write('\n')
    f.write('\t'.join(['Nominal Features'] + string))
    f.write('\n')
    f.write('\t'.join(['Sampling','Yes']))
    f.write('\n')
    f.write('\t'.join([' ','Accuracy','F1 score','auc']))
    f.write('\n')
    f.write('\t'.join(res_svm))
    f.write('\t'.join(res_rf))
    f.write('\t'.join(res_nb))
    f.write('\t'.join(res_knn))
    f.write('\t'.join(res_nn))
    f.write('\t'.join(res_lr))
    f.write('\n\n')
