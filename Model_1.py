from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import OneHotEncoding
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import f1_score
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt


def preprocess_features(path, numericfeature, nominalfeature, stringfeature, prediction):
    df_train = pd.read_csv(path)
    df_train = df_train[:]
    df_train = df_train.sample(n=4000, random_state=1)
    df_train = df_train[['FL_DATE','OP_UNIQUE_CARRIER','OP_CARRIER_FL_NUM','ORIGIN'	,
                          'ORIGIN_STATE_ABR','DEST','DEST_STATE_ABR','CRS_DEP_TIME','DEP_TIME','HOLIDAY',
                         'DEP_DELAY_NEW','DISTANCE','aver_temp','total_snow','aver_wind_speed','precipitation','aver_visibility']]
    df_train.dropna(inplace=True)
    feature, label = OneHotEncoding.feature_processing(df_train,
                                                       numericfeature,
                                                       nominalfeature,stringfeature, prediction)
    print('Preprocessing done!')
    train_x = feature[:len(df_train), :]
    train_y = label[:len(df_train), :]
    df_train['label'] = train_y
    df_train.to_csv('./output/label.csv', index=False)
    return train_x, train_y, df_train


def SVM_result(train_x, train_y, df_train):
    # do CV GridSearch, test on test set
    # model = OneHotEncoding.svm_cross_validation(train_x,train_y.reshape(-1,))
    # print(np.average(cross_val_score(model, train_x, train_y.reshape(-1, ), cv=10)))

    # # Cross Validation on training dataset
    m2 = svm.SVC(C=200, gamma=0.001, kernel='poly')
    # print(np.average(cross_val_score(m2,train_x,train_y.reshape(-1,), cv=10)))
    predicted = cross_val_predict(m2, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print("SVM accuracy", metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    print(len(df_train.iloc[error_index]),'\n')
    error_df = df_train.iloc[error_index].sort_values(by=['FL_DATE'])
    error_df.to_csv('./output/Error_svm.csv', index=False)

    f1 = f1_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("f1 score: ", f1)

    auc = roc_auc_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("auc score: ", auc)
    return [str(k) for k in ['SVM',metrics.accuracy_score(train_y.reshape(-1, ), predicted),f1,auc]]


def random_forests_model(train_x, train_y, df_train):
    classifier = RandomForestClassifier(max_depth=10, random_state=0)
    predicted = cross_val_predict(classifier, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print("RF accuracy", metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    # print(len(df_train.iloc[error_index]))
    error_df = df_train.iloc[error_index].sort_values(by=['FL_DATE'])
    error_df.to_csv('./output/Error_svm.csv', index=False)

    f1 = f1_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("f1 score: ", f1)

    auc = roc_auc_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("auc scoreL ", auc)
    ############################################
    # model = classifier.fit(train_x, train_y.reshape(-1, ))
    # importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in model.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)[::-1]
    # # Print the feature ranking
    # print("Feature ranking:")
    # for f in range(train_x.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(train_x.shape[1]), importances[indices],
    #         color="r", yerr=std[indices], align="center")
    # plt.xticks(range(train_x.shape[1]), indices)
    # plt.xlim([-1, train_x.shape[1]])
    # plt.show()
    return [str(k) for k in ['RandomForest',metrics.accuracy_score(train_y.reshape(-1, ), predicted),f1,auc]]


if __name__ == '__main__':
    train_x, train_y, df_train = preprocess_features()
    SVM_result(train_x, train_y, df_train)
    random_forests_model(train_x, train_y, df_train)
