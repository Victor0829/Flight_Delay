import pandas as pd
from sklearn import preprocessing
import numpy as np
import sys


def feature_processing(df, numericfeature, boolfeature, categoricalfeature, prediction):
    if numericfeature:
        numericfeature = scale_numeric_features(df.loc[:, numericfeature].values.reshape(len(df),-1))
        # print(numericfeature.shape)
        # numericfeature = df.loc[:, numericfeature].values.reshape(len(df), -1)
    else:
        numericfeature = np.zeros((len(df),1))
    print('Numeric Feature done!')

    if boolfeature:
        boolfeature = process_bool_features(df.loc[:, boolfeature].values.reshape(len(df),-1))
    else:
        boolfeature = np.zeros((len(df),1))
    print('Bool Feature done!')

    if categoricalfeature:
        categoricalfeature = process_categorical_features(df.loc[:, categoricalfeature].values.reshape(len(df),-1))
    else:
        categoricalfeature = np.zeros((len(df),1))
    print('Categorical Feature done!')

    # print(numericfeature.shape, boolfeature.shape, categoricalfeature.shape)
    # print(sys.getsizeof(categoricalfeature))
    feature = np.hstack((numericfeature,boolfeature,categoricalfeature))
    # print(feature.shape)

    classlabel = df.loc[:, prediction].values
    classlabel[classlabel < 15] = 0
    classlabel[classlabel >= 15] = 1
    classlabel = classlabel.reshape(len(df), -1)
    label = process_bool_features(classlabel)
    # print(classlabel)

    return feature, label

def write_ndarray(arr, name):
    with open('./output/'+ name + '.txt', 'w') as f:
        f.write('\n'.join(str(i) for i in arr))


def process_numeric_features(numericFeature):
    enc = preprocessing.OneHotEncoder()
    f = enc.fit_transform(numericFeature).toarray()
    return f


def scale_numeric_features(numericFeature):
    scaler = preprocessing.StandardScaler()
    f = scaler.fit_transform(numericFeature)
    return f


def process_bool_features(boolFeature):
    nominalf = np.zeros(boolFeature.shape)
    for i in range(boolFeature.shape[1]):
        # print('Processing Nonimal Feature',i)
        le = preprocessing.LabelEncoder()
        nominalf[:, i] = le.fit_transform(boolFeature[:,i])
    return nominalf


def process_categorical_features(categoricalFeature):
    f = process_bool_features(categoricalFeature)
    enc = preprocessing.OneHotEncoder()
    return enc.fit_transform(f).toarray()


def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1, cv=10)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    print(grid_search.best_score_)
    return model


if __name__ == '__main__':
    path = './output/eyeFeatures.csv'
    df = pd.read_csv(path, index_col=False)  # input
    print(df)
