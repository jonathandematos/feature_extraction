#!/usr/bin/python
#
from __future__ import print_function
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
#
if(len(sys.argv) != 6):
    print("classify.py [nr_features] [zoom] [features_file] [fold_file] [inductor]")
    exit(0)
#
nr_features = 162
zoom = 40
features = "data/breakhis_pftas/40.txt"
fold_file = "folds/dsfold1.txt"
inductor = "svm"
#
nr_features = int(sys.argv[1])
zoom = int(sys.argv[2])
features = sys.argv[3]
fold_file = sys.argv[4]
inductor = sys.argv[5]
#
#
#
def TumorToLabel(tumor):
    if(tumor.find("SOB_B_F") != -1):
        return 1
    if(tumor.find("SOB_M_MC") != -1):
        return 0
    if(tumor.find("SOB_M_PC") != -1):
        return 0
    if(tumor.find("SOB_M_DC") != -1):
        return 0
    if(tumor.find("SOB_B_TA") != -1):
        return 1
    if(tumor.find("SOB_B_A") != -1):
        return 1
    if(tumor.find("SOB_M_LC") != -1):
        return 0
    if(tumor.find("SOB_B_PT") != -1):
        return 1
    print("Error tumor type: {}".format(tumor))
    exit(0)
    return -1
#
#
#
def load_dataset(filename):
    f = open(filename, "r")
    #
    X = list()
    Y = list()
    Z = list()
    for i in f:
        line = i[:-1].split(";")
        #
        x = np.array(line[2:]).astype("float32")
        if(len(x) == nr_features):       
            X.append(x)
            Z.append(line[1])
            Y.append(TumorToLabel(line[1]))
        else:
            print("Erro: {} {}".format(line[1], len(x)))
    #
    f.close()
    return X, Y, Z
#
#
#
def generate_fold(X, Y, Z, fold_file, zoom):
    imgs_train = list()
    imgs_test = list()
    f = open(fold_file, "r")
    for i in f:
        linha = i[:-1].split("|")
        if(int(linha[1]) == zoom):
            img = linha[0].split(".")[0]
            if(linha[3] == "train"):
                imgs_train.append(img)
            if(linha[3] == "test"):
                imgs_test.append(img)
    f.close()
    X_train = list()
    Y_train = list()
    Z_train = list()
    X_test = list()
    Y_test = list()
    Z_test = list()
    print(len(imgs_train), len(imgs_test), len(X))
    for i in range(len(X)):
        tmp_img = Z[i].split("-")
        main_img = tmp_img[0]+"-"+tmp_img[1]+"-"+tmp_img[2]+"-"+tmp_img[3]+"-"+tmp_img[4].split("_")[0]
        #print(imgs_train[0])
        #exit(0)
        if(main_img in imgs_train):
            X_train.append(X[i])
            Y_train.append(Y[i])
            Z_train.append(Z[i])
        if(main_img in imgs_test):
            X_test.append(X[i])
            Y_test.append(Y[i])
            Z_test.append(Z[i])
    return X_train, Y_train, Z_train, X_test, Y_test, Z_test
#
#
#
def grid_report(clf, X_test, Y_test):
    print("Melhores parametros:")
    print(clf.best_params_)
    #
    print("\nScores:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    #
    print("Relatorio do teste:")
    Y_pred = clf.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
#
#
#
X, Y, Z = load_dataset(features)
scaling = False
if(scaling == True):
    X_train_u, Y_train, Z_train, X_test_u, Y_test, Z_test = generate_fold(X, Y, Z, fold_file, zoom)
    #
    print(len(X_train_u), len(X_test_u)) 
    scaler = preprocessing.StandardScaler().fit(X_train_u)
    X_train = scaler.transform(X_train_u)
    X_test = scaler.transform(X_test_u)
else:
    X_train, Y_train, Z_train, X_test, Y_test, Z_test = generate_fold(X, Y, Z, fold_file, zoom)
    print(len(X_train), len(X_test)) 
#
#
#param_dist = {"n_estimators": [200,400,600,800]}
#              "max_depth": [3, None],
#              "max_features": sp_randint(1, 11),
#              "max_features": [1, 11],
#              "min_samples_split": sp_randint(1, 11),
#              "min_samples_split": [2, 11],
#              "min_samples_leaf": sp_randint(1, 11),
#              "min_samples_leaf": [1, 11],
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"]}
#
#scores = ['precision_macro', 'recall_macro', 'accuracy']
scores = ['accuracy']
#
for i in scores:
    #
    if(inductor == "svm"):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-5, 1e-2, 1e-1, 1e-3, 1],
                     'C': [1, 10, 50]}]
                    #{'kernel': ['linear'], 'C': [1, 10, 50, 100]}]
        clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring=i, n_jobs=4, verbose=10)
    #
    if(inductor == "rf"):
        param_dist = {"n_estimators": [50,100,200,400]}
        clf = GridSearchCV(RandomForestClassifier(), param_dist, cv=5, scoring=i, n_jobs=8)
    #
    clf.fit(X_train, Y_train)
    grid_report(clf, X_test, Y_test)
#
for i in range(len(X_test)):
    pred = clf.predict_proba([X_test[i]])
    pred_np = np.squeeze(pred)
    print("{};{};".format(Z_test[i], Y_test[i]), end="")
    for j in pred_np:
        print("{:.6f};".format(j), end="")
    print()
#
