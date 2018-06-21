#!/usr/bin/python
#
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
import sys
#
X = list()
Y = list()
Z = list()
W = list()
#
tissues = sys.argv[1]
nr_features = int(sys.argv[2])
nr_final_features = int(sys.argv[3])
check_size = False
#
#
#
def TumorToLabel(tumor):
    if(tumor.find("SOB_B_F") != -1):
        return 0
    if(tumor.find("SOB_M_MC") != -1):
        return 4
    if(tumor.find("SOB_M_PC") != -1):
        return 5
    if(tumor.find("SOB_M_DC") != -1):
        return 6
    if(tumor.find("SOB_B_TA") != -1):
        return 1
    if(tumor.find("SOB_B_A") != -1):
        return 2
    if(tumor.find("SOB_M_LC") != -1):
        return 7
    if(tumor.find("SOB_B_PT") != -1):
        return 3
    print("Error tumor type: {}".format(tumor))
    exit(0)
    return -1
#
#
#
def load_crc(filename):
    X = list()
    Z = list()
    W = list()
    f = open(tissues, "r")
    for i in f:
        line = i[:-1].split(";")
        label = int(line[0][0:2])
        x = np.array(line[2:-1])
        # discard bad images (with less attributes due to image format)
        if(len(x) == nr_features):
            # selection of relevant and irrelevant
            X.append(x.astype(np.float))
            Z.append(line[1])
            W.append(label)
        else:
    	    print("{} {} {}".format(line[0], line[1], len(x)))
    f.close()
    return X, Z, W
#
#
#
def load_breakhis(filename):
    f = open(filename, "r")
    #
    X = list()
    Y = list()
    Z = list()
    W = list()
    for i in f:
        line = i[:-1].split(";")
        #
        x = np.array(line[2:])
        if(len(x) == nr_features):       
            X.append(x.astype(np.float))
            Z.append(line[1])
            W.append(line[0])
            Y.append(TumorToLabel(line[1]))
        else:
            print("Erro: {} {}".format(line[1], len(x)))
    #
    f.close()
    return X, Z, W
#
#
#
X, Z, W = load_breakhis(tissues)
#
#
#
#del Z, W
#
if(check_size == True):
    pca = PCA(n_components=nr_final_features)
    pca.fit(X)
    #
    perc = np.percentile(pca.explained_variance_,75)
    print(perc)
    total = np.sum(pca.explained_variance_)
    j=0
    var_acc = 0.0
    for i in pca.explained_variance_:
        var_acc += i
        if(i < perc):
            print("{} {:.4f} {:.4f}".format(j, i, var_acc/total))
            break
        j += 1
    #
    j=0
    var_acc = 0.0
    for i in pca.explained_variance_:
        var_acc += i
        if(var_acc/total > 0.95):
            print("{} {:.4f} {:.4f}".format(j, i, var_acc/total))
            exit(0) 
        j += 1
    exit(0)
else:
    pca = PCA(n_components=nr_final_features)
    pca.fit(X)

    x_tmp = list()
    for i in range(len(X)):
        x_tmp.append(np.squeeze(pca.transform([X[i]]))) #pca.transform([X[i]]) #np.squeeze(pca.transform([X[i]]))
    del X
    X, Z, W = load_breakhis(tissues)
    del X
    for i in range(len(x_tmp)):
        print("{};{}".format(W[i], Z[i]), end="")
        for j in x_tmp[i]:
            print(";{:.7f}".format(j), end="")
        print()
