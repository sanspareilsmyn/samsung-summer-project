# -*- coding: utf-8 -*-
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
from config import *
import numpy as np

def train_svm():
    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the Class-1 features
    for feat_path in glob.glob(os.path.join(c1_feat_ph,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the Class-2 features
    for feat_path in glob.glob(os.path.join(c2_feat_ph,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print (np.array(fds).shape,len(labels))
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print ("Training a Linear SVM Classifier")
        clf.fit(fds, labels)
        joblib.dump(clf, model_path+'hogsvm.model')              
        print ("Classifier saved to {}".format(model_path))
        
train_svm()
