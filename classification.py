# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:16:48 2023

@author: Sebastian
"""



DATA_DIRECTORY = "data/"
OUTPUT_DIRECTORY = "output/"
#%% Import libraries

import helpers as hlp

from sklearn import neighbors, svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

import glob

import matplotlib.pyplot as plt
import seaborn as sb

#%% import data

food_data = pd.read_csv(DATA_DIRECTORY + "food_data.csv")

# Picking these nutrients because afterwards there's a real jump in missing values
with(open(DATA_DIRECTORY + "nutrients.txt", "r")) as nutrient_file:
    nutrients = nutrient_file.read().splitlines()

#Choosing not to impute - 1197 rows is ~15%
# No I didn't use a pipeline
XY = food_data[nutrients + ["nutrition_grade_fr"]].dropna()

X = XY[nutrients]
Y = XY["nutrition_grade_fr"]

Y = Y.astype("category")    
Y = Y.cat.reorder_categories(["a", "b", "c", "d", "e"])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

#%% Define models

n_folds = 5

k_neighbours = neighbors.KNeighborsClassifier(weights = "distance")
random_forest = RandomForestClassifier()
decision_tree = DecisionTreeClassifier()
svm_linear = svm.SVC(kernel = "linear", C = 0.01)
svm_nonlinear = svm.SVC(kernel = "rbf", C = 0.01)
ridge = RidgeClassifier()

models = (
    ("K nearest neighbours", k_neighbours),
    ("Decision tree", decision_tree),
    ("Random forest", random_forest),
    ("Linear SVM", svm_linear),
    ("RBF SVM", svm_nonlinear),
    ("Ridge", ridge)
)

cf_dict = {}
cfn_dict = {}

for model_name, classifier in models:
    
    print(model_name)
    print("-- Cross-validating model")
    
    cv_results = cross_validate(classifier, X_train, Y_train, cv = n_folds, return_train_score=True)
    
    cv_df = pd.DataFrame({k: cv_results[k] for k in ("test_score", "train_score")})
    cv_df["fold"] = list(range(1, n_folds + 1))
    cv_df["model"] = model_name
    cv_df = cv_df.melt(id_vars=("fold", "model"), value_vars=("test_score", "train_score"), var_name="set", value_name="score")
    
    
    sb.barplot(cv_df, x="fold", y="score", hue="set").set(title=f"Barplot of {model_name} cross-validation with {n_folds} folds")
    plt.ylim(0,1)
    plt.savefig(DATA_DIRECTORY + "temp_class_" + model_name + ".png")
    plt.close()
    
    print("-- Fitting model")
    classifier.fit(X_train, Y_train)
    
    print("-- Predicting")
    Y_pred = classifier.predict(X_test)
    cf_dict[model_name] = confusion_matrix(Y_test, Y_pred)
    cfn_dict[model_name] = confusion_matrix(Y_test, Y_pred, normalize = 'true')
    

hlp.collateImages(glob.glob(DATA_DIRECTORY + "temp_class_*"), "classification_analysis.png.png")

for model,cm in cf_dict.items():
    disp = ConfusionMatrixDisplay(cm, display_labels=["a", "b", "c", "d", "e"])    
    disp.plot()
    plt.title(f"{model} confusion matrix")
    plt.savefig(f"{DATA_DIRECTORY}temp_validation_{model}.png")
    plt.close()
    
    disp = ConfusionMatrixDisplay(cfn_dict[model], display_labels=["a", "b", "c", "d", "e"])    
    disp.plot()
    plt.title(f"{model} normalised confusion matrix")
    plt.savefig(f"{DATA_DIRECTORY}temp_normalisedvalidation_{model}.png")
    plt.close()
    
    
    
hlp.collateImages(glob.glob(f"{DATA_DIRECTORY}/temp_validation_*.png"), "classification_validation.png")
hlp.collateImages(glob.glob(f"{DATA_DIRECTORY}/temp_normalisedvalidation_*.png"), "classification_normalisedvalidation.png")
