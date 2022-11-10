import pandas as pd
import numpy as np
from sklearn import tree, ensemble, svm, neighbors, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

respondents = pd.read_csv("drug_consumption.csv")

feature_list = [
"ID",
"Age",
"Gender",
"Education",
"Country",
"Ethnicity",
"Nscore",
"Escore",
"Oscore",
"Ascore",
"Cscore",
"Impulsive",
"SS",
"Alcohol",
"Amphet",
"Amyl",
"Benzos",
"Caff",
"Cannabis",
"Choc",
"Coke",
"Crack",
"Ecstasy",
"Heroin",
"Ketamine",
"Legalh",
"LSD",
"Meth",
"Mushrooms",
"Nicotine",
"Semer",
"VSA"
]
drugs_list = ["Alcohol",
"Amphet",
"Amyl",
"Benzos",
"Caff",
"Cannabis",
"Choc",
"Coke",
"Crack",
"Ecstasy",
"Heroin",
"Ketamine",
"Legalh",
"LSD",
"Meth",
"Mushrooms",
"Nicotine",
"Semer",
"VSA"]

# create features
respondents.columns = feature_list

# create labels
drugs = respondents[drugs_list].copy()
respondents = respondents.drop(drugs_list, axis=1)

# create label values function 
def BinaryLabel(df, column):
	for i in df[column].index:
		if(df[column][i]=="CL0" or df[column][i]=="CL1"):
			df[column][i] = 0
		else:
			df[column][i] = 1


# create and plot confusion matrix
def createConfusionMatrix(test_col, pred_col, drug_name, col_num, axes):
	cm = metrics.confusion_matrix(test_col.values.tolist(), pred_col)
	disp = metrics.ConfusionMatrixDisplay(cm)
	disp.plot(ax=axes[col_num], xticks_rotation=45)
	disp.ax_.set_title(drug_name)
	disp.im_.colorbar.remove()
	disp.ax_.set_xlabel('predicted labels')


# label data with value 1 for user and value 0 for non-user
for drug in drugs:
	BinaryLabel(drugs,drug)


# feature selection
feature_selector = VarianceThreshold()
respondents = feature_selector.fit_transform(respondents)




drugs_subset = ["Cannabis"]

