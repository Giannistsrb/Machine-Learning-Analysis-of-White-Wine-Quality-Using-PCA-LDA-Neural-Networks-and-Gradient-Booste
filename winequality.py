# Load libraries
import math
import matplotlib
from matplotlib import pyplot, colors
import numpy
# Load libraries
import math
import matplotlib
from matplotlib import pyplot, colors
import numpy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc

#-------------------------------------------------------
#EXERCISE 1
dataset = pandas.read_excel('winequality-white_without9and3.xlsx')
dataset=dataset.dropna()
names = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"]

vars = names[:-1]
    
target = ["quality_class"]

# See the number of rows and columns
print("Rows, columns: " + str(dataset.shape))

# See the first five rows of the dataset
dataset.head()
#Making the histogram of quality
print(dataset["quality"].value_counts(normalize=True))
classify_quality = sns.countplot(x="quality", data=dataset)

plt.show()

# Create Classification version of target variable
dataset["quality_class"]=[1 if x>=6 else 0 for x in dataset["quality"]]
dataset = dataset.drop(columns="quality")

datasets = dataset.groupby('quality_class')
data_good = datasets.get_group(1)
data_bad = datasets.get_group(0)
print(data_good)
print(data_bad)



#See proportion of good vs bad wines:
Proportion_of_Good_vs_Bad_Wines = dataset['quality_class'].value_counts()
print('Proportion of Good vs Bad Wines:') 
print(Proportion_of_Good_vs_Bad_Wines)

#------------------------------------------
#EXERCISE 2

# since the good and bad events are not equal, we will use the same number of events from each class
eventNo = min(len(data_good.index),len(data_bad.index))

# split into train (75%) and test (25%) datasets
#set seed
train_good, test_good = train_test_split(data_good, test_size=int(0.25*eventNo), train_size=int(0.75*eventNo))
train_bad, test_bad = train_test_split(data_bad, test_size=int(0.25*eventNo), train_size=int(0.75*eventNo))

#Data shapes
print('Traning bad data shape:', train_bad.shape)
print('Traning good data shape:', train_good.shape)

# specify the source and the target variables
train_good_sourcevars = train_good[vars]
train_good_targetvar = train_good[target]
test_good_sourcevars = test_good[vars]
test_good_targetvar = test_good[target]
train_bad_sourcevars = train_bad[vars]
train_bad_targetvar = train_bad[target]
test_bad_sourcevars = test_bad[vars]
test_bad_targetvar = test_bad[target]

# define the train and test datasets by merging the good and bad events
data_train_sourcevars = train_good_sourcevars.append(train_good_sourcevars)
data_train_targetvar = train_good_targetvar.append(train_good_targetvar)
data_test_sourcevars = test_good_sourcevars.append(test_good_sourcevars)
data_test_targetvar = test_good_targetvar.append(test_good_targetvar)

#-------------------------------------------------------------------------
#EXERCISE 3
# histograms

for k in vars:
    df = pandas.DataFrame(
    {
        "good_wine": train_good_sourcevars[k],
        "bad_wine": train_bad_sourcevars[k],
    },
    columns=["good_wine", "bad_wine"],
    )
    df.plot.hist(alpha=0.5,bins=20)
    pyplot.xlabel(k)

pyplot.show()

#------------------------------------------------------------------------
#EXERCISE 4
#Correlation matrix for good class
corr_good = data_good[vars].corr()
matplotlib.pyplot.subplots(figsize=(15,10))
sns.heatmap(corr_good, xticklabels=corr_good.columns, yticklabels=corr_good.columns, annot=True, 
            cmap=sns.diverging_palette(220, 20, as_cmap=True))

#Correlation matrix for bad class
corr_bad = data_bad[vars].corr()
matplotlib.pyplot.subplots(figsize=(15,10))
sns.heatmap(corr_bad, xticklabels=corr_bad.columns,yticklabels=corr_good.columns, annot=True,
            cmap=sns.diverging_palette(220, 20, as_cmap=True))



#-------------------------------------------------------------------------
#EXERCISE 5

#Standardization of variables
data_train_sourcevars_scaled = preprocessing.StandardScaler().fit_transform(data_train_sourcevars)
df = pandas.DataFrame(data_train_sourcevars_scaled)
df1=pandas.DataFrame(preprocessing.StandardScaler().fit_transform(dataset.drop(columns="quality_class")))
#PCA Analysis

wineVars=dataset.drop(columns="quality_class")
pca_wine = PCA(n_components = 7)
principalComponents_wine = pca_wine.fit_transform(df)

pca_transformed = pandas.DataFrame(pca_wine.transform(df1))
pca_inverse = pandas.DataFrame(pca_wine.inverse_transform(pca_transformed), columns=wineVars.columns)


#Variance explained in PCA Analysis with plot (using numpy)
covar_matrix = PCA(n_components = 7)
covar_matrix.fit(df)
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var=numpy.cumsum(numpy.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features

figure=plt.figure(figsize=(10,6))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')

plt.plot(var)

#Ratio of the components
print('Explained variation per principal component: {}'.format(pca_wine.explained_variance_ratio_))

#Eigenvalues
cov_mat = numpy.cov(principalComponents_wine.T)
eigen_vals, eigen_vecs = numpy.linalg.eig(cov_mat)
eigen_vals.sort()

#Scree plot of eigenvalues
figure=plt.figure(figsize=(10,6))
sing_vals=numpy.arange(len(eigen_vals)) + 1
plt.plot(sing_vals,eigen_vals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue') 
plt.show() 


#----------------------------------------------------------------------------
#EXERCISE 6

from sklearn.preprocessing import StandardScaler
features = dataset.drop(['quality_class'], axis=1)
labels = dataset["quality_class"]
X_train, X_test, y_train, y_test = train_test_split(pca_inverse, labels,  test_size=int(0.25*eventNo), train_size=int(0.75*eventNo), random_state=0)
stand_scal = StandardScaler()
X_train = stand_scal.fit_transform(X_train)
X_test = stand_scal.transform (X_test)

#RandomForestClassification
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print('Random Forest Classifier:', accuracy_score(y_test, y_pred))

#LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_model = LDA(n_components = 1)
X_train_lda = lda_model.fit_transform(X_train, y_train)
X_test_lda = lda_model.transform(X_test)

print(X_train_lda.shape)
print(X_test_lda.shape)

model = LDA(solver='lsqr', shrinkage='auto')
model.fit(X_train_lda,
          y_train)
y_pred = model.predict(X_test_lda)

print('LDA Model:', accuracy_score(y_test, y_pred))

# # probability to get the prediction bad,good for the sample: sample_LDA
#sample_LDA = data_good[vars].append(data_bad[vars]).values.tolist()
P_bad = model.predict_proba(X_train_lda)[:,0]
P_good = model.predict_proba(X_train_lda)[:,1]

# #οριζουμε τα βαρη των μεταβλητων
W_LDA = model.coef_[0]
W0_LDA = model.intercept_

print("===== LDA WEIGHTS =======")
print("W = ",W_LDA)
print("W0 = ",W0_LDA)
print("Normalized W = ",W_LDA/W_LDA[0])
print("Normalized W0 = ",W0_LDA/W_LDA[0])

# plot the probability distribution for the train sample: sample_LDA
plot2 = pyplot.figure(2)
bins = numpy.linspace(-0.1, 1.1, 48) # fixed bin size
pyplot.hist(P_bad, bins=bins, alpha=0.2,label='bad')
pyplot.hist(P_good, bins=bins, alpha=0.2,label='good')
pyplot.title('Probability distributions')
pyplot.xlabel('probability')
pyplot.ylabel('count')
pyplot.yscale('log')
pyplot.legend(loc='upper center')
pyplot.show()
# #ΣΧΟΛΙΟ: απο το ιστογραμμα συμπεραινουμε πως τα περισσοτερα κρασια, σε αντιθεση με το ιστογραμμα του exercise 2II
# # που εχουμε πολλα γεγονοτα στα ακρα (0 και 100%), πεφτουν κοντα στη μεση (μετρια κρασια 4-6 βαθμολογια) ενω στα ακρα
# # (πολυ καλα ή πολυ ασχημα) έχουμε λιγα γεγονοτα (counts). Δεν χρησιμοποιησαμε τα τεστ και train που εχουμε φτιαξει γιατι 
# # ο linear δεν μπορει να παθει overtrain.

# #--------------------------------------------------
#EXERCISE 7


clf = MLPClassifier(
    hidden_layer_sizes=(7,7),
     activation="relu",
     max_iter=400

 )



wineQuality = dataset.loc[:, "quality_class"]
X_trainNew, X_testNew = train_test_split(pca_inverse, test_size=int(0.25*eventNo), train_size=int(0.75*eventNo), random_state=0)
y_trainNew, y_testNew = train_test_split(wineQuality, test_size=int(0.25*eventNo), train_size=int(0.75*eventNo), random_state=0)

x = []
y = []

pyplot.figure()
count = 0
color_list = ["red","green","blue","orange","cyan","magenta","darkblue","darkorange","teal","olivedrab","crimson"]

for k in vars:
      vars1 = vars.copy()
      vars1.remove(k)
      print(vars1)
    
      clf.fit(X_trainNew[vars1].values, 
              y_trainNew.values.ravel())
    
      fpr, tpr, threshold = roc_curve(y_trainNew, 
                                      clf.predict_proba(X_trainNew[vars1])[:,1])
      roc_auc = auc(fpr, tpr)
      x.append(k)
      y.append(roc_auc)
    
      pyplot.plot(
      fpr,
      tpr,
    color=color_list[count],
    lw=1,
    label=k
    )
      count+=1
    
pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.title("Receiver operating characteristic example")
pyplot.legend(loc="lower right")
pyplot.show()

plot1 = pyplot.figure(1)
pyplot.ylabel('auc')
pyplot.xlabel('removed variable')
pyplot.plot(x, y)
pyplot.show()

#Συγκριση μεταξυ νευρωνικων με διαφορετικες συναρτησεις ενεργοποιησης
clf1 = MLPClassifier(
    hidden_layer_sizes=(7,7),
    activation="relu",
    verbose=True, #an thelo na vlepo tin poreia toy kathos trexei
    max_iter=300
)
clf2 = MLPClassifier(
    hidden_layer_sizes=(7,7),
    activation="logistic",
    verbose=False,
    max_iter=300
)
clf3 = MLPClassifier(
    hidden_layer_sizes=(7,7),
    activation="tanh",
    verbose=False,
    max_iter=300
)

clf1.fit(X_trainNew[vars1].values, y_trainNew.values.ravel() )
clf2.fit(X_trainNew[vars1].values, y_trainNew.values.ravel() )
clf3.fit(X_trainNew[vars1].values, y_trainNew.values.ravel() )

fpr1, tpr1, threshold1 = roc_curve(y_trainNew, clf1.predict_proba(X_trainNew[vars1])[:,1])
fpr2, tpr2, threshold2 = roc_curve(y_trainNew, clf2.predict_proba(X_trainNew[vars1])[:,1])
fpr3, tpr3, threshold3 = roc_curve(y_trainNew, clf3.predict_proba(X_trainNew[vars1])[:,1])
fpr4, tpr4, threshold4 = roc_curve(y_train, model.predict_proba(X_train_lda)[:,1])

print(clf1.score(X_trainNew[vars1].values, y_trainNew.values.ravel()))
print(clf1.score(X_testNew[vars1].values, y_testNew.values.ravel()))
print(clf2.score(X_trainNew[vars1].values, y_trainNew.values.ravel()))
print(clf2.score(X_testNew[vars1].values, y_testNew.values.ravel()))
print(clf3.score(X_trainNew[vars1].values, y_trainNew.values.ravel()))
print(clf3.score(X_testNew[vars1].values, y_testNew.values.ravel()))
print(model.score(X_train_lda, y_train))
print(model.score(X_train_lda, y_train))

roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)

pyplot.plot(clf1.loss_curve_,color="darkorange",label="MLP1")
pyplot.plot(clf2.loss_curve_,color="red",label="MLP2")
pyplot.plot(clf3.loss_curve_,color="green",label="MLP3")
pyplot.legend(loc="upper right")
pyplot.show()

pyplot.figure()
lw = 2
pyplot.plot(
    fpr1,
    tpr1,
    color="darkorange",
    lw=lw,
    label="relu (area = %0.2f)" % roc_auc1,
)
pyplot.plot(
    fpr2,
    tpr2,
    color="red",
    lw=lw,
    label="logistic (area = %0.2f)" % roc_auc2,
)
pyplot.plot(
    fpr3,
    tpr3,
    color="green",
    lw=lw,
    label="tanh (area = %0.2f)" % roc_auc3,
)


pyplot.plot(
    fpr4,
    tpr4,
    color="black",
    lw=lw,
    label="LDA (area = %0.2f)" % roc_auc4,
)
pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.title("Receiver operating characteristic example")
pyplot.legend(loc="lower right")
pyplot.show()
  
#--------------------------------------------------------------------------
# #EXERCISE 8

#GradientBoostingClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree

GBC = GradientBoostingClassifier(max_depth=3, 
                                      n_estimators=32, 
                                      random_state=123).fit(X_trainNew, y_trainNew)
predictions = GBC.predict(X_trainNew)

print("Algorithm: Gradient Boosted Decision Trees:")
print("0-1 Loss = " + str(numpy.round(zero_one_loss(predictions, 
                                                    y_trainNew),2)))

#---------------------------------------------------------------------------
#EXERCISE 9 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#LDA report
model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
model.fit(X_train_lda, y_train)
predictedLDA = model.predict(X_test_lda)
matrix = confusion_matrix(y_test, predictedLDA)
print(matrix)
report = classification_report(y_test, predictedLDA)
print(report)

#Neural Network report

for k in vars:
      vars1 = vars.copy()
      vars1.remove(k)
      print(vars1)
      
      predictNEURAL = clf.predict(X_trainNew[vars1])
     

      print("Confusion Matrix:")
      print(confusion_matrix(y_trainNew, predictNEURAL))

      print("Classification Report")
      print(classification_report(y_trainNew, predictNEURAL))

#Boosted Tree report

predictGBC = GBC.predict(X_trainNew)

print("Confusion Matrix:")
print(confusion_matrix(y_trainNew, predictGBC))

print("Classification Report")
print(classification_report(y_trainNew, predictGBC))


#----------------------------------------------------------------------------
#EXERCISE 10

#Timers:
import time
# (a) Linear Discriminant Analysis

start = time.time()
lda_model.fit(X_train_lda,
          y_train)
stop = time.time()
print(f"Training time LDA: {stop - start}s")

# (b) Neural Network 

start = time.time()
clf.fit(X_trainNew,
          y_trainNew)
stop = time.time()
print(f"Training time Neural: {stop - start}s")

# (c) Boosted Tree 

start = time.time()
GBC.fit(X_trainNew,
          y_trainNew)
stop = time.time()
print(f"Training time Boost: {stop - start}s")

#____________________________________________________________________________
#____________________________________________________________________________

#Roc Curve Total (LDA+BOOST DECISION TREE+CLF)

x = []
y = []

pyplot.figure()
count = 0
color_list = ["red","green","blue","orange","cyan","magenta","darkblue","darkorange","teal","olivedrab","crimson"]

for k in vars:
      vars1 = vars.copy()
      vars1.remove(k)
      print(vars1)
    
      clf.fit(X_trainNew[vars1].values, 
              y_trainNew.values.ravel())
    
      fpr, tpr, threshold = roc_curve(y_trainNew, 
                                      clf.predict_proba(X_trainNew[vars1])[:,1])
      roc_auc = auc(fpr, tpr)
      x.append(k)
      y.append(roc_auc)
    
      pyplot.plot(
      fpr,
      tpr,
    color=color_list[count],
    lw=1,
    label=k
    )
      count+=1
    
pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.title("Receiver operating characteristic example")
pyplot.legend(loc="lower right")
pyplot.show()

plot1 = pyplot.figure(1)
pyplot.ylabel('auc')
pyplot.xlabel('removed variable')
pyplot.plot(x, y)
pyplot.show()

fprBOOST, tprBOOST, thresholdBOOST = roc_curve(y_trainNew, GBC.predict_proba(X_trainNew[vars])[:,1])
fprNEURAL, tprNEURAL, thresholdNEURAL = roc_curve(y_trainNew, clf.predict_proba(X_trainNew[vars1])[:,1])
fprLDA, tprLDA, thresholdLDA = roc_curve(y_train, model.predict_proba(X_train_lda)[:,1])


roc_aucBOOST = auc(fprBOOST, tprBOOST)
roc_aucNEURAL = auc(fprNEURAL, tprNEURAL)
roc_aucLDA = auc(fprLDA, tprLDA)

pyplot.xlabel("Iterations")
pyplot.ylabel("Train Error")
plt.plot(clf.loss_curve_,label='Train set')
plt.plot(clf.loss_curve_,label='Test set')
pyplot.legend(loc="upper right")
pyplot.show()

pyplot.figure()
lw = 2
pyplot.plot(
    fprBOOST,
    tprBOOST,
    color="red",
    lw=lw,
    label="Boost Decision Tree (area = %0.2f)" % roc_aucBOOST,
)
pyplot.plot(
    fprNEURAL,
    tprNEURAL,
    color="green",
    lw=lw,
    label="Neural (area = %0.2f)" % roc_aucNEURAL,
)


pyplot.plot(
    fprLDA,
    tprLDA,
    color="black",
    lw=lw,
    label="LDA (area = %0.2f)" % roc_aucLDA,
)
pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.title("Receiver operating characteristic example")
pyplot.legend(loc="lower right")
pyplot.show()

#____________________________________________________________________________
#____________________________________________________________________________

#kolmogorov-smirnov histograms (LDA, Neural, Boost Tree)
#---------------------------------------------------------------
#LDA_HISTOGRAM

P_bad_LDA = model.predict_proba(X_train_lda)[:,0]
P_good_LDA = model.predict_proba(X_train_lda)[:,1]

# plot the probability distribution for the train sample: sample_LDA
plot2 = pyplot.figure(2)
bins = numpy.linspace(-0.1, 1.1, 48) # fixed bin size
pyplot.hist(P_bad_LDA, bins=bins, alpha=0.2,label='bad')
pyplot.hist(P_good_LDA, bins=bins, alpha=0.2,label='good')
pyplot.title('Probability distributions LDA')
pyplot.xlabel('probability')
pyplot.ylabel('count')
pyplot.yscale('log')
pyplot.legend(loc='upper center')
pyplot.show()
#-------------------------------------------
#NEURAL_HISTOGRAM
for k in vars:
      vars1 = vars.copy()
      vars1.remove(k)
      print(vars1)

P_bad_NEURAL = clf.predict_proba(X_trainNew[vars1])[:,0]
P_good_NEURAL = clf.predict_proba(X_trainNew[vars1])[:,1]

# plot the probability distribution for the train sample: NEURAL
plot2 = pyplot.figure(3)
bins = numpy.linspace(-0.1, 1.1, 48) # fixed bin size
pyplot.hist(P_bad_NEURAL, bins=bins, alpha=0.2,label='bad')
pyplot.hist(P_good_NEURAL, bins=bins, alpha=0.2,label='good')
pyplot.title('Probability distributions NEURAL')
pyplot.xlabel('probability')
pyplot.ylabel('count')
pyplot.yscale('log')
pyplot.legend(loc='upper center')
pyplot.show()

#----------------------------------------------------------
#BOOSTED_TREE_HISTOGRAM

P_bad_BOOST = GBC.predict_proba(X_trainNew)[:,0]
P_good_BOOST = GBC.predict_proba(X_trainNew)[:,1]


# plot the probability distribution for the train sample: BOOSTED TREE
plot2 = pyplot.figure(4)
bins = numpy.linspace(-0.1, 1.1, 48) # fixed bin size
pyplot.hist(P_bad_BOOST, bins=bins, alpha=0.2,label='bad')
pyplot.hist(P_good_BOOST, bins=bins, alpha=0.2,label='good')
pyplot.title('Probability distributions GRADIENT BOOSTED TREE')
pyplot.xlabel('probability')
pyplot.ylabel('count')
pyplot.yscale('log')
pyplot.legend(loc='upper center')
pyplot.show()

#___________________________________________________________________
#Features Importances

from sklearn.datasets import make_classification

#------------
#LDA 
modelLDA = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
# fit the model
modelLDA.fit(X_trainNew, y_train)
# get importance
importance = modelLDA.coef_
importance = importance.ravel()
# summarize feature importance
plot3=pyplot.figure(5)
pyplot.title('LDA - Feature Importances')
pyplot.xlabel('Features')
pyplot.ylabel('Importances')
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

#-------------
#NeuralNetwork

from sklearn.linear_model import Perceptron

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X_trainNew, y_trainNew)

coeffs = clf.coef_
coeffs=coeffs.ravel()/10

plot3=pyplot.figure(5)
pyplot.title('Neural Network - Feature Importances')
pyplot.xlabel('Features')
pyplot.ylabel('Importances')
for i,v in enumerate(coeffs):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(coeffs))], coeffs)
pyplot.show()
#-----------------------
#Boosted Tree

from numpy import loadtxt
from matplotlib import pyplot

# fit model no training data
modelGBC = GradientBoostingClassifier()
modelGBC.fit(X_trainNew, y_trainNew)
# feature importance
print(modelGBC.feature_importances_)
# plot
plot3=pyplot.figure(6)
pyplot.title('Boost Tree - Feature Importances')
pyplot.xlabel('Features')
pyplot.ylabel('Importances')
pyplot.bar(range(len(modelGBC.feature_importances_)), modelGBC.feature_importances_)
pyplot.show()




#---------------------------------------------------------------------------------------
#TSORMPATZOGLOU IOANNIS
#DRAGONA DIONYSIA
#ATHENS 2022 























