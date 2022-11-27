import numpy as np
from sklearn import datasets as ds
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.colors as colors
from matplotlib.markers import MarkerStyle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import sys
from sklearn import tree
import csv
from sklearn.base import clone



np.set_printoptions(threshold=sys.maxsize)
	
""" ADABOOST IMPLEMENTATION ================================================="""
def adaboost_clf(Y_train, X_train, M, clf, algo, classcount, a, a0):
	
	n_train = len(X_train) # get data point count
	weights = np.empty(shape=(M, n_train)) # create array for weights to be saved at each iteration
	partitions = np.array([]) # save partitions, which are used to normalize weights after update
	unsumpartitions = np.empty(shape=(M, n_train)) # save the individual values that sum together to make patition
	classcount = len(np.unique(Y_train)) # count of unique labels to be used
	estimators = np.array([]) # save trees that act as hypotheses for adaboost
	alphas = np.array([]) # learning coefficients
	edges = np.array([])
	mist_dichs = np.empty(shape=(T, len(X_train)))
	
	w = np.ones(n_train, dtype=float) 
	w = w.astype(float)
	w = w / float(n_train)# Initialize weights, normalized

	
	for i in range(M):
		#  Fit a classifier with the specific weights
		
		if i == 0:
		
			clf.fit(X_train, Y_train, sample_weight=w) # fit classifier using data and weights
			sigfeature = int(np.nonzero(clf.feature_importances_ != 0)[0][0]) # count significant features
			X_col = X_train[ : , sigfeature]

			index1 = np.argsort(X_col) # sort columns of data features so that certain printouts return nicer

			X_train = X_train[index1]
			Y_train = Y_train[index1]
			clf = DecisionTreeClassifier(max_depth = a, max_leaf_nodes = a0, criterion='entropy', splitter='best')
		
		clf.fit(X_train, Y_train, sample_weight=w)

		estimators = np.append(estimators, clf) # save classifiers

		pred_train_i = clf.predict(X_train) # predict data feature labels using classifier

		miss = np.array([int(-1) if x==1 else int(1) for x in (pred_train_i != Y_train)]) # check for wrong predictions
		Y_label = np.array([int(-1) if x==1 else int(1) for x in Y_train]) # insure that labels are 1 and -1 for binary classification

		edge_m = np.dot(miss, w) # calculate edge for weight update, which is 1 - 2 * (error of classifier)
	
		edges = np.append(edges, edge_m) # array of edges
		mist_dichs[i] = np.array([np.ubyte(1) if x == int(-1) else np.ubyte(0) for x in miss], dtype=np.ubyte)	# mistake dichotomies in (-1)-1 form

		weights[i] = w # array of weights
		
		if algo == 'Binary Classification AdaBoost': # run specific choice of algorithm version
		
			alpha = (1 / 2) * np.log((1 + edge_m) / (1 - edge_m))
		
		if algo == 'SAMME':
			
			alpha = (1 / 2) * np.log((1 + edge_m) / (1 - edge_m)) + (1 / 2) * np.log(classcount - 1) # removing the 1/2 does something weird

		if algo == 'TEST':
		
			alpha = (1 / 2) * np.log((1 + edge_m) / (1 - edge_m)) - (1 / 2) * np.log(1 / relcount)

		if i == 0:
		
			unsumpartitions[0] = np.zeros(n_train) + (miss * alpha)
			
		else:
			
			unsumpartitions[i] = unsumpartitions[i - 1] + (miss * alpha)
			
		partitions = np.append(partitions, np.sum(w))
		
		w = w * np.exp(-miss * alpha) #update weights and normalize
		w /= np.sum(w) 
		alphas = np.append(alphas, alpha)

	print('end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')		

	return (weights, edges, mist_dichs, alphas, estimators, partitions, unsumpartitions) # weights, check, edges, dichs, classcount in Y_train, estimators, alphas
		

print('###################################################################')

def data_csv(name, sample, excel=False):
	
	
	if excel==False:
	
		dat = pd.read_csv(name + '.txt', sep='\t')
		
	else:
	
		dat = pd.read_excel('Dry_Bean_Dataset.xlsx') # dry bean is best for high data pt cycling
		datfile = np.array([np.array(datfile[i][0].split(',')) for i in range(len(datfile))])
		
		#  example encoding for Dry Bean Dataset
		#  Y = np.array([int(0) if x == 'SEKER' else (int(1) if x == 'BARBUNYA' else (int(2) if x == 'BOMBAY' else (int(3) 
		#  if x == 'CALI' else (int(4) if x == 'HOROZ' else (int(5) if x == 'SIRA' else int(6)))))) for x in Y])
		
	datfile = dat.to_numpy() # for .data files or csv's

	np.random.shuffle(datfile) #  shuffle to return fresh sampling 
							   #  with each call when sample is smaller that data set size

	X = datfile[ : ,  : -1 ]
	Y = datfile[ : , -1 :  ]
	Y = np.array([Y[i][0] for i in range(len(Y))])

	

	X = X[:sample]
	Y = Y[:sample]
	
	return X,Y
	
def find_defects(W0): #  this function outputs iterations at which the algorithm is NOT cycling
					  #  i used this to find suitable datasets that had a likely change of cycling to
					  #  add to my paper. many datasets will not cycle, and hence wont even get close to doing so

	mist_check = np.array([], dtype=np.ubyte)
	defect_sum = np.array([], dtype=int)
	Echeck = np.array([])

	for i in range(len(E0list)): 
		W00 = W0[i]

		if i < len(E0list) - 1:

			mist_check = np.ubyte(mist_dichs[i] + mist_dichs[i + 1])
			
		nabla = True
		zip = np.dstack((mist_check, W00))[0].astype(np.ubyte) # zip together mist_check and W0[i]
		defects = np.array([np.ubyte(1) if x[0] == 2 and x[1] == 1 else np.ubyte(0) for x in zip], dtype=np.ubyte)#finds defects that are also support vectors
		defect_sum = np.append(defect_sum, int(np.sum(defects)))

		if 1 in defects: # accounts for non-support vectors
			
			nabla = False
		
		if nabla == True:
			
			# print('dich.s', i, 'and', i + 1, 'have nabla and', int(np.sum(W00)), 'support vectors')
			Echeck = np.append(Echeck, 1)
			
		if nabla == False:
		
			# print('dich.s' ,i , 'and', i + 1, 'do not have nabla with', np.count_nonzero(defects == 1), 'defects and', int(np.sum(W00)), 'support vectors')
			Echeck = np.append(Echeck, 0)

	for i in range(int(len(defect_sum) / scale)):
		
		total = np.sum(defect_sum[i * scale:(i + 1) * scale])
		print('we have', total, 'defects with', b, 'data points for iterations', i * scale, 'to', (i + 1) * scale)
		supvecs = int(np.sum(W0[i * scale:(i + 1) * scale]))
		print('we have', supvecs, 'support vectors with', b, 'data points for iterations', i * scale, 'to', (i + 1) * scale)

	for i in range(1, int(T / scale)):
		
		print(np.nonzero(W0[int(i * scale)] == 1)[0])
		
	return Echeck # array that contains information on the propotion of potential cycling condition in datasets
	
scale = 1000
T = 5 * 1000
a = 3 # tree depth
a0 = 4 # max leaves 
minvalscale = 10000000 # set scale to detect as a support vector, which have very small weight that converges to 0
sample = 200
algos = np.array(['Binary Classification AdaBoost', 'SAMME', 'TEST'])
algo = algos[0] #'SAMME' or 'Binary Classification AdaBoost' or 'TEST'
sample = int(sample)
data_from = 'sklearn'

if data_from == 'csv':
	
	data_csv('seeds_dataset', sample)

# for sklearn datasets

if data_from == 'sklearn':

	points = ds.load_iris()

	X = points.data
	Y = points.target

	index0 = np.array([i for i in range(len(X))])

	np.random.shuffle(index0)
	X = X[index0]
	Y = Y[index0]
	
	
	if sample < len(X):
	
		X = X[:sample]
		Y = Y[:sample]
	
	
features = len(X[0])
X_train = X
Y_train = Y
classcount = len(np.unique(Y_train))
b = len(X_train) # data points

# run adaboost from scratch algorithm, functions like sklearn algorithm
clf_tree = DecisionTreeClassifier(max_depth = a, max_leaf_nodes = a0, criterion='entropy', splitter='best')
B = adaboost_clf(Y_train, X_train, T, clf_tree, algo, classcount, a, a0)
weights = B[0]
edges = B[1]
mist_dichs = B[2]
alphas = B[3]
estimators = B[4]
partitions = B[5]
unsumpartitions = B[6]
mist_pm_one_form = np.array([np.array([float(-1) if x == np.ubyte(1) else float(1) for x in mist_dichs[i]], dtype=np.ubyte) for i in range(len(mist_dichs))])



minval = 1 / (minvalscale * b) # find non-support vector data points
W0 = np.array([np.array([np.ubyte(0) if x < minval else np.ubyte(1) for x in weights[i]]) for i in range(T)], dtype=np.ubyte)


indices = np.unique(mist_dichs, axis=0, return_index=True)[1]
index = np.sort(indices)
E0unique = mist_dichs[index].astype(np.ubyte)
E0list = np.array([], dtype=int)
print('making E0list')

for i in range(len(mist_dichs)): # updated algo for getting for retrieving dich. order in a run
	
	res = (E0unique == mist_dichs[i])
	
	for j in range(len(res)):
		
		checkthing = False
		
		if False not in res[j]:
			
			jj = int(j)
			checkthing = True
			break
	 
	if checkthing == True:
	
		E0list = np.append(E0list, jj)

Echeck = find_defects(W0)		

print('algorithm:', algo) #  print various important values and parameters of a run of AdaBoost
print('sample size: ', len(X))
print('training sample size: ', b)
print('testing sample size: ', len(X) - b)
print('class count:', classcount)
print('tree depth: ', a)
print('max tree leaves: ', a0)
print('data feature count:', features)
print('relevant feature count of final estimator:', int(np.count_nonzero(estimators[-1].feature_importances_ != 0)))
print('iterations: ', T)
print('number of unique dich.s:', len(np.unique(E0list)))
print('number of unique dich.s in second half:', len(np.unique(E0list[int(len(E0list) / 2):])))
print('number of unique dich.s in last tenth:', len(np.unique(E0list[int(len(E0list) / 10):])))
print('proportion nabla iterations:', np.sum(Echeck) / len(Echeck))
print('average edge: ', np.round(float(np.sum(edges[15:])/(len(edges[15:]))), 10))

print('###################################################################')


R = np.array([t for t in range(len(weights))])
R = R.astype(float)
for i in range(len(edges)):
	edges[i] = float(edges[i])
edges = edges.astype(float)




plt.scatter(R, edges, c='red', s=1, facecolors='none', cmap='viridis', linewidths=1, alpha=1)
plt.ylabel("Edge values")
plt.xlabel("Iterations of AdaBoost")
plt.show()

print('###################################################################')

