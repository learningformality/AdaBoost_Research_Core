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


def adaboost_clf(Y_train, X_train, M, clf, edges, mist_dichs, algo, classcount, a, a0):
    n_train = len(X_train)  # get data point count
    weights = np.empty(shape=(M, n_train))  # create array for weights to be saved at each iteration
    partitions = np.array([])  # save partitions, which are used to normalize weights after update
    unsumpartitions = np.empty(shape=(M, n_train))  # save the individual values that sum together to make patition
    mist_dichs_out = np.copy(mist_dichs)  # list that is final output of mistake dichotomies from algorithm
    classcount = len(np.unique(Y_train))  # count of unique labels to be used
    estimators = np.array([])  # save trees that act as hypotheses for adaboost
    alphas = np.array([])  # learning coefficients

    w = np.ones(n_train, dtype=float)
    w = w.astype(float)
    w = w / float(n_train)  # Initialize weights, normalized

    for i in range(M):
        # Fit a classifier with the specific weights

        if i == 0:
            clf.fit(X_train, Y_train, sample_weight=w)
            sigfeature = int(np.nonzero(clf.feature_importances_ != 0)[0][0])
            X_col = X_train[:, sigfeature]

            index1 = np.argsort(X_col)

            X_train = X_train[index1]
            Y_train = Y_train[index1]
            clf = DecisionTreeClassifier(max_depth=a, max_leaf_nodes=a0, criterion='entropy', splitter='best')

        clf.fit(X_train, Y_train, sample_weight=w)

        estimators = np.append(estimators, clf)

        pred_train_i = clf.predict(X_train)

        miss = np.array([int(-1) if x == 1 else int(1) for x in (pred_train_i != Y_train)])
        Y_label = np.array([int(-1) if x == 1 else int(1) for x in Y_train])
        hypothesis = np.multiply(miss, Y_label)

        edge_m = np.dot(miss, w)

        edges = np.append(edges, edge_m)  # array of edges
        mist_dichs[i] = np.array([np.ubyte(1) if x == int(-1) else np.ubyte(0) for x in miss],
                                 dtype=np.ubyte
                                 )  # mistake dichotomies in (-1)-1 form
        mist_dichs_out[i] = hypothesis.astype(int)

        weights[i] = w  # array of weights

        if algo == 'Binary Classification AdaBoost':
            alpha = (1 / 2) * np.log((1 + edge_m) / (1 - edge_m))

        if algo == 'SAMME':
            alpha = (1 / 2) * np.log((1 + edge_m) / (1 - edge_m)) + (1 / 2) * np.log(
                classcount - 1
            )  # removing the 1/2 does something weird

        if algo == 'TEST':
            alpha = (1 / 2) * np.log((1 + edge_m) / (1 - edge_m)) - (1 / 2) * np.log(1 / relcount)

        w = w * np.exp(-miss * alpha)

        if i == 0:

            unsumpartitions[0] = np.zeros(n_train) + (miss * alpha)

        else:

            unsumpartitions[i] = unsumpartitions[i - 1] + (miss * alpha)

        partitions = np.append(partitions, np.sum(w))

        w /= np.sum(w)
        alphas = np.append(alphas, alpha)

    print('end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    return (weights, check, edges, mist_dichs, alphas, estimators, partitions, unsumpartitions,
            mist_dichs_out)  # weights, check, edges, dichs, classcount in Y_train, estimators, alphas


print('###################################################################')

edges = np.array([])
check = False
scale = 1000
T = 5 * scale
a = 3  # tree depth
a0 = 4  # max leaves
minvalscale = 10000000  # set scale to detect as a support vector, which have very small weight that converges to 0
sample = 200
algos = np.array(['Binary Classification AdaBoost', 'SAMME', 'TEST'])
algo = algos[0]  # 'SAMME' or 'Binary Classification AdaBoost' or 'TEST'
sample = int(sample)

'''
dat = pd.read_csv('seeds_dataset.txt', sep='\t')

#dat = pd.read_excel('Dry_Bean_Dataset.xlsx') #dry bean is best for high data pt cycling

datfile = dat.to_numpy() #for .data files or csv's

np.random.shuffle(datfile)

#datfile = np.array([np.array(datfile[i][0].split(',')) for i in range(len(datfile))])

X = datfile[ : ,  : -1 ]
Y = datfile[ : , -1 :  ]
Y = np.array([Y[i][0] for i in range(len(Y))])
'''
# Y = np.array([int(0) if x == 'SEKER' else (int(1) if x == 'BARBUNYA' else (int(2) if x == 'BOMBAY' else (int(3) if x == 'CALI' else (int(4) if x == 'HOROZ' else (int(5) if x == 'SIRA' else int(6)))))) for x in Y])
'''
X = X[:sample]
Y = Y[:sample]


print(datfile)
print(X)
print(Y)
print(len(X))
quit()
'''
# for sklearn datasets

points = ds.load_iris()

X = points.data
Y = points.target

index0 = np.array([i for i in range(len(X))])

np.random.shuffle(index0)
X = X[index0]
Y = Y[index0]

features = len(X[0])
'''
X = X[:sample]
Y = Y[:sample]
'''
# Y_train = np.array([int(-1) if x == 1 else int(1) for x in Y_train])


X_train = X
Y_train = Y

classcount = len(np.unique(Y_train))

b = len(X_train)  # data points
'''
X = X[:sample]
Y = Y[:sample]

X_train = X_train[:sample]
Y_train = Y_train[:sample]

import codecs #for csv files

with codecs.open('SomervilleHappinessSurvey2015.csv', 'rU', 'utf-16') as file:
	
	read = csv.reader(file, delimiter=',')
	
	for row in read:

		unproc_data.append(np.array(row))

unproc_data = np.array(unproc_data[1:]) #unprocessed data
unproc_data = np.array([unproc_data[i].astype(int) for i in range(len(unproc_data))])

X = unproc_data[ : , 1 : ]
Y = unproc_data[ : , 0 : 1]
Y = np.array([Y[i][0] for i in range(len(Y))])
'''
'''
print(X)
print(Y)


X = X[:sample]
Y = Y[:sample]

X_train = X_train[:sample]
Y_train = Y_train[:sample]
'''

clf_tree = DecisionTreeClassifier(max_depth=a, max_leaf_nodes=a0, criterion='entropy', splitter='best')

# run adaboost from scratch algorithm, functions like sklearn algorithm
B = adaboost_clf(Y_train, X_train, T, clf_tree, edges, mist_dichs, algo, classcount, a, a0)
weights = B[0]
edges = B[2]
mist_dichs = B[3]
alphas = B[4]
estimators = B[5]
partitions = B[6]
unsumpartitions = B[7]
mist_dichs_out = B[8]
mist_pm_one_form = np.array(
    [np.array([float(-1) if x == np.ubyte(1) else float(1) for x in mist_dichs[i]], dtype=np.ubyte) for i in
     range(len(mist_dichs))]
)

minval = 1 / (minvalscale * b)
W0 = np.array([np.array([np.ubyte(0) if x < minval else np.ubyte(1) for x in weights[i]]) for i in range(T)],
              dtype=np.ubyte
              )

indices = np.unique(mist_dichs, axis=0, return_index=True)[1]
index = np.sort(indices)
E0unique = mist_dichs[index].astype(np.ubyte)
E0list = np.array([], dtype=int)
print('making E0list')

for i in range(len(mist_dichs)):  # updated algo for getting for retrieving dich. order in a run

    res = (E0unique == mist_dichs[i])

    for j in range(len(res)):

        checkthing = False

        if False not in res[j]:
            jj = int(j)
            checkthing = True
            break

    if checkthing == True:
        E0list = np.append(E0list, jj)

mist_check = np.array([], dtype=np.ubyte)
defect_sum = np.array([], dtype=int)
Echeck = np.array([])

for i in range(len(E0list)):
    W00 = W0[i]

    if i < len(E0list) - 1:
        mist_check = np.ubyte(mist_dichs[i] + mist_dichs[i + 1])

    nabla = True
    zip = np.dstack((mist_check, W00))[0].astype(np.ubyte)  # zip together mist_check and W0[i]
    defects = np.array([np.ubyte(1) if x[0] == 2 and x[1] == 1 else np.ubyte(0) for x in zip],
                       dtype=np.ubyte
                       )  # finds defects that are also support vectors
    defect_sum = np.append(defect_sum, int(np.sum(defects)))

    if 1 in defects:  # accounts for non-support vectors

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

'''
alphadich = np.array([alphas[i] * mist_pm_one_form[i] for i in range(len(mist_pm_one_form))])
totalmargins = np.sum(alphadich, axis=0)
#supptotalmargins = np.array([totalmargins[i] for i in range(len(totalmargins)) if W0[-1][i] == 1])
print('normalized support vector margins')
normtotalmargins = totalmargins / np.sum(totalmargins)
print(normtotalmargins)
enttotalmargins = np.array([x * np.log(x) for x in normtotalmargins])
print('entropy of margins distribution:', -np.sum(enttotalmargins), 'with', np.log(b), 'max entropy at data point count')
'''

for i in range(1, int(T / scale)):
    print(np.nonzero(W0[int(i * scale)] == 1)[0])

# print('T:\n',E5)
# print('Tx:\n',E5x)
# print('T diff:\n',E6)
# print('unique dichotomies:\n')

# print('unique dichotomies:',p)
# print('unique matches:',p0 / 2)
# print('d/iter ratio:',p / T)
# print('dimensions: ',t)
print('algorithm:', algo)
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
print('average edge: ', np.round(float(np.sum(edges[15:]) / (len(edges[15:]))), 10))
'''
enumlist = np.array([np.array([np.ubyte(mist_dichs[j][i]) for i in range(len(mist_dichs[0])) if W0[j][i] != 0]) for j in range(len(E0list[T - int(len(E0list) / 10):]))], dtype=object)
enumlist0 = np.array([np.array([unsumpartitions[j][i] for i in range(len(mist_dichs[0])) if W0[j][i] != 0]) for j in range(len(E0list[T - int(len(E0list) / 10):]))], dtype=object)
for i in range(len(enumlist)):
	
	#print(enumlist[i].astype(int), int(E0list[T - int(len(E0list) / 10) + i]), 'dich')
	print(np.array([unsumpartitions[T - int(len(E0list) / 10) + i][j] for j in range(len(W0[0])) if W0[T - int(len(E0list) / 10) + i][j] != 0]))
	print(np.array([unsumpartitions[T - int(len(E0list) / 10) + i][j] for j in range(len(W0[0])) if W0[T - int(len(E0list) / 10) + i][j] != 0]) / np.sum(np.array([unsumpartitions[T - int(len(E0list) / 10) + i][j] for j in range(len(W0[0])) if W0[T - int(len(E0list) / 10) + i][j] != 0])))
	print(np.sum(np.array([unsumpartitions[T - int(len(E0list) / 10) + i][j] for j in range(len(W0[0])) if W0[T - int(len(E0list) / 10) + i][j] != 0])))
	print(np.array([weights[T - int(len(E0list) / 10) + 1 + i][j] for j in range(len(W0[0])) if W0[T - int(len(E0list) / 10) + 1 + i][j] != 0]))
	#print(enumlist0[i].astype(int), int(E0list[T - int(len(E0list) / 10) + i]), 'hyp')
'''
if sklearn_alg == True:
    print('sklearn adaboost score:', np.round(C.score(X_test, Y_test), 10))
    print('sklearn single estimator score:', np.round(D.score(X_test, Y_test), 10))

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
