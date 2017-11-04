import numpy as np
import warnings
from numpy import linalg
import sys
import math
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

dataset=[]

def read_data():
	f=open("Data_SVM.csv", 'r')
	lines=[line.strip() for line in f.readlines()]
    
    # f.close()
	lines=[line.split(",") for line in lines if line]
	X=np.array([line[:2] for line in lines if line[-1]=="1" or line[-1]=="-1"], dtype=np.float)
	# class2=np.array([line[:2] for line in lines if line[-1]=="-1"], dtype=np.float)
	Y=np.array([line[2] for line in lines if line[-1]=="1" or line[-1]=="-1"], dtype=np.int)
	Z1=Y.tolist()

	dataset.append(lines)

	print X
	print Z1
	# print dataset

	# dataset.pop(0)
	# Z1.pop(0)
	# X.pop(0)
	# for i in Z1:
	# 	i=float(i)
	# for i in X:
	# 	i[0]=float(i[0])
	# 	i[1]=float(i[1])
	# for i in dataset:
	# 	for j in i:
	# 		j=float(j)

	# print class2
	# print lines


	x1=X[:,0]
	x2=X[:,1]
	# for i in range(len(X)):
	# 	x1.append(X[i][0])
	# 	x2.append(X[i][1])


	# return class1, class2

	# c=[1,10,100,1000,10000]
	# d=[2,3,4]

	# data=np.array(dataset)
	# print data

	# # clf=svm.SVC(kernel='poly',C=1,degree=3)
	# # clf.fit(X,Z1)


	# for i in c:
	# 	for j in d:
	# 		accuracy=[]
	# 		for val in range(0,30):
	# 			kf = KFold(n_splits=10,shuffle=True)
	# 			for test,train in kf.split(data):
	# 				Xtrain, Xtest, Ytrain, Ytest = data[train][:,:2], data[test][:,:2], data[train][:,2], data[test][:,2]

	# 				svc=svm.SVC(C=i,degree=j,kernel='poly')
	# 				svc.fit(Xtrain,Ytrain)

	# 				acc=svc.score(Xtest,Ytest)
	# 				# print acc
	# 				accuracy.append(acc*100)

	# 		mean=0.0
	# 		for k in accuracy:
	# 			mean=mean+k
	# 		mean=(1.0*mean)/len(accuracy)	
	# 		sd=0.0
	# 		for k in accuracy:
	# 			sd=sd+((k-mean)*(k-mean))
	# 		sd=sd/len(accuracy)
	# 		sd=math.sqrt(sd)
	# 		print "c = ",
	# 		print i
	# 		print "d = ",
	# 		print j
	# 		print mean	
	# 		print sd


	C=10000
	h=0.2
	svc = svm.SVC(kernel='poly', C=C,degree=2).fit(X, Z1)

	# x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	# y_min, y_max = X[:, -1].min() - 1, X[:, -1].max() + 1
	# h = (x_max / x_min)/100

	x_min, x_max = -2, 2
	y_min, y_max = -2, 2

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

	# print xx
	# print yy
 	# plt.subplot(1, 1, 1)

	Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

	plt.scatter(x1, x2, c=Z1, edgecolor='black', cmap=plt.cm.Paired)

	axis=plt.gca()
	axis.set_xlim([-1.5,1.5])
	axis.set_ylim([-1.5,1.5])
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	# plt.xlim(xx.min(), xx.max())
	# plt.title('SVC with nonlinear kernel')
	# plt.axis('tight')
	plt.show()

read_data()

# def main():

# 	class1, class2=read_data()
# 	print class1
# 	print class2

# 	x1=[]
# 	y1=[]
# 	x2=[]
# 	y2=[]


# 	plt.ylim(-1.5,1.5)
# 	plt.xlim(-1.5,1.5)

# 	for x in class1:
# 		x1.append(x[0])
# 		y1.append(x[1])

# 	plt.plot(x1,y1,"ro")	

# 	for x in class2:
# 		x2.append(x[0])
# 		y2.append(x[1])

# 	plt.plot(x2,y2, "go")

# 	plt.show()

# main()	