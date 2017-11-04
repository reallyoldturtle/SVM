import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
import csv
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def fitandpredict(x,y,X,Y,c,deg,h,dataset):
	accuracy=[]
	for i in range(0,30):
		k_fold=KFold(n_splits=10,shuffle=True)
		for train,test in k_fold.split(dataset):
			x_train=dataset[train][:,:2]
			y_train=dataset[train][:,2]

			x_test=dataset[test][:,:2]
			y_test=dataset[test][:,2]

			svc=svm.SVC(C=c,degree=deg,kernel='poly')
			svc.fit(x_train,y_train)

			accuracy.append(svc.score(x_test,y_test)*100)

	# mean=float(sum(accuracy)/len(accuracy))
	mean=np.mean(accuracy)
	var=0.0
	for k in accuracy:
		var=var+((k-mean)*(k-mean))
	var=var/len(accuracy)
	sd=math.sqrt(var)

	print"----------"
	print"----------"
	print "c=",c
	print "degree=",deg
	print "mean=",mean
	print "SD=",sd

	return mean




#read the file
x=[]
y=[]
line=[]
with open('Data_SVM.csv',"rb") as fo:
	reader=csv.reader(fo)
	for row in reader:
		# print row[0]
		x.append((row[:2]))
		y.append((row[2]))
		line.append(row)

line.pop(0)
x.pop(0)
y.pop(0)
X=np.array(x).astype(np.float)
Y=np.array(y).astype(np.int)

# print X,Y
h=.02 # step size in the mesh

x=X[:,0]
y=X[:,1]
c=[1,10,100,1000,10000,20000]
d=[2,3,4]
C=10000
h=0.2
dataset=np.array(line)
max_val=0.0
max_c,max_d=0,0
for c_val in c:
	for deg_val in d:
		avg_acc=fitandpredict(x,y,X,Y,c_val,deg_val,h,dataset)

		if(avg_acc>max_val):
			max_val=avg_acc
			max_c=c_val
			max_d=deg_val


print "\n\n---"
print "Max C=",max_c
print "Max D=",max_d
print "Max Acc=",max_val

# plot the data acc to this max_c and max_d
svc = svm.SVC(kernel='poly', C=max_c,degree=max_d).fit(X,Y)
x_min, x_max = -2, 2
y_min, y_max = -2, 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.PuOr_r)

plt.contourf(xx, yy, Z, cmap=plt.cm.PuOr_r, alpha=0.8)

plt.scatter(x, y, c=Y, cmap=plt.cm.PuOr_r,edgecolor="black")

axis=plt.gca()
axis.set_xlim([-1.5,1.5])
axis.set_ylim([-1.5,1.5])
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()









