'''
iris dataset is found in sklearn library
uses supervised learning
150 samples(50 per calss in dimensionality of 4)
4 features(petal width and length & sepal width and length )
3 labels(Iris-setosa,Iris-versicolor, ris-virginica)
'''
#download iris dataset
from sklearn import datasets
iris = datasets.load_iris()

#assign features(x-axis) and labels(y-axis) to their specific variables
x = iris.data
y = iris.target

#split the dataset and you train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,t_test = train_test_split(x,y,test_size=0.5)

#
