import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

#looking into the raw data
peek = data.head(20) #the first 20 rows
print(peek)

#viewing the dimention/shape/sixe of your dataset by first peeking into the data
print(data.shape)

#knowing the type of attribute for the purpose of classification
types = data.dtypes
print(types)

#descriptive statistic that can give one a great insight on how to shape each attribute
pandas.set_option('display.width', 100)
pandas.set_option('precision', 3)
description = data.describe()
print(description)

#distribution of class attributes inorder to know how to balance the class values
class_counts = data.groupby('class').size()
print(class_counts)

#using persion correlation mthd to calculate the correlation(rltship between 2 var and how they may/maynot change together) between the attribute
pandas.set_option('display.width', 100)
pandas.set_option('precision', 3)
correlations = data.corr(method='pearson')
print(correlations)

#determine the skewness(extention of shift/squash of a curve to another direction) of the data using gaussian distribution
skew = data.skew()
print(skew)
