#Titanic Survival Prediction

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier

sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(1,1), random_state=0)

df=pd.read_csv("C:/Users/Riya/Downloads/tested.csv")

#print(df)

#print(x)
#print(y)

#df.info()

#print(df.isna().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
#print(df.isna().sum())

df['Sex'].replace({'female':0, 'male':1}, inplace=True)
df['Embarked'].replace({'Q':0, 'S':1, 'C':2}, inplace=True)

x8=df.drop("Pclass",axis=1)
x7=x8.drop("Survived",axis=1)
x6=x7.drop("Name",axis=1)
x5=x6.drop("SibSp",axis=1)
x4=x5.drop("Parch",axis=1)
x3=x4.drop("Ticket",axis=1)
x2=x3.drop("Cabin",axis=1)
x=x2.drop("Embarked",axis=1)
y=df["Survived"]

#print(x)
#print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)

sv.fit(x_train,y_train)
y_svp=sv.predict(x_test)

nn.fit(x_train,y_train)
y_nnp=nn.predict(x_test)


print('SVM : ' ,accuracy_score(y_test,y_svp))
print('Neural Networks : ' ,accuracy_score(y_test,y_nnp))
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_knnp=knn.predict(x_test)
print('KNN : ' ,accuracy_score(y_test,y_knnp))

#Feature engineering using PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(x)
x=pca.transform(x)

# from imblearn.over_sampling import RandomOverSampler
# ros=RandomOverSampler(random_state=1)
# x,y=ros.fit_resample(x,y)

# from imblearn.over_sampling import SMOTE
# sms=SMOTE(random_state=1)
# x,y=sms.fit_resample(x,y)

# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model = ExtraTreesClassifier()
# model.fit(x,y)
# print(model.feature_importances_)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# bestfeaures = SelectKBest(score_func=chi2, k='all')
# fit = bestfeaures.fit(x,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(x.columns)
# featuresScores = pd.concat([dfcolumns,dfscores],axis=1)
# featuresScores.columns=['Specs','Score']
#
# print(featuresScores)
# feat_importance = pd.Series(model.feature_importances_,index=x.columns)
# feat_importance.nlargest(4).plot(kind='barh')
# plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)

sv.fit(x_train,y_train)
y_svp=sv.predict(x_test)

nn.fit(x_train,y_train)
y_nnp=nn.predict(x_test)

print('')
print('After feature engineering ')
print('SVM : ' ,accuracy_score(y_test,y_svp))
print('Neural Networks : ' ,accuracy_score(y_test,y_nnp))
knn.fit(x_train,y_train)
y_knnp=knn.predict(x_test)
print('KNN : ' ,accuracy_score(y_test,y_knnp))


'''
OUTPUT 
SVM :  0.6746031746031746
Neural Networks :  0.6746031746031746
KNN :  0.6984126984126984

After feature engineering 
SVM :  0.6825396825396826
Neural Networks :  0.6746031746031746
KNN :  0.6984126984126984

Process finished with exit code 0
'''