# -*- coding: utf-8 -*-
"""
@author: Lahari
"""

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

#numpy, pandas and random
import pandas as pd
import numpy as np
#import random as rnd

#ML models 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

#Data 
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_ids = test_df["PassengerId"]
merge = [train_df, test_df]

#Inspection of data
print('Training data Info')
print(train_df.info())
print('')
print('*'*40)
print('*'*40)
print('')
print('Test data Info')
print(test_df.info())

#Printing the first 5 rows of train and test data
print(train_df.head())
print(test_df.head())

#Data Visualization
#Plots to understand what the best features are
all_data = pd.concat([train_df, test_df], sort = False)
sns.catplot(x = 'Embarked', kind = 'count', data = all_data)
plt.show()

sns.histplot(data=all_data['Fare'], color='teal', kde=True)
plt.show()

sns.histplot(data=all_data['Fare'], color='teal', kde=True)
plt.show()

#Correlation between variables
sns.heatmap(all_data.corr(),annot=True)
plt.show()

#Analysis by using group by feature to pivot the variables
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('')
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('')
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('')
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('')

a = sns.FacetGrid(train_df, col='Survived')
a.map(plt.hist, 'Age', bins=20)

block = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
block.map(plt.hist, 'Age', alpha=.5, bins=20)
block.add_legend();

##### Data preprocessing
#Removing attributes
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
merge = [train_df, test_df]


#Extracting titles from Names
for ds in merge:
    ds['Title'] = ds.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

for ds in merge:
    ds['Title'] = ds['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    ds['Title'] = ds['Title'].replace('Mlle', 'Miss')
    ds['Title'] = ds['Title'].replace('Ms', 'Miss')
    ds['Title'] = ds['Title'].replace('Mme', 'Mrs')
    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for ds in merge:
    ds['Title'] = ds['Title'].map(title_mapping)
    ds['Title'] = ds['Title'].fillna(0)

print(train_df.head())

#Removing attributes
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
merge = [train_df, test_df]
print(train_df.shape, test_df.shape)

#Gender/sex attribute

for ds in merge:
    ds['Sex'] = ds['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

#Guessing age based on the gender/sex 
guess_ages = np.zeros((2,3))
print(guess_ages)

for ds in merge:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = ds[(ds['Sex'] == i) & \
                                  (ds['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            ds.loc[ (ds.Age.isnull()) & (ds.Sex == i) & (ds.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    ds['Age'] = ds['Age'].astype(int)
    
train_df['AgeBand'] = pd.cut(train_df['Age'], 9)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for ds in merge:    
    ds.loc[ ds['Age'] <= 5, 'Age'] = 0
    ds.loc[(ds['Age'] > 5) & (ds['Age'] <= 15), 'Age'] = 1
    ds.loc[(ds['Age'] > 15) & (ds['Age'] <= 20), 'Age'] = 2
    ds.loc[(ds['Age'] > 20) & (ds['Age'] <= 25), 'Age'] = 3
    ds.loc[(ds['Age'] > 25) & (ds['Age'] <= 35), 'Age'] = 4
    ds.loc[(ds['Age'] > 35) & (ds['Age'] <= 45), 'Age'] = 5
    ds.loc[(ds['Age'] > 45) & (ds['Age'] <= 55), 'Age'] = 6
    ds.loc[(ds['Age'] > 55) & (ds['Age'] <= 65), 'Age'] = 7
    ds.loc[ ds['Age'] > 65, 'Age']

train_df = train_df.drop(['AgeBand'], axis=1)
merge = [train_df, test_df]
print(train_df.head())

#Family size
for ds in merge:
    ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))


for ds in merge:
    ds['IsAlone'] = 0
    ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

#Dropping Parch attribute
train_df = train_df.drop(['Parch'], axis=1)
test_df = test_df.drop(['Parch'], axis=1)
merge = [train_df, test_df]


#Age and class attrbutes combination
for ds in merge:
    ds['Age*Class'] = ds.Age * ds.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

#Embarked
freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)

for ds in merge:
    ds['Embarked'] = ds['Embarked'].fillna(freq_port)
    
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for ds in merge:
    ds['Embarked'] = ds['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
#Fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print(test_df.head())

train_df['FareBand'] = pd.qcut(train_df['Fare'], 2)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

for ds in merge:
    ds.loc[ ds['Fare'] <= 8.662, 'Fare'] = 0
    ds.loc[(ds['Fare'] > 8.662) & (ds['Fare'] <= 26), 'Fare'] = 1
    ds.loc[ ds['Fare'] > 26, 'Fare'] = 2
     
    
    ds['Fare'] = ds['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
merge = [train_df, test_df]

#Printing the first 5 rows of train and test data
print(train_df.head(10))

print(test_df.head(10))

#Final data preparation step 
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

#Models
#Model used for Kaggle submission - Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('Logistic Regression accuracy score:',acc_log)

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))

#Exporting predictions to .csv file
submission_df = pd.DataFrame({"PassengerId":test_ids.values,
                              "Survived": Y_pred, })
submission_df.to_csv('submission.csv', index=False)


#Others models used for predictions
# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print('SVM  accuracy score:',acc_svc)


#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('KNN  accuracy score:',acc_knn)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print('Gaussian Naive Bayes  accuracy score:',acc_gaussian)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print('Perceptron  accuracy score:',acc_perceptron)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print('Linear SVC accuracy score:',acc_linear_svc)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print('SGD  accuracy score:',acc_sgd)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('Decision tree accuracy score:',acc_decision_tree)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Random forest accuracy score:',acc_random_forest)

#XGBoost
#from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
model.score(X_train, Y_train)
acc_xgboost = round(model.score(X_train, Y_train) * 100, 2)
print('XGBoost  accuracy score:', acc_xgboost)

#All model accuracies
print('All model accuracies')
print('')
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'XGBoost'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_xgboost]})
print(models.sort_values(by='Score', ascending=False))




