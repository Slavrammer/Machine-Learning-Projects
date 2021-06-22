# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

os.chdir(fr'C:\Users\moogl\Desktop\Projekty ML\\titanic')
pd.set_option('display.max_columns', None)

# accuire datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# festures in the dataset
print(train_df.columns.values)

# preview the data
print(train_df.head())
print(train_df.tail())
print(train_df.info())
print('_'*40)
print(test_df.info())
print(train_df.describe())
print(train_df.describe(include=['O']))

# analysis of features
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False))
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False))
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False))
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False))
# visualisations of data with histograms
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=2)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()

# Correcting by dropping features
print('Before', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print('After', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

# Creating new features from existing ones
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady',
                                                 'Countess',
                                                 'Capt',
                                                 'Col',
                                                 'Don',
                                                 'Dr',
                                                 'Major',
                                                 'Rev',
                                                 'Sir',
                                                 'Jonkheer',
                                                 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean())

# Categorical to ordinal conversion
title_mapping = {'Mr' : 1, "Miss" : 2, "Mrs" : 3, "Master" : 4, "Rare" : 5}
for dataset in combine:
    dataset["Title"] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head(50))
