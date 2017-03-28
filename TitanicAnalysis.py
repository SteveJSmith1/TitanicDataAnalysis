#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:11:14 2017

@author: steve
"""

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Reading Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# Describing data

print(train_df.columns.values)
"""
['PassengerId' 'Survived' 'Pclass' ..., 'Fare' 'Cabin' 'Embarked']
"""

print(test_df.columns.values)
"""
['PassengerId' 'Pclass' 'Name' ..., 'Fare' 'Cabin' 'Embarked']
"""

# Categorical data:
    
# Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.

# Numerical data:
    
# Continuous: Age, Fare. Discrete: SibSp, Parch.


#--------------------------------

# Previewing the data

train_df.head()
"""
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  
"""

# Mixed data types

# Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.

# Which features may contain errors or typos?

# Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.

train_df.tail()

"""
     PassengerId  Survived  Pclass                                      Name  \
886          887         0       2                     Montvila, Rev. Juozas   
887          888         1       1              Graham, Miss. Margaret Edith   
888          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   
889          890         1       1                     Behr, Mr. Karl Howell   
890          891         0       3                       Dooley, Mr. Patrick   

        Sex   Age  SibSp  Parch      Ticket   Fare Cabin Embarked  
886    male  27.0      0      0      211536  13.00   NaN        S  
887  female  19.0      0      0      112053  30.00   B42        S  
888  female   NaN      1      2  W./C. 6607  23.45   NaN        S  
889    male  26.0      0      0      111369  30.00  C148        C  
890    male  32.0      0      0      370376   7.75   NaN        Q  
"""

# Which features contain blank, null or empty values?

# Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.

# Cabin > Age are incomplete in case of test dataset.


# variable types

train_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
"""

test_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
"""

# 7 integer/float features in the train dataframe

# 6 integer/float features in the test dataframe

# 5 features are strings

# Survived missing in the test dataframe, this is the feature we wish to predict


# How representative is the training dataset of the actual problem domain?

train_df.describe()
"""
       PassengerId    Survived      Pclass         Age       SibSp  \
count   891.000000  891.000000  891.000000  714.000000  891.000000   
mean    446.000000    0.383838    2.308642   29.699118    0.523008   
std     257.353842    0.486592    0.836071   14.526497    1.102743   
min       1.000000    0.000000    1.000000    0.420000    0.000000   
25%     223.500000    0.000000    2.000000   20.125000    0.000000   
50%     446.000000    0.000000    3.000000   28.000000    0.000000   
75%     668.500000    1.000000    3.000000   38.000000    1.000000   
max     891.000000    1.000000    3.000000   80.000000    8.000000   

            Parch        Fare  
count  891.000000  891.000000  
mean     0.381594   32.204208  
std      0.806057   49.693429  
min      0.000000    0.000000  
25%      0.000000    7.910400  
50%      0.000000   14.454200  
75%      0.000000   31.000000  
max      6.000000  512.329200  
"""

# distribution of categorical features:
    # quick sort in excel shows
    
# Names are unique
# sex variable, Male/Female, 65% male
# Cabin values have duplicates, perhaps several passengers shared a cabin

# embarked takes three possible values relating to the port of embarkation

# S(outhamption) is most

# Ticket feature has high ratio (22%) of duplicate values


                            
# correlating

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

"""
Out[192]: 
   Pclass  Survived
0       1  0.629630
1       2  0.472826
2       3  0.242363
"""

# Looks promising

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

"""
Out[194]: 
      Sex  Survived
0  female  0.742038
1    male  0.188908
"""

# again, looks promising

train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

"""
Out[195]: 
   SibSp  Survived
1      1  0.535885
2      2  0.464286
0      0  0.345395
3      3  0.250000
4      4  0.166667
5      5  0.000000
6      8  0.000000
"""

# one or two siblings look promising, no survivors if 5 or 8 siblings/spouses

train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

"""
   Parch  Survived
3      3  0.600000
1      1  0.550847
2      2  0.500000
0      0  0.343658
5      5  0.200000
4      4  0.000000
6      6  0.000000
"""

# consider the family tree
# Parch is vertical
# Sibsp is horizontal

# Consider Parch 6, this is likely to be someone travelling with:
    # 6 children - this equates to SibSp = 5 for each child need to find how many SibSp = 6

train_df["SibSp"].value_counts()
"""
Out[197]: 
0    608
1    209
2     28
4     18
3     16
8      7
5      5
Name: SibSp, dtype: int64
"""
# This shows that sSibSp = 5 are the children of one family, all of which died 
# SibSp = 8 has a return of 7. This means there was one family, 1 member had a spouse and 7 siblings, the seven siblings had no spouse. They all died.
# As there is no ParCh 7, these travelled with no parents. They could well have had children. 

train_df[train_df["SibSp"] == 8]
"""
     PassengerId  Survived  Pclass                               Name     Sex  \
159          160         0       3         Sage, Master. Thomas Henry    male   
180          181         0       3       Sage, Miss. Constance Gladys  female   
201          202         0       3                Sage, Mr. Frederick    male   
324          325         0       3           Sage, Mr. George John Jr    male   
792          793         0       3            Sage, Miss. Stella Anna  female   
846          847         0       3           Sage, Mr. Douglas Bullen    male   
863          864         0       3  Sage, Miss. Dorothy Edith "Dolly"  female   

     Age  SibSp  Parch    Ticket   Fare Cabin Embarked  
159  NaN      8      2  CA. 2343  69.55   NaN        S  
180  NaN      8      2  CA. 2343  69.55   NaN        S  
201  NaN      8      2  CA. 2343  69.55   NaN        S  
324  NaN      8      2  CA. 2343  69.55   NaN        S  
792  NaN      8      2  CA. 2343  69.55   NaN        S  
846  NaN      8      2  CA. 2343  69.55   NaN        S  
863  NaN      8      2  CA. 2343  69.55   NaN        S  
"""

# They all have the same surname, they all travelled with two parents, did they die too?

train_df[train_df["Name"].str.contains("Sage,")]
"""
     PassengerId  Survived  Pclass                               Name     Sex  \
159          160         0       3         Sage, Master. Thomas Henry    male   
180          181         0       3       Sage, Miss. Constance Gladys  female   
201          202         0       3                Sage, Mr. Frederick    male   
324          325         0       3           Sage, Mr. George John Jr    male   
792          793         0       3            Sage, Miss. Stella Anna  female   
846          847         0       3           Sage, Mr. Douglas Bullen    male   
863          864         0       3  Sage, Miss. Dorothy Edith "Dolly"  female   

     Age  SibSp  Parch    Ticket   Fare Cabin Embarked  
159  NaN      8      2  CA. 2343  69.55   NaN        S  
180  NaN      8      2  CA. 2343  69.55   NaN        S  
201  NaN      8      2  CA. 2343  69.55   NaN        S  
324  NaN      8      2  CA. 2343  69.55   NaN        S  
792  NaN      8      2  CA. 2343  69.55   NaN        S  
846  NaN      8      2  CA. 2343  69.55   NaN        S  
863  NaN      8      2  CA. 2343  69.55   NaN        S  
"""

# parents not included on this list


# parch 6:
    # 1 parent and 5 children
    # 2 parents and 4 children


# Parch 4, this is likely to be someone travelling with:
    # 4 Children 
    # 2 parents and 2 children
    
#==================================
# visualising data

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g.savefig("SurvivedAge.PNG", dpi=600)

# Age has distribution features that show it should be considered

# Need to fill NaN with values in age

grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
grid.savefig("SurvivedPClass1.PNG", dpi=600)
# shows pclass is a factor, pclass3 definitely shows
# lower survival rates

grid = sns.FacetGrid(train_df, col="Survived", row='Pclass', size=2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
grid.savefig("SurvivedPClass.PNG", dpi=600)

# survival in Pclass, sex on embarked

grid = sns.FacetGrid(train_df, row="Embarked", size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid.savefig("EmbarkationSurvival.PNG", dpi=600)

# both embarkation and pclass give different survival rates


# Fare and Embarkation and survival on Sex

grid = sns.FacetGrid(train_df, row="Embarked", col="Survived", size=2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
grid.savefig("FareEmbarkSurvival.PNG", dpi=600)


# Dropping features

# Based on our assumptions, drop cabin and ticket features

# perform operations on both train_df and test_df

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
"""
Before (891, 12) (418, 11) (891, 12) (418, 11)
"""

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

"""
After (891, 10) (418, 9) (891, 10) (418, 9)
"""

# Adding new features

# Title in name

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_df['Title'], train_df['Sex'])
"""
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
"""

# replacing titles

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

"""
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.347826
"""
# i.e. women and children first!

# converting the categorical titles to ordinals

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
"""
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch     Fare Embarked  Title  
0      0   7.2500        S      1  
1      0  71.2833        C      3  
2      0   7.9250        S      2  
3      0  53.1000        S      3  
4      0   8.0500        S      1  
"""

# we can drop the name and passenger id

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# converting categorical features
# gender: female = 1, male = 0

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
 
train_df.head()
"""
   Survived  Pclass  Sex   Age  SibSp  Parch     Fare Embarked  Title
0         0       3    0  22.0      1      0   7.2500        S      1
1         1       1    1  38.0      1      0  71.2833        C      3
2         1       3    1  26.0      0      0   7.9250        S      2
3         1       1    1  35.0      1      0  53.1000        S      3
4         0       3    0  35.0      0      0   8.0500        S      1
"""

# filling in the missing values

# More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
grid.savefig("AgeClassSex.PNG", dpi=600)

# prepare an empty array to contain guessed Age values based on Pclass/Gender combinations

guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
"""
   Survived  Pclass  Sex  Age  SibSp  Parch     Fare Embarked  Title
0         0       3    0   22      1      0   7.2500        S      1
1         1       1    1   38      1      0  71.2833        C      3
2         1       3    1   26      0      0   7.9250        S      2
3         1       1    1   35      1      0  53.1000        S      3
4         0       3    0   35      0      0   8.0500        S      1
"""

guess_ages
""" 
array([[ 42.,  28.,  24.],
       [ 41.,  24.,  22.]])
"""

# Creating age bands

train_df['AgeBand'] = pd.cut(train_df['Age'],5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

"""
       AgeBand  Survived
0  (-0.08, 16]  0.550000
1     (16, 32]  0.337374
2     (32, 48]  0.412037
3     (48, 64]  0.434783
4     (64, 80]  0.090909
"""

# replacing Age with ordinals

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_df.head()

"""
   Survived  Pclass  Sex  Age  SibSp  Parch     Fare Embarked  Title   AgeBand
0         0       3    0    1      1      0   7.2500        S      1  (16, 32]
1         1       1    1    2      1      0  71.2833        C      3  (32, 48]
2         1       3    1    1      0      0   7.9250        S      2  (16, 32]
3         1       1    1    2      1      0  53.1000        S      3  (32, 48]
4         0       3    0    2      0      0   8.0500        S      1  (32, 48]
"""


"""
    Survived  Pclass  Sex  Age  SibSp  Parch     Fare Embarked  Title
0         0       3    0    1      1      0   7.2500        S      1
1         1       1    1    2      1      0  71.2833        C      3
2         1       3    1    1      0      0   7.9250        S      2
3         1       1    1    2      1      0  53.1000        S      3
4         0       3    0    2      0      0   8.0500        S      1
"""

# creating a new feature, familysize which combines parch and sibsp

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    
"""
   FamilySize  Survived
3           4  0.724138
2           3  0.578431
1           2  0.552795
6           7  0.333333
0           1  0.303538
4           5  0.200000
5           6  0.136364
7           8  0.000000
8          11  0.000000
"""
# Removing the AgeBand feature

train_df = train_df.drop(['AgeBand'], axis=1)

train_df.head()
# we can use this data to create an 'IsAlone' feature

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

"""
   IsAlone  Survived
0        0  0.505650
1        1  0.303538
"""

# let us drop Parch and SibSp

train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

train_df.head()

"""
   Survived  Pclass  Sex  Age     Fare Embarked  Title   AgeBand  FamilySize  \
0         0       3    0    0   7.2500        S      1  (16, 32]           2   
1         1       1    1    0  71.2833        C      3  (32, 48]           2   
2         1       3    1    0   7.9250        S      2  (16, 32]           1   
3         1       1    1    0  53.1000        S      3  (32, 48]           2   
4         0       3    0    0   8.0500        S      1  (32, 48]           1   

   IsAlone  
0        0  
1        0  
2        1  
3        0  
4        1  
"""

# creating artificial feature combining Pclass and Age.

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
  
train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

"""
   Age*Class  Age  Pclass
0          3    1       3
1          2    2       1
2          3    1       3
3          2    2       1
4          6    2       3
5          3    1       3
6          3    3       1
7          0    0       3
8          3    1       3
9          0    0       2
"""

# Filling the two missing values for embarkation

freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
"""
Out[369]: 'S'
"""
# this seems reasonable, similar fares eminate from this port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

"""
  Embarked  Survived
0        C  0.553571
1        Q  0.389610
2        S  0.339009
"""

# converting categorical to numeric

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C':1, 'Q':2}).astype(int)

train_df.head()
"""
   Survived  Pclass  Sex  Age     Fare  Embarked  Title  FamilySize  IsAlone  \
0         0       3    0    1   7.2500         0      1           2        0   
1         1       1    1    2  71.2833         1      3           2        0   
2         1       3    1    1   7.9250         0      2           1        1   
3         1       1    1    2  53.1000         0      3           2        0   
4         0       3    0    2   8.0500         0      1           1        1   

   Age*Class  
0          3  
1          2  
2          3  
3          2  
4          6  
"""

# replacing the single missing fare value with the median

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# creating a fare band.

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

"""
        FareBand  Survived
0       [0, 7.91]  0.197309
1  (7.91, 14.454]  0.303571
2    (14.454, 31]  0.454955
3   (31, 512.329]  0.581081
"""
# convert the far feature to ordinal values based on the fare band

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

train_df.head(10)
"""
   Survived  Pclass  Sex  Age  Fare  Embarked  Title  FamilySize  IsAlone  \
0         0       3    0    1     0         0      1           2        0   
1         1       1    1    2     3         1      3           2        0   
2         1       3    1    1     1         0      2           1        1   
3         1       1    1    2     3         0      3           2        0   
4         0       3    0    2     1         0      1           1        1   
5         0       3    0    1     1         2      1           1        1   
6         0       1    0    3     3         0      1           1        1   
7         0       3    0    0     2         0      4           5        0   
8         1       3    1    1     1         0      3           3        0   
9         1       2    1    0     2         1      3           2        0   

   Age*Class  
0          3  
1          2  
2          3  
3          2  
4          6  
5          3  
6          3  
7          0  
8          3  
9          0  
"""

test_df.head()
"""
   PassengerId  Pclass  Sex  Age  Fare  Embarked  Title  FamilySize  IsAlone  \
0          892       3    0    2     0         2      1           1        1   
1          893       3    1    2     0         0      3           2        0   
2          894       2    0    3     1         2      1           1        1   
3          895       3    0    1     1         0      1           1        1   
4          896       3    1    1     1         0      3           3        0   

   Age*Class  
0          6  
1          6  
2          6  
3          3  
4          3
"""

# Modelling:
    
    # Identify relationship between 
    # output: survived
    # input: other variables
    # problem is a classification and regression problem
    
    # Training model with given dataset
    # - Supervised Machine Learning

# suitable models

"""
Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine
"""

# split the data

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

X_test = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
"""
Out[386]: ((891, 9), (891,), (418, 9))
"""

# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train)*100, 2)
acc_log
"""
Out[393]: 81.590000000000003
"""

coeff_df = pd.DataFrame(train_df.columns.delete(0))
"""
            0
0      Pclass
1         Sex
2         Age
3        Fare
4    Embarked
5       Title
6  FamilySize
7     IsAlone
8   Age*Class
"""

coeff_df.columns = ['Feature']
coeff_df
"""
      Feature
0      Pclass
1         Sex
2         Age
3        Fare
4    Embarked
5       Title
6  FamilySize
7     IsAlone
8   Age*Class
"""

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

"""
Out[400]: 
      Feature  Correlation
1         Sex     2.215671
5       Title     0.476947
3        Fare     0.255126
4    Embarked     0.211899
8   Age*Class    -0.157424
2         Age    -0.325558
7     IsAlone    -0.353892
6  FamilySize    -0.458992
0      Pclass    -0.697766
"""

# Sex: Female very much more likely to survive
# Having a rare title helps
# Pclass, the higher the class number the less likely of survivale
# family size, the greater the family size, the less likely you are to survive

#======
# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train)*100,2)

acc_svc
"""
Out[405]: 83.840000000000003
"""

# higher than the LogReg model

#=======

# k-Nearest Neighbours algorithm

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train)*100, 2)
acc_knn
"""
Out[410]: 84.180000000000007
"""

# highest of all

#=========

# Gaussain Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)
acc_gaussian
"""
Out[413]: 80.359999999999999
"""

#===========

## Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train)*100, 2)
acc_perceptron

"""
Out[414]: 81.140000000000001
"""

#===========

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train)*100, 2)
acc_linear_svc
"""
Out[415]: 81.480000000000004
"""

#============

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train)*100, 2)
acc_sgd
"""
Out[417]: 80.469999999999999
"""

#============

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100, 2)
acc_decision_tree
"""
Out[418]: 88.549999999999997
"""

# Excellent!

#============

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100, 2)
acc_random_forest

"""
Out[420]: 88.549999999999997
"""

# Same excellent value!

#==========

# Model evaluation

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

"""
                        Model  Score
3               Random Forest  88.55
8               Decision Tree  88.55
1                         KNN  84.18
0     Support Vector Machines  83.84
2         Logistic Regression  81.59
7                  Linear SVC  81.48
5                  Perceptron  81.14
6  Stochastic Gradient Decent  80.47
4                 Naive Bayes  80.36
"""

submission = pd.DataFrame({
        "PassengerId": test_df['PassengerId'],
        "Survived": Y_pred
        })
submission.head()
          
"""
   PassengerId  Survived
0          892         0
1          893         0
2          894         0
3          895         0
4          896         0
"""

submission.tail()
"""
     PassengerId  Survived
413         1305         0
414         1306         1
415         1307         0
416         1308         0
417         1309         1
"""


