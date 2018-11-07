# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 08:55:20 2018

@author: elainequ
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

train = pd.read_csv("/Users/elainequ/Documents/kaggle/titanic/all/train.csv")
test = pd.read_csv("/Users/elainequ/Documents/kaggle/titanic/all/test.csv")



##Model Diagnostics

sub_df = train[train['Age'] <=3 ]
plt.scatter(x = sub_df['Age'], y = sub_df['Survived'])


sub_df = train[(train['Age'] >=3)&(train['Age']<=12)]
plt.scatter(x = sub_df['Age'], y = sub_df['Survived'])


sub_df = train[(train['Age'] >12)&(train['Age']<=19)]
plt.scatter(x = sub_df['Age'], y = sub_df['Survived'])


sub_df = train[(train['Age'] >19)&(train['Age']<=30)]
plt.scatter(x = sub_df['Age'], y = sub_df['Survived'])


sub_df = train[(train['Age'] >30)&(train['Age']<=60)]
plt.scatter(x = sub_df['Age'], y = sub_df['Survived'])

sub_df = train[(train['Age'] >60)]
plt.scatter(x = sub_df['Age'], y = sub_df['Survived'])


##Transformation 

##combine train set with test set, set to df, df = cbind(train, test)
train['train_indicator'] = 1
test['train_indicator'] = 0
y_train = train['Survived']
train.drop('Survived', axis=1, inplace=True)

assert(train.shape[1] == test.shape[1]) ##assert their dimensions match
assert(list(train) == list(test)) ##check their column names/orders to be the same 

df = train.append(test)


age = pd.cut(df['Age'], bins = [0,3,12,20,30,60,100])
age = age.replace(np.nan, 'missing', regex=True)
df['Age'] = age
df['Pclass'] = df['Pclass'].astype(str)  ##convert the class from numeric to categorical 



##select columns 
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

df['train_indicator'].value_counts()


##categorize string columns 

str_colname = ['Pclass','Sex','Age','Embarked'] ##get column name of string columns 
dummies = pd.DataFrame(index = df.index)
for col_idx in str_colname:
    
    
    dummy = pd.get_dummies(df[col_idx])    
    dummy.columns  = [col_idx + '_'+ x for x in dummy.columns.tolist()] ##set new column name
    ##remove last column to avoid perfect correlation
    if(len(dummy.columns)>1):
        dummy.drop(dummy.columns[[-1,]], axis=1, inplace=True)
    dummies = dummies.join(dummy)
    
    

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


idx_train  = df.index[df['train_indicator']==1].tolist()

#### 3. Fit Logistic Regression 
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(df, y_train)



empty=pd.DataFrame(columns=['a'])
empty['b'] = None
df = df.assign(c=None)
df = df.assign(d=df['a'])
df['e'] = pd.Series(index=df.index)   
df = pd.concat([df,pd.DataFrame(columns=list('f'))])





