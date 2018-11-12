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
varname = ['SibSp','Parch','Fare']

df['train_indicator'].value_counts()


##categorize string columns 

str_colname = ['Pclass','Sex','Age','Embarked'] ##get column name of string columns 
df.index = list(range(0, df.shape[0])) ##reorder index, becuase index is not unique due to rbinding test and train
dummies = pd.DataFrame(index = df.index)
for col_idx in str_colname:
    
    
    dummy = pd.get_dummies(df[col_idx])    
    dummy.columns  = [col_idx + '_'+ x for x in dummy.columns.tolist()] ##set new column name
    ##remove last column to avoid perfect correlation
    if(len(dummy.columns)>1):
        dummy.drop(dummy.columns[[-1,]], axis=1, inplace=True)
    dummies = dummies.join(dummy)
    
    

##combine the converted categorical variables with the other numeric variables
#varname = varname + dummies.columns.tolist()
df_final = df[varname].join(dummies)


idx_train  = df['train_indicator'] == 1

#### 3. Fit Logistic Regression 
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(df_final.loc[idx_train], y_train)


##generate predictions on training data
train_pred = clf.predict(df_final.loc[idx_train])

##generate a simple prediction accuracy on training prediction 
score = clf.score(df_final.loc[idx_train], y_train) #NOTE: the more correct way is to use the out of sample test data here for accuracy
print(score)  ##our prediction accuracy on training data on the most simple logistic regression model is 0.81


####4 predictions 
##generate predictions on the test dataset using our model 

test_df = df_final.loc[-idx_train]
idx = np.where(np.asanyarray(np.isnan(test_df))) ##check for missing value in test data
print idx 
 ##there's only one record with missing value in the test datast, and that is fare; fill that in with the column average
test_df = test_df.fillna(test_df.mean())
np.where(np.asanyarray(np.isnan(test_df)))  ##check there's no more missings


test_pred = clf.predict(test_df)  #<---final predictions 





