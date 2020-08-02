import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline
train=pd.read_csv('C:\\Users\\Abinaya\\Desktop\\mnist_train.csv')
train.head()
a=train.iloc[3,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)
df_x=train.iloc[:,1:]#expect labels
df_y=train.iloc[:,0]#only labels
df_x.head()
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2, random_state=0)
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
pred
b=y_test.values#check prediction accuracy
count=0
for i in range (len(pred)):# calculating no of correctly predicted values
    if pred[i]==b[i]:
        count=count+1
count
len(pred)
11645/12000#accuracy
