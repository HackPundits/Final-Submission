import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
%matplotlib inline

dataset= pd.read_csv("C:\\Users\\Anwar\\Downloads\\enamebaseddata.csv")
dataset1 = dataset[['users','ename']]
dataset2 = dataset[['ecount','ename']]

plt.scatter(df_new['ename'],df_new['users'])
plt.scatter(df_new['ename'],df_new['ecount'])

clf= LinearRegression()
y = df_new['ename']
x = df_new[['','ecount']]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
acc = accuracy_score(pred, y_test)

print(acc)