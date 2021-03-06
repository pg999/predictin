
# Purpose

To apply for a master's degree is a very expensive and intensive work.Students will guess their capacities and they will decide whether to apply for a master's degree or not.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
df = pd.read_csv("data/Admission_Predict.csv", sep=",")
df.head()
```






<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
           'LOR ', 'CGPA', 'Research', 'Chance of Admit '],
          dtype='object')




```python
df=df.rename(columns = {'LOR ':'LOR','Chance of Admit ':'Chance of Admit'})
```
## Correlation Matrix

```python
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()
```


![png](img/Figure_1.png)

Correaltion suggest that "Chance of Admit" is most related to 'GRE Score','CGPA', 'TOEFL Score'
and least related to 'SOP','Research','LOR'

```python
x = ['Having Research','Not Having Research']
y = np.array([len(df[df.Research == 1]),len(df[df.Research == 0])])
plt.bar(x,y)
plt.title('Research Frequency')
plt.xlabel('Candidates')
plt.ylabel('Frequency')
plt.show()
```


![png](img/Figure_2.png)



```python
df.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>200.500000</td>
      <td>316.807500</td>
      <td>107.410000</td>
      <td>3.087500</td>
      <td>3.400000</td>
      <td>3.452500</td>
      <td>8.598925</td>
      <td>0.547500</td>
      <td>0.724350</td>
    </tr>
    <tr>
      <th>std</th>
      <td>115.614301</td>
      <td>11.473646</td>
      <td>6.069514</td>
      <td>1.143728</td>
      <td>1.006869</td>
      <td>0.898478</td>
      <td>0.596317</td>
      <td>0.498362</td>
      <td>0.142609</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>290.000000</td>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.800000</td>
      <td>0.000000</td>
      <td>0.340000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>100.750000</td>
      <td>308.000000</td>
      <td>103.000000</td>
      <td>2.000000</td>
      <td>2.500000</td>
      <td>3.000000</td>
      <td>8.170000</td>
      <td>0.000000</td>
      <td>0.640000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>200.500000</td>
      <td>317.000000</td>
      <td>107.000000</td>
      <td>3.000000</td>
      <td>3.500000</td>
      <td>3.500000</td>
      <td>8.610000</td>
      <td>1.000000</td>
      <td>0.730000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>300.250000</td>
      <td>325.000000</td>
      <td>112.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>9.062500</td>
      <td>1.000000</td>
      <td>0.830000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>400.000000</td>
      <td>340.000000</td>
      <td>120.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>9.920000</td>
      <td>1.000000</td>
      <td>0.970000</td>
    </tr>
  </tbody>
</table>
</div>



TOEFL Score Distribution


```python
plt.hist(df['TOEFL Score'],bins=4)
plt.xlabel("TOEFL Score")
plt.ylabel("No. of Candidates")
plt.title("TOEFL Score Distribution")
plt.show()
```


![png](img/Figure_3.png)


Looking into the correlation "University Ranking" and "CGPA" are strongly related.


```python
plt.scatter(df['University Rating'],df['CGPA'])
plt.title("University Ranking for CGPA")
plt.xlabel('University Ranking')
plt.ylabel("CGPA")
plt.show()
```


![png](img/Figure_4.png)



```python
s = df[df["Chance of Admit"] >= 0.75]['University Rating'].head()
```


```python
s.plot(kind="bar",figsize=(10,10))
plt.title("University Ratings of Candidates with an 75% acceptance chance")
plt.xlabel("University Rating")
plt.ylabel("No. Of Candidates")
plt.show()
```


![png](img/Figure_5.png)


# Regression Algorithm


```python
data = df
data.drop(['Serial No.'],axis =1, inplace =True)
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = df['Chance of Admit'].values
x = df.drop(['Chance of Admit'],axis=1)
x.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Scaling Features


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.15, random_state = 42)
# print(train_x)
# print(train_x.describe())
scale_x = MinMaxScaler(feature_range=(0,1))
train_x = scale_x.fit_transform(train_x)
test_x = scale_x.transform(test_x)
```




## Linear Regression


```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_x,train_y)
predict_lr = lr.predict(test_x)
print("Linear Regression")
print("1.Actual Value:"+ str(test_y[0])+", Predicted Value:"+str(predict_lr[0]))
print("2.Actual Value:"+ str(test_y[1])+", Predicted Value:"+str(predict_lr[1]))
print("3.Actual Value:"+ str(test_y[2])+", Predicted Value:"+str(predict_lr[2]))
print("4.Actual Value:"+ str(test_y[3])+", Predicted Value:"+str(predict_lr[3]))
print("5.Actual Value:"+ str(test_y[4])+", Predicted Value:"+str(predict_lr[4]))
```

    Linear Regression
    1.Actual Value:0.68, Predicted Value:0.6528245394121167
    2.Actual Value:0.68, Predicted Value:0.7261341431057075
    3.Actual Value:0.9, Predicted Value:0.9356591927994817
    4.Actual Value:0.79, Predicted Value:0.8232906077342225
    5.Actual Value:0.44, Predicted Value:0.5823340994089246



```python
from sklearn.metrics import r2_score
r2_score_linear_regression = r2_score(predict_lr,test_y)
print("r square score (linear regression):",r2_score_linear_regression)
```

    r square score (linear regression): 0.7139090848879419


## Random Forest


```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(train_x,train_y)
predict_rf = rf.predict(test_x)
print("Random Forest Regression")
print("1.Actual Value:"+ str(test_y[0])+", Predicted Value:"+str(predict_rf[0]))
print("2.Actual Value:"+ str(test_y[1])+", Predicted Value:"+str(predict_rf[1]))
print("3.Actual Value:"+ str(test_y[2])+", Predicted Value:"+str(predict_rf[2]))
print("4.Actual Value:"+ str(test_y[3])+", Predicted Value:"+str(predict_rf[3]))
print("5.Actual Value:"+ str(test_y[4])+", Predicted Value:"+str(predict_rf[4]))
```

    Random Forest Regression
    1.Actual Value:0.68, Predicted Value:0.6635999999999997
    2.Actual Value:0.68, Predicted Value:0.7260000000000001
    3.Actual Value:0.9, Predicted Value:0.9407000000000002
    4.Actual Value:0.79, Predicted Value:0.8182000000000006
    5.Actual Value:0.44, Predicted Value:0.5952000000000005






```python
r2_score_randomforest_regression = r2_score(predict_rf,test_y)
print("r square score (random forest regression):",r2_score_randomforest_regression)
```

    r square score (random forest regression): 0.7364368314006701


## Decision Tree


```python
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=5,random_state = 42)
dt.fit(train_x,train_y)
predict_dt = dt.predict(test_x)
print("Decision Tree Regression")
print("1.Actual Value:"+ str(test_y[0])+", Predicted Value:"+str(predict_dt[0]))
print("2.Actual Value:"+ str(test_y[1])+", Predicted Value:"+str(predict_dt[1]))
print("3.Actual Value:"+ str(test_y[2])+", Predicted Value:"+str(predict_dt[2]))
print("4.Actual Value:"+ str(test_y[3])+", Predicted Value:"+str(predict_dt[3]))
print("5.Actual Value:"+ str(test_y[4])+", Predicted Value:"+str(predict_dt[4]))
```

    Decision Tree Regression
    1.Actual Value:0.68, Predicted Value:0.5800000000000001
    2.Actual Value:0.68, Predicted Value:0.7142857142857142
    3.Actual Value:0.9, Predicted Value:0.945
    4.Actual Value:0.79, Predicted Value:0.8293939393939396
    5.Actual Value:0.44, Predicted Value:0.5643750000000001



```python
r2_score_decisiontree_regression = r2_score(predict_dt,test_y)
print("r square score (Decision Tree regression):",r2_score_decisiontree_regression)
```

    r square score (Decision Tree regression): 0.698465958263528



```python
x = ['Linear Regression','Random Forest Regression','Decision Tree Regression']
y = np.array([r2_score_linear_regression,r2_score_randomforest_regression,r2_score_decisiontree_regression])
plt.bar(x,y)
plt.title("Comparision between Linear, Random Forest and Decision Tree Regression")
plt.ylabel("r_square Score")
plt.show()
```


![png](img/Figure_6.png)



```python
x_scatter = np.arange(0,60)
# print(x_scatter.shape)
# print(predict_lr)
plt.scatter(x_scatter,predict_lr,color="red")
plt.scatter(x_scatter,predict_rf,color="blue")
plt.scatter(x_scatter,predict_dt,color="green")
plt.scatter(x_scatter,test_y,color="black")
plt.show()
```


![png](img/Figure_7.png)



```python
df[df['Chance of Admit']>=0.70].shape
```




    (247, 8)




```python
plt.hist(df['Chance of Admit'],bins =200)
plt.show()
```


![png](img/Figure_8.png)


candidate's Chance of Admit is greater than 80%, the candidate will receive the 1 label.
candidate's Chance of Admit is lesser than 80%, the candidate will receive the 0 label.


```python
train_y_01 = [1 if each > 0.8 else 0 for each in train_y]
test_y_01  = [1 if each > 0.8 else 0 for each in test_y]
train_y_01 = np.array(train_y_01)
test_y_01 = np.array(test_y_01)
```

## Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(train_x,train_y_01)
predict_lgr = lgr.predict(test_x)
score_lgr = lgr.score(test_x,test_y_01)
print("Score(Logistic Regression):",score_lgr)
```

    Score(Logistic Regression): 0.9333333333333333






```python
print("Logistic Regression")
print("1.Actual Value:"+ str(test_y_01[0])+", Predicted Value:"+str(predict_lgr[0]))
print("2.Actual Value:"+ str(test_y_01[1])+", Predicted Value:"+str(predict_lgr[1]))
print("3.Actual Value:"+ str(test_y_01[2])+", Predicted Value:"+str(predict_lgr[2]))
print("4.Actual Value:"+ str(test_y_01[3])+", Predicted Value:"+str(predict_lgr[3]))
print("5.Actual Value:"+ str(test_y_01[4])+", Predicted Value:"+str(predict_lgr[4]))
```

    Logistic Regression
    1.Actual Value:0, Predicted Value:0
    2.Actual Value:0, Predicted Value:0
    3.Actual Value:1, Predicted Value:1
    4.Actual Value:0, Predicted Value:1
    5.Actual Value:0, Predicted Value:0



```python
from sklearn.metrics import confusion_matrix
cm_lgr = confusion_matrix(test_y_01,predict_lgr)
print(cm_lgr)
```

    [[37  1]
     [ 3 19]]



```python
sns.heatmap(cm_lgr,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f")
plt.title("Confusion Matrix of Logistic Regression")
plt.xlabel("Predicted Values")
plt.ylabel("Acutal Values")
plt.show()
```


![png](img/Figure_9.png)



```python
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
print("Accuracy",accuracy_score(test_y_01,predict_lgr))
print("Precision",precision_score(test_y_01,predict_lgr))
print("Recall/Sesitivity",recall_score(test_y_01,predict_lgr))
print("F1 Score",f1_score(test_y_01,predict_lgr))
```

    Accuracy 0.9333333333333333
    Precision 0.95
    Recall/Sesitivity 0.8636363636363636
    F1 Score 0.9047619047619048


## Support Vector Machine


```python
from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(train_x,train_y_01)
predict_svm = svm.predict(test_x)
score_svm = svm.score(test_x,test_y_01)
print("Score(SVM):",score_svm)
```

    Score(SVM): 0.9333333333333333


 



```python
cm_svm = confusion_matrix(test_y_01,predict_svm)
print(cm_svm)
```

    [[37  1]
     [ 3 19]]



```python
print("Accuracy",accuracy_score(test_y_01,predict_svm))
print("Precision",precision_score(test_y_01,predict_svm))
print("Recall/Sesitivity",recall_score(test_y_01,predict_svm))
print("F1 Score",f1_score(test_y_01,predict_svm))
```

    Accuracy 0.9333333333333333
    Precision 0.95
    Recall/Sesitivity 0.8636363636363636
    F1 Score 0.9047619047619048


## K Nearest Neighbour


```python
from sklearn.neighbors import KNeighborsClassifier
scores = []
for each in range(1,60):
    knn_n = KNeighborsClassifier(n_neighbors=each)
    knn_n.fit(train_x,train_y_01)
    scores.append(knn_n.score(test_x,test_y_01))
plt.plot(range(1,60),scores)
plt.show()
```


![png](img/Figure_10.png)



```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x,train_y_01)
predict_knn = knn.predict(test_x)
score_knn = knn.score(test_x,test_y_01)
cm_knn = confusion_matrix(test_y_01,predict_knn)
sns.heatmap(cm_knn,annot=True,linewidths=0.5,linecolor="red",fmt = ".0f")
plt.title("Confusion Matrix of KNN")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()
print("Score (KNN) :",score_knn)
```


![png](img/Figure_11.png)


    Score (KNN) : 0.9333333333333333



```python
print("Accuracy",accuracy_score(test_y_01,predict_knn))
print("Precision",precision_score(test_y_01,predict_knn))
print("Recall/Sesitivity",recall_score(test_y_01,predict_knn))
print("F1 Score",f1_score(test_y_01,predict_knn))
```

    Accuracy 0.9333333333333333
    Precision 0.9090909090909091
    Recall/Sesitivity 0.9090909090909091
    F1 Score 0.9090909090909091



```python
y = np.array([score_svm,score_lgr,score_knn])
x = ["SVM","LogisticReg.","KNN"]
plt.bar(x,y)
plt.title("Comparison of Classification Algorithms")
plt.xlabel("Classfication")
plt.ylabel("Score")
plt.show()
```


![png](img/Figure_12.png)

