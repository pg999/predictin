#!/usr/bin/env python
# coding: utf-8

# # Purpose

# To apply for a master's degree is a very expensive and intensive work.Students will guess their capacities and they will decide whether to apply for a master's degree or not.

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
df = pd.read_csv("data/Admission_Predict.csv", sep=",")
df.head()


# In[4]:


df.columns


# In[5]:


df=df.rename(columns = {'LOR ':'LOR','Chance of Admit ':'Chance of Admit'})

## Correlation Matrix
# In[6]:


fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()

#Correaltion suggest that "Chance of Admit" is most related to 'GRE Score','CGPA', 'TOEFL Score'
#and least related to 'SOP','Research','LOR'
# In[7]:


x = ['Having Research','Not Having Research']
y = np.array([len(df[df.Research == 1]),len(df[df.Research == 0])])
plt.bar(x,y)
plt.title('Research Frequency')
plt.xlabel('Candidates')
plt.ylabel('Frequency')
plt.show()


# In[8]:


df.describe()


# TOEFL Score Distribution

# In[9]:


plt.hist(df['TOEFL Score'],bins=4)
plt.xlabel("TOEFL Score")
plt.ylabel("No. of Candidates")
plt.title("TOEFL Score Distribution")
plt.show()


# Looking into the correlation "University Ranking" and "CGPA" are strongly related.

# In[10]:


plt.scatter(df['University Rating'],df['CGPA'])
plt.title("University Ranking for CGPA")
plt.xlabel('University Ranking')
plt.ylabel("CGPA")
plt.show()


# In[11]:


s = df[df["Chance of Admit"] >= 0.75]['University Rating'].head()


# In[12]:


s.plot(kind="bar",figsize=(10,10))
plt.title("University Ratings of Candidates with an 75% acceptance chance")
plt.xlabel("University Rating")
plt.ylabel("No. Of Candidates")
plt.show()


# # Regression Algorithm

# In[13]:


data = df
data.drop(['Serial No.'],axis =1, inplace =True)
data.head()


# In[14]:


y = df['Chance of Admit'].values
x = df.drop(['Chance of Admit'],axis=1)
x.head()


# ## Scaling Features

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.15, random_state = 42)
# print(train_x)
# print(train_x.describe())
scale_x = MinMaxScaler(feature_range=(0,1))
train_x = scale_x.fit_transform(train_x)
test_x = scale_x.transform(test_x)


# ## Linear Regression

# In[16]:


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


# In[17]:


from sklearn.metrics import r2_score
r2_score_linear_regression = r2_score(predict_lr,test_y)
print("r square score (linear regression):",r2_score_linear_regression)


# ## Random Forest

# In[18]:


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


# In[19]:


r2_score_randomforest_regression = r2_score(predict_rf,test_y)
print("r square score (random forest regression):",r2_score_randomforest_regression)


# ## Decision Tree

# In[20]:


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


# In[21]:


r2_score_decisiontree_regression = r2_score(predict_dt,test_y)
print("r square score (Decision Tree regression):",r2_score_decisiontree_regression)


# In[22]:


x = ['Linear Regression','Random Forest Regression','Decision Tree Regression']
y = np.array([r2_score_linear_regression,r2_score_randomforest_regression,r2_score_decisiontree_regression])
plt.bar(x,y)
plt.title("Comparision between Linear, Random Forest and Decision Tree Regression")
plt.ylabel("r_square Score")
plt.show()


# In[23]:


x_scatter = np.arange(0,60)
# print(x_scatter.shape)
# print(predict_lr)
plt.scatter(x_scatter,predict_lr,color="red")
plt.scatter(x_scatter,predict_rf,color="blue")
plt.scatter(x_scatter,predict_dt,color="green")
plt.scatter(x_scatter,test_y,color="black")
plt.show()


# In[24]:


df[df['Chance of Admit']>=0.70].shape


# In[25]:


plt.hist(df['Chance of Admit'],bins =200)
plt.show()


# candidate's Chance of Admit is greater than 80%, the candidate will receive the 1 label.
# candidate's Chance of Admit is lesser than 80%, the candidate will receive the 0 label.

# In[26]:


train_y_01 = [1 if each > 0.8 else 0 for each in train_y]
test_y_01  = [1 if each > 0.8 else 0 for each in test_y]
train_y_01 = np.array(train_y_01)
test_y_01 = np.array(test_y_01)


# ## Logistic Regression

# In[56]:


from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(train_x,train_y_01)
predict_lgr = lgr.predict(test_x)
score_lgr = lgr.score(test_x,test_y_01)
print("Score(Logistic Regression):",score_lgr)


# In[28]:


print("Logistic Regression")
print("1.Actual Value:"+ str(test_y_01[0])+", Predicted Value:"+str(predict_lgr[0]))
print("2.Actual Value:"+ str(test_y_01[1])+", Predicted Value:"+str(predict_lgr[1]))
print("3.Actual Value:"+ str(test_y_01[2])+", Predicted Value:"+str(predict_lgr[2]))
print("4.Actual Value:"+ str(test_y_01[3])+", Predicted Value:"+str(predict_lgr[3]))
print("5.Actual Value:"+ str(test_y_01[4])+", Predicted Value:"+str(predict_lgr[4]))


# In[29]:


from sklearn.metrics import confusion_matrix
cm_lgr = confusion_matrix(test_y_01,predict_lgr)
print(cm_lgr)


# In[45]:


sns.heatmap(cm_lgr,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f")
plt.title("Confusion Matrix of Logistic Regression")
plt.xlabel("Predicted Values")
plt.ylabel("Acutal Values")
plt.show()


# In[31]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
print("Accuracy",accuracy_score(test_y_01,predict_lgr))
print("Precision",precision_score(test_y_01,predict_lgr))
print("Recall/Sesitivity",recall_score(test_y_01,predict_lgr))
print("F1 Score",f1_score(test_y_01,predict_lgr))


# ## Support Vector Machine

# In[55]:


from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(train_x,train_y_01)
predict_svm = svm.predict(test_x)
score_svm = svm.score(test_x,test_y_01)
print("Score(SVM):",score_svm)


# In[37]:


cm_svm = confusion_matrix(test_y_01,predict_svm)
print(cm_svm)


# In[39]:


print("Accuracy",accuracy_score(test_y_01,predict_svm))
print("Precision",precision_score(test_y_01,predict_svm))
print("Recall/Sesitivity",recall_score(test_y_01,predict_svm))
print("F1 Score",f1_score(test_y_01,predict_svm))


# ## K Nearest Neighbour

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
scores = []
for each in range(1,60):
    knn_n = KNeighborsClassifier(n_neighbors=each)
    knn_n.fit(train_x,train_y_01)
    scores.append(knn_n.score(test_x,test_y_01))
plt.plot(range(1,60),scores)
plt.show()


# In[58]:


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


# In[52]:


print("Accuracy",accuracy_score(test_y_01,predict_knn))
print("Precision",precision_score(test_y_01,predict_knn))
print("Recall/Sesitivity",recall_score(test_y_01,predict_knn))
print("F1 Score",f1_score(test_y_01,predict_knn))


# In[59]:


y = np.array([score_svm,score_lgr,score_knn])
x = ["SVM","LogisticReg.","KNN"]
plt.bar(x,y)
plt.title("Comparison of Classification Algorithms")
plt.xlabel("Classfication")
plt.ylabel("Score")
plt.show()

