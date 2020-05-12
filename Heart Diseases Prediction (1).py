#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import cufflinks as cf
from sklearn.metrics import accuracy_score


# In[2]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[3]:


df=pd.read_csv('C:/Users/user/Desktop/heart.csv')


# In[4]:


df


# In[5]:


info=["age","1:male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain,4: asymptomatic","resting blood pressure"," serum cholestrol in mg/dl","fasting blood sugar > 120 mg/dl"," resting electrocardiographic results (values 0,1,2)","maximum heart rate achieved","exercise induce angina","oldpeake=ST depressioninduced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessel (0-3)colored by flourosopy","thal:3= normal;6=fixed defect;7=reversable defect"]
for i in range(len(info)):
    print(df.columns[i]+"\t\t\t"+info[i])


# In[6]:


df['target']


# In[7]:


df.groupby('target').size()


# In[8]:


df.size


# In[9]:


df.describe()


# In[10]:


# to check the null value
df.info()


# In[11]:


#Visualisation


# In[12]:


df.hist(figsize=(14,14))
plt.show()


# In[13]:


sns.barplot(df['sex'],df['target'])
plt.show()


# In[14]:


sns.barplot(df['sex'],df['age'],hue=df['target'])
plt.show()


# In[15]:


px.bar(df,df['sex'],df['target'])


# In[ ]:





# In[16]:


numeric_column=['trestbps','chol','age','oldpeak','thalach']


# In[17]:


sns.heatmap(df[numeric_column].corr(),annot=True,cmap='terrain',linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[ ]:





# In[18]:


plt.figure(figsize=(12,10))
plt.subplot(221)
sns.distplot(df[df['target']==0].age)
plt.title('Age of patients without Heart Diseases')

plt.subplot(222)
sns.distplot(df[df['target']==1].age)
plt.title('Age of patients with Heart Diseases')

plt.subplot(223)
sns.distplot(df[df['target']==0].thalach)
plt.title('Maximum  heart rate of patient without Heart Diseases')

plt.subplot(224)
sns.distplot(df[df['target']==1].thalach)
plt.title('Maximum  heart rate of patient with Heart Diseases')


# In[ ]:





# In[20]:


#Data Preprocessing


# In[21]:


X,y=df.loc[:,:'thal'],df['target']
X


# In[22]:


y


# In[23]:


X.size


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.3,shuffle=True)


# In[26]:


X_train


# In[27]:


X_test


# In[28]:


from sklearn.preprocessing import  StandardScaler


std=StandardScaler().fit(X)


# In[29]:


# Decsion Tree Classifier


# In[30]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[31]:


prediction=dt.predict(X_test)


# In[32]:


prediction


# In[33]:


dt.score(X_test,y_test)


# In[34]:


accuracy_dt=accuracy_score(y_test,prediction)


# In[35]:


accuracy_dt


# In[36]:


# to find the most important feature
dt.feature_importances_


# In[37]:


def plot_feature_importance(model):
    plt.figure(figsize=(8,6))
    n_features=13
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),X)
    plt.xlabel("Important Features")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)
    
    
plot_feature_importance(dt)


# In[ ]:





# In[38]:


df


# In[39]:


Category=['No,you dont have heart disease','Yes,You have heart disease']


# In[40]:


custom_data=np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])


# In[41]:


custom_data_prediction_dt=dt.predict(custom_data)


# In[42]:


custom_data_prediction_dt


# In[43]:


print(Category[int(custom_data_prediction_dt)])


# In[44]:


custom_data1=np.array([[57,0,1,130,236,0,0,174,0,0.0,1,1,2]])


# In[45]:


custom_data_prediction_dt1=dt.predict(custom_data1)


# In[46]:


custom_data_prediction_dt1


# In[47]:


print(Category[int(custom_data_prediction_dt1)])


# In[ ]:





# In[ ]:


# RandomForestClassifier


# In[72]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=4)


# In[74]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=1000)
clf.fit(X_train,y_train)


# In[78]:


pre=clf.predict(X_test)


# In[79]:


pre


# In[80]:


accuracy_clf=accuracy_score(y_test,pre)


# In[82]:


accuracy_clf


# In[77]:


clf.score(X_test,y_test)


# In[48]:


#KNN Algorithm


# In[49]:


from sklearn.preprocessing import  StandardScaler


std=StandardScaler().fit(X)


# In[50]:


std=StandardScaler().fit(X)
X_std=std.transform(X)


# In[51]:


X


# In[52]:


X_std


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


X_train_std,X_test_std,y_train,y_test=train_test_split(X_std,y,random_state=10,test_size=0.3,shuffle=True)


# In[ ]:





# In[ ]:





# In[55]:


from sklearn.neighbors import KNeighborsClassifier


knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_std,y_train)


# In[56]:


knn_prediction=knn.predict(X_test_std)
knn_prediction


# In[57]:


accuracy_dt_knn=accuracy_score(y_test,knn_prediction)


# In[58]:


accuracy_dt_knn


# In[59]:


Category=['No,you dont have heart disease','Yes,You have heart disease']


# In[60]:


custom_data_knn=np.array([[57,0,1,130,236,0,0,174,0,0.0,1,1,2]])


# In[61]:


custom_data_knn_std=std.transform(custom_data_knn)


# In[62]:


custom_data_knn_std


# In[63]:


custom_data_prediction_knn=knn.predict(custom_data_knn_std)


# In[64]:


custom_data_prediction_knn


# In[65]:


print(Category[int(custom_data_prediction_knn)])


# In[ ]:





# In[66]:


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    knn_prediction=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,knn_prediction)
    scores_list.append(accuracy_score(y_test,knn_prediction))


# In[67]:


scores


# In[68]:


plt.plot(k_range,scores_list)


# In[ ]:





# In[69]:


px.line(x=k_range,y=scores_list)


# In[ ]:





# In[83]:


algorithm=['Decision Tree','KNN Algorithm','RandomForestClassifier']
scores=[accuracy_dt,accuracy_dt_knn,accuracy_clf]


# In[84]:


sns.set(rc={'figure.figsize':(15,7)})
plt.xlabel('Algorithm')
plt.ylabel('Accuracy Score')

sns.barplot(algorithm,scores)


# In[ ]:





# In[ ]:




