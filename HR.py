#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


hr = pd.read_csv("hr.csv")


# In[3]:


hr.head()


# In[4]:


hr.tail()


# In[5]:


feats = ['department','salary']
hr_final = pd.get_dummies(hr,columns=feats,drop_first=True)
print(hr_final)


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


x= hr_final.drop(['left'], axis=1).values
y= hr_final['left'].values


# In[8]:


print(x)


# In[9]:


print(y)


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)


# In[11]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)


# In[12]:


print(x_train)


# In[13]:


print(x_test)


# In[14]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[15]:


classifier= Sequential()


# In[16]:


classifier.add(Dense(9, kernel_initializer = "uniform", activation= "relu", input_dim=18))


# In[17]:


classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))


# In[18]:


classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])


# In[21]:


import time
a = [10,20,30,40,50]
for i in a:
    start = time.time()
    classifier.fit(x_train, y_train, batch_size=i, epochs = 10)
    end= time.time()
    print(end-start)
    


# In[22]:


y_pred = classifier.predict(x_test)


# In[23]:


y_pred = (y_pred > 0.5)


# In[24]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[25]:


new_pred = classifier.predict(sc.transform(np.array([[0.26,0.7,3.,
                                                     238.,6.,0.,
                                                     0.,0.,0.,0.,0.,
                                                     0.,0.,0.,1.,0.,
                                                     0.,1.]])))


# In[26]:


new_pred = (new_pred > 0.5)
new_pred


# In[27]:


new_pred = (new_pred > 0.6)
new_pred


# In[28]:


from  keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[29]:


def make_classifier():
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation= "relu", input_dim=18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))
    classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])
    return classifier

    


# In[30]:


classifier = KerasClassifier(build_fn = make_classifier, batch_size=10, nb_epoch =1)


# In[31]:


accuracies = cross_val_score(estimator = classifier, X = x_train, y= y_train, cv =10 )


# In[56]:


mean = accuracies.mean()
mean


# In[58]:


variance = accuracies.var()
variance


# In[60]:


from keras.layers import Dropout
    
classifier = Sequential()
classifier.add(Dense(9, kernel_initializer = "uniform", activation= "relu", input_dim=18))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))
classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])


# In[84]:


from sklearn.model_selection import GridSearchCV
def make_clasifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation= "relu", input_dim=18))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))
    classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])
    return classifier


# In[85]:


classifier = KerasClassifier(build_fn = make_classifier)


# In[89]:


def build_classifier(optimizer = 'adam'):
  
  classifier.compile(optimizer=optimizer , loss = 'binary_crossentropy' , 
  metrics=['accuracy'])
  
  return classifier


# In[90]:


params = {
    'batch_size':[20,35],
    'epochs':[2,3],
    'optimizer':['adam','rmsprop']
}


# In[91]:


grid_search = GridSearchCV(estimator=classifier,
                          param_grid=params,
                          scoring="accuracy",
                          cv=2)


# In[93]:


get_ipython().system('pip install scikit-learn --user')


# In[ ]:


batch_size = [20, 35]
epochs = [2, 3]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)


# In[96]:


grid_result =grid_search.fit(x, y)


# Applying RFE (Recursive feature elimination) and then building logistics regression model to predict which variable might impact the employees attrition rate.
# 

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[29]:


pd.crosstab(hr.department, hr.left).plot(kind='bar')
plt.title('Employee attrition per Department')
plt.xlabel('Department')
plt.ylabel('Employee Attrition')
plt.savefig('Employee attrition per Deparment')


# In[30]:


pd.crosstab(hr.salary, hr.left).plot(kind='bar')
plt.title('Employee attrition as per Salary')
plt.xlabel('Salary')
plt.ylabel('Employee Attrition')
plt.savefig('Employee attrition as per Salary')


# In[32]:



pd.crosstab(hr.salary, hr.promotion_last_5years).plot(kind='bar')
plt.title('promotion_last_5years for Salary categories')
plt.xlabel('Salary')
plt.ylabel('promotion_last_5years')
plt.savefig('promotion_last_5years for Salary categories')


# In[35]:


hr_final.shape


# In[33]:


#applying RFE to filter best suitable variables for our model

from sklearn.feature_selection import RFE


# In[39]:


rfe = RFE(model, 5)
rfe = rfe.fit(x, y)
print(rfe.support_)
print(rfe.ranking_)


# We have 5 variables for our model, which are marked as true in the support_array and "1" in the ranking_array. They are: 'satisfaction_level', 'Work_accident', 'promotion_last_5years', 'salary_low', 'salary_medium'.
# 
# Now, we will build Logistic regression model by usig these stated 5 variables.

# In[41]:


vars=['satisfaction_level', 'Work_accident', 'promotion_last_5years', 'salary_low', 'salary_medium']
X = hr_final[vars]
Y = hr_final['left']


# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
model.fit(X_train, Y_train)


# In[64]:


from sklearn.metrics import accuracy_score
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(Y_test, model.predict(X_test))))


# In[66]:


from sklearn.metrics import classification_report
import seaborn as sns
print(classification_report(Y_test, model.predict(X_test)))


# In[71]:


pred_Y = model.predict(X_test)
model_cm = metrics.confusion_matrix(pred_Y, Y_test, [1,0])
model_cm


# In[73]:


sns.heatmap(model_cm, annot=True, fmt= '.2f', xticklabels = ["Left", "Stayed"], yticklabels =["Left", "Stayed"])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')
plt.savefig('HR_Logistic_regression_CM')


# In[ ]:




