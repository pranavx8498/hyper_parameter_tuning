#!/usr/bin/env python
# coding: utf-8

# # hyperparameter tuning

# In[12]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=400, max_features='sqrt', min_samples_leaf=2,
                       min_samples_split=6, n_estimators=500)


# In[13]:


import numpy as np
df.Glucose=np.where(df['Glucose']==0,df.Glucose.mean(),df['Glucose'])


# In[14]:


import pandas as pd
df=pd.read_csv('diabetes.csv')


# In[15]:


import warnings
warnings.filterwarnings('ignore')
df.head()


# In[16]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1:]


# In[17]:


df.SkinThickness=np.where(df.SkinThickness==0,df.SkinThickness.mean(),df.SkinThickness)
df.Insulin=np.where(df.Insulin==0,df.Insulin.mean(),df.Insulin)


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=10)


# In[24]:


rf.fit(x_train,y_train)


# In[25]:


rf.score(x_test,y_test)


# In[27]:


predict=rf.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(y_test,predict))
print(accuracy_score(y_test,predict))
print(classification_report(y_test,predict))


# # random search cv
# randomly choose the value

# In[9]:


n_estimators=[int (i) for i in np.linspace(200,2000,num=10)]
max_depth=[int (i) for i in np.linspace(100,1000,10)]
max_features=['auto', 'sqrt','log2','None']
min_samples_split=[1,2,6,7,9]
min_samples_leaf=[1, 2, 4,6,8]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}


# In[29]:


rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)


# In[30]:


rf_randomcv.fit(x_train,y_train)


# In[31]:


rf_randomcv.score(x_test,y_test)


# In[32]:


rf_randomcv.best_params_


# In[33]:


rf_randomcv.best_estimator_


# In[34]:


param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 
                         rf_randomcv.best_params_['min_samples_leaf']+2, 
                         rf_randomcv.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
                          rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'], 
                          rf_randomcv.best_params_['min_samples_split'] +1,
                          rf_randomcv.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200, rf_randomcv.best_params_['n_estimators'] - 100, 
                     rf_randomcv.best_params_['n_estimators'], 
                     rf_randomcv.best_params_['n_estimators'] + 100, rf_randomcv.best_params_['n_estimators'] + 200]
}


# In[19]:


from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10,n_jobs=-1,verbose=2)


# In[20]:


gs.fit(x_train,y_train)


# In[21]:


gs.score(x_test,y_test)


# In[22]:


gs.best_estimator_


# In[40]:


from sklearn.model_selection import cross_val_score


# # automated search cv
# hp.choice=string values
# hp.uniform=intger parameter
# three steps to this
# 1.loss function
# 2.space the parameter we want to give
# 3.define improvment algorithum

# In[41]:


from hyperopt import hp,fmin,STATUS_OK,Trials,tpe


# In[42]:



space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 10, 1200, 10),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
    }


# In[43]:


def objective(space):
    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    accuracy = cross_val_score(model, x_train, y_train, cv = 5).mean()
    return {'loss': -accuracy, 'status': STATUS_OK }


# In[44]:


trials=Trials()


# In[45]:


best=fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=80,trials=trials)


# In[46]:


best


# In[47]:


crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}
trainedforest=RandomForestClassifier(criterion = crit[best['criterion']], max_depth = best['max_depth'], 
                                       max_features = feat[best['max_features']], 
                                       min_samples_leaf = best['min_samples_leaf'], 
                                       min_samples_split = best['min_samples_split'], 
                                       n_estimators = est[best['n_estimators']]).fit(x_train,y_train)


# In[48]:


predictionforest = trainedforest.predict(x_test)
print(confusion_matrix(y_test,predictionforest))
print(accuracy_score(y_test,predictionforest))
print(classification_report(y_test,predictionforest))
acc5 = accuracy_score(y_test,predictionforest)


# # Genetic Algorithum
# result what we got select the best have of it improve them also sprout them and imporove them iterate the the process upto you find the best one

# In[1]:


from tpot import TPOTClassifier


# In[21]:


tt=TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,
                                 config_dict={'sklearn.ensemble.RandomForestClassifier': random_grid}, 
                                 cv = 4, scoring = 'accuracy')


# In[22]:


tt.fit(x_train,y_train)


# In[24]:


trf=RandomForestClassifier(criterion='gini', max_depth=700, max_features='log2', min_samples_leaf=6, min_samples_split=2, n_estimators=1600)


# In[25]:


trf.fit(x_train,y_train)


# In[27]:


accc=tt.score(x_test,y_test)


# In[28]:


accc


# # optuna

# In[39]:


import optuna
from sklearn.model_selection import cross_val_score


# In[40]:


def objective(trials):
    criterion=trials.suggest_categorical('criterion',['gini','entropy'])
    max_depth=trials.suggest_int('max_depth',10,1200,log=True)
    max_features=trials.suggest_int('max_features',2,5,6)
    n_estimators=trials.suggest_int('n_estimators',100,500)
    rf=RandomForestClassifier(n_estimators=n_estimators,
    criterion=criterion,
    max_depth=max_depth,max_features=max_features)
    score = cross_val_score(rf, x_train,y_train, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


# In[41]:


opt = optuna.create_study(direction="maximize")
opt.optimize(objective, n_trials=15)


# In[42]:


best_opt=opt.best_params


# In[43]:


best_opt


# In[45]:


opt_rf=RandomForestClassifier(criterion= 'gini', max_depth= 181, max_features= 2, n_estimators= 238)


# In[46]:


opt_rf.fit(x_train,y_train)


# In[47]:


opt_rf.score(x_test,y_test)


# In[ ]:




