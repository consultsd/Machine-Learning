
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import pandas_profiling 

#To find version of pandas
pd.__version__

get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 25,10
import matplotlib.pyplot as plt

pd.set_option('display.precision',2)
pd.set_option('display.float_format','{:,}'.format)


# In[2]:

#Importing train and test datasets
data_path = "C:/Users/Sharath P Dandamudi/Desktop/"
train_file = data_path + "healthcare.xlsx"

train1 = pd.read_excel(train_file,header=0) 


# In[3]:

train = train1


# In[5]:

#Checking the dimensions of train dataset
print train.shape


# In[6]:

#Checking the data types in train dataset
#train.columns
train.columns


# In[7]:

cat_var =['']


# In[8]:

train[cat_var] =train[cat_var].astype(str)


# In[9]:

train.drop('QS5B_TBDER',axis=1,inplace=True)


# In[10]:

train.dtypes


# In[11]:

train['QS5B_DER'] = train['QS5B.1'].astype(str) + train['QS5B.2'].astype(str) + train['QS5B.3'].astype(str) + train['QS5B.4'].astype(str)+ train['QS5B.5'].astype(str) + train['QS5B.6'].astype(str) + train['QS5B.98'].astype(str)


# In[12]:

train.head()


# In[52]:

#EDA 
pfr = pandas_profiling.ProfileReport(train)
output_file = data_path + "healthcare_output.html"
pfr.to_file(output_file)


# In[13]:

def f(row):
    if row['QS5B_DER'] == '1000000':
        val = ''
    elif row['QS5B_DER'] == '0100000':
        val = ''
    elif row['QS5B_DER'] == '0010000':
        val = ' Native'
    elif row['QS5B_DER'] == '0001000':
        val = ''
    elif row['QS5B_DER'] == '0000100':
        val = ' Islander'
    elif row['QS5B_DER'] == '000010':
        val = ''       
    else:
        val = "Prefer_not_to_answer"
    return val


# In[14]:

train['QS5B_DER_VAR'] = train.apply(f, axis=1)


# In[15]:

train[''].head(10)


# In[16]:

train[''] =train[''].astype(str)


# In[17]:

train.dtypes


# In[18]:

train.rename(columns={'Behavioral_adjust': 'Behavioral_adjust_CAT'}, inplace=True)


# In[19]:

train.columns


# In[26]:

# train.to_csv(data_path + 'train_modified.csv',index=False)


# In[20]:

train['Profitability_2_DV'].value_counts()


# In[21]:

train['QB6_CAT'].value_counts()


# In[22]:

train['QB6_CAT'].replace('nan','MISSING',inplace=True)


# In[50]:

train['QB6_CAT'].value_counts()


# In[23]:

train['Cost_coverage_CAT'].value_counts()


# In[24]:

train['Cost_coverage_CAT'].replace('nan','MISSING',inplace=True)


# In[25]:

train['Cost_coverage_CAT'].value_counts()


# In[26]:

train['Behavioral_adjust_CAT'].value_counts()


# In[27]:

train['Behavioral_adjust_CAT'].replace('nan','MISSING',inplace=True)


# In[28]:

train['Behavioral_adjust_CAT'].value_counts()


# In[29]:

train['QS7_CAT'].value_counts()


# In[30]:

train['QS7_CAT'].replace('nan','MISSING',inplace=True)


# In[31]:

train['QS7_CAT'].value_counts()


# In[32]:

train['Tech_insurer_all'].fillna(train['Tech_insurer_all'].median(),inplace=True)


# In[33]:

train['Tech_insurer_all'].value_counts()


# In[36]:

train['QC17_6_ORD'].fillna(train['QC17_6_ORD'].median(),inplace=True)


# In[38]:

train['QC17_6_ORD'].value_counts(dropna=False)


# In[39]:

train.dtypes


# In[97]:

train_health3 = train.loc[train['Health_status'] == 3]


# In[98]:

train_health3.shape


# In[99]:

train_health3['Profitability_2_DV'].value_counts(dropna=False)


# In[100]:

train_health3.dropna(subset = ['Profitability_2_DV'],axis=0, inplace=True)


# In[101]:

train_health3.shape


# In[102]:

train_ansys2 = train_health3.loc[train_health3['Profitability_2_DV'].str.strip().isin(['Healthy Unprofitable','Healthy Profitable'])]


# In[103]:

train_ansys2.shape


# In[47]:

train_ansys2.dtypes


# In[104]:

train_ansys2.to_csv(data_path + 'train_ansys2.csv',index=False)


# In[110]:




# In[52]:

categorical_var=[]


# In[107]:

train_ansys2 = pd.get_dummies(train_ansys2, columns=categorical_var)


# In[108]:

train_ansys2.head(10)


# In[109]:

train_ansys2.to_csv(data_path + 'train_ansys2_dummy.csv',index=False)


# In[106]:

train_ansys2.columns


# In[58]:

iv_cat=[
]


# In[67]:

iv_cont=[]


# In[73]:

iv_tot = iv_cat + iv_cont


# In[74]:

iv_tot


# In[81]:

def my_plot_importance(booster, figsize, **kwargs): 
    from matplotlib import pyplot as plt
    from xgboost import plot_importance
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax, **kwargs)


# In[94]:

train_ansys2['Profitability_2_DV'].value_counts()


# In[ ]:




# In[87]:

#Plotting feature importance using built-in function - XGBoost
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

X = train_ansys2[iv_tot]
Y = train_ansys2['Profitability_2_DV']

#Fitting model on training data
model = XGBClassifier()
model.fit(X, Y)

#Plotting feature importance
plot_importance(model)
pyplot.show()


# In[93]:

# use feature importance for feature selection
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# split data into X and y
X = train_ansys2[iv_tot]
Y = train_ansys2['Profitability_2_DV']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# In[ ]:



