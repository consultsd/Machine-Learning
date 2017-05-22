
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

cat_var =['QGEN_CAT',
'QS6_CAT',
'QS7_CAT',
'QS9_CAT',
'QS10_CAT',
'QS12_CAT',
'QF5_CAT',
'QS17_CAT',
'QA7_CAT',
'QB3_CAT',
'QB4R1_CAT',
'QB4R2_CAT',
'QB4R3_CAT',
'QB4R4_CAT',
'QB5_CAT',
'QB7_CAT',
'QH4R18_CAT',
'QD2R13_99_CAT',
'Info_gathering_CAT',
'Care_interaction_CAT',
'Healthcare_process_CAT',
'Admin_tasks_CAT',
'Cost_coverage_CAT',
'QF4R1_CAT',
'QF4R6_CAT',
'QF4R9_CAT',
'QF4R12_CAT',
'QS5B_TBDER',
'QB6_CAT']


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
        val = 'White'
    elif row['QS5B_DER'] == '0100000':
        val = 'African_American'
    elif row['QS5B_DER'] == '0010000':
        val = 'American_Indian_or_Alaska Native'
    elif row['QS5B_DER'] == '0001000':
        val = 'Asian'
    elif row['QS5B_DER'] == '0000100':
        val = 'Native_Hawaiian_or_other_Pacific Islander'
    elif row['QS5B_DER'] == '000010':
        val = 'Some_other_race'       
    else:
        val = "Prefer_not_to_answer"
    return val


# In[14]:

train['QS5B_DER_VAR'] = train.apply(f, axis=1)


# In[15]:

train['Behavioral_adjust'].head(10)


# In[16]:

train['Behavioral_adjust'] =train['Behavioral_adjust'].astype(str)


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

categorical_var=['Admin_tasks_CAT',
'Behavioral_adjust_CAT',
'Care_interaction_CAT',
'Chronic',
'Cost_coverage_CAT',
'Healthcare_process_CAT',
'Info_gathering_CAT',
'QA7_CAT',
'QB3_CAT',
'QB4R1_CAT',
'QB4R2_CAT',
'QB4R3_CAT',
'QB4R4_CAT',
'QB5_CAT',
'QB6_CAT',
'QB7_CAT',
'QD2R13_99_CAT',
'QF4R1_CAT',
'QF4R12_CAT',
'QF4R6_CAT',
'QF4R9_CAT',
'QF5_CAT',
'QGEN_CAT',
'QH4R18_CAT',
'QS10_CAT',
'QS12_CAT',
'QS17_CAT',
'QS5B_DER_VAR',
'QS6_CAT',
'QS7_CAT',
'QS9_CAT']


# In[107]:

train_ansys2 = pd.get_dummies(train_ansys2, columns=categorical_var)


# In[108]:

train_ansys2.head(10)


# In[109]:

train_ansys2.to_csv(data_path + 'train_ansys2_dummy.csv',index=False)


# In[106]:

train_ansys2.columns


# In[58]:

iv_cat=['Admin_tasks_CAT_0',
'Admin_tasks_CAT_1',
'Behavioral_adjust_CAT_0.0',
'Behavioral_adjust_CAT_1.0',
'Care_interaction_CAT_0',
'Care_interaction_CAT_1',
'Chronic_Chronic_treated',
'Chronic_Chronic_untreated',
'Chronic_Not_chronic',
'Cost_coverage_CAT_0.0',
'Cost_coverage_CAT_1.0',
'Healthcare_process_CAT_0',
'Healthcare_process_CAT_1',
'Info_gathering_CAT_0',
'Info_gathering_CAT_1',
'QA7_CAT_1',
'QA7_CAT_2',
'QB3_CAT_1',
'QB3_CAT_2',
'QB4R1_CAT_1',
'QB4R1_CAT_2',
'QB4R1_CAT_3',
'QB4R1_CAT_4',
'QB4R1_CAT_98',
'QB4R2_CAT_1',
'QB4R2_CAT_2',
'QB4R2_CAT_3',
'QB4R2_CAT_4',
'QB4R2_CAT_98',
'QB4R3_CAT_1',
'QB4R3_CAT_2',
'QB4R3_CAT_3',
'QB4R3_CAT_4',
'QB4R3_CAT_98',
'QB4R4_CAT_1',
'QB4R4_CAT_2',
'QB4R4_CAT_3',
'QB4R4_CAT_4',
'QB4R4_CAT_98',
'QB5_CAT_1',
'QB5_CAT_2',
'QB5_CAT_98',
'QB6_CAT_1.0',
'QB6_CAT_2.0',
'QB6_CAT_98.0',
'QB6_CAT_MISSING',
'QB7_CAT_1',
'QB7_CAT_2',
'QB7_CAT_3',
'QB7_CAT_4',
'QB7_CAT_5',
'QB7_CAT_97',
'QB7_CAT_99',
'QD2R13_99_CAT_0',
'QD2R13_99_CAT_1',
'QF4R1_CAT_1',
'QF4R1_CAT_2',
'QF4R1_CAT_3',
'QF4R1_CAT_4',
'QF4R1_CAT_5',
'QF4R12_CAT_1',
'QF4R12_CAT_2',
'QF4R12_CAT_3',
'QF4R12_CAT_4',
'QF4R12_CAT_5',
'QF4R6_CAT_1',
'QF4R6_CAT_2',
'QF4R6_CAT_3',
'QF4R6_CAT_4',
'QF4R6_CAT_5',
'QF4R9_CAT_1',
'QF4R9_CAT_2',
'QF4R9_CAT_3',
'QF4R9_CAT_4',
'QF4R9_CAT_5',
'QF5_CAT_1',
'QF5_CAT_2',
'QF5_CAT_3',
'QF5_CAT_4',
'QF5_CAT_5',
'QF5_CAT_98',
'QF5_CAT_99',
'QGEN_CAT_1',
'QGEN_CAT_2',
'QH4R18_CAT_1',
'QH4R18_CAT_2',
'QH4R18_CAT_3',
'QH4R18_CAT_5',
'QS10_CAT_1',
'QS10_CAT_2',
'QS10_CAT_3',
'QS10_CAT_4',
'QS10_CAT_5',
'QS10_CAT_6',
'QS10_CAT_7',
'QS10_CAT_97',
'QS12_CAT_1',
'QS12_CAT_2',
'QS12_CAT_3',
'QS12_CAT_4',
'QS12_CAT_5',
'QS17_CAT_1',
'QS17_CAT_2',
'QS5B_DER_VAR_African_American',
'QS5B_DER_VAR_American_Indian_or_Alaska Native',
'QS5B_DER_VAR_Asian',
'QS5B_DER_VAR_Prefer_not_to_answer',
'QS5B_DER_VAR_White',
'QS6_CAT_1',
'QS6_CAT_2',
'QS6_CAT_3',
'QS7_CAT_1.0',
'QS7_CAT_2.0',
'QS7_CAT_3.0',
'QS7_CAT_4.0',
'QS7_CAT_5.0',
'QS7_CAT_MISSING',
'QS9_CAT_1',
'QS9_CAT_2',
'QS9_CAT_3',
'QS9_CAT_4',
'QS9_CAT_5',
'QS9_CAT_6',
'QS9_CAT_7',
'QS9_CAT_8'
]


# In[67]:

iv_cont=['Confidence_system',
'Fin_incentive',
'Financial_concern_all',
'Motivation',
'Non_fin_incentive',
'QBMI_CONT',
'QC17_6_ORD',
'QS1_CONT',
'QS11',
'QS8C_CONT',
'Reliance_on_others',
'Retail_clinic',
'Tech_insurer_all',
'Tech_manage_all',
'Tech_use_all']


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



