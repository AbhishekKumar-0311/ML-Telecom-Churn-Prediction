# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Telecom Churn Prediction

# #### Business Problem Overview
# - In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# - For many incumbent operators, retaining high profitable customers is the number one business goal.
# - To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
#
# In this project, you will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.

# ## Index
#
# 1. Environment Setup
# 2. Reading the Input data (csv) file
# 3. Data Analysis & Cleaning & Derivation
# 4. Data Visualization
# 5. Data Preparation
#     1. Train Test Split
#     2. Scaling - RobustScaler
#     3. Fixing Imbalance - SMOTE    
# 6. Data Modelling
#     1. PCA + Logistic Regression + Hyperparameter Tuning
#     2. Decision Tree Classification
#     3. Random Forest Classifier + Hyperparameter Tuning
#     4. XGBoost
# 7. Feature Selection
#     1. RFE + Logistic Regression + Hyperparameter Tuning
# 8. Summary

# ## 1. Environment Setup

# +
# To get multiple outputs in the same cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# +
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# +
# Importing data prep and EDA, plotting Libraries

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

# +
# Importing Machine learning Scikit-learn Libraries

#Feature Scaling, hyper parameter tuning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

# from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

## Evaluation Metrics
from sklearn.metrics import classification_report,  auc, roc_auc_score, roc_curve, precision_recall_curve,make_scorer
from sklearn.metrics import precision_score,recall_score, accuracy_score, confusion_matrix, f1_score,r2_score

#building models
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn import set_config
set_config(print_changed_only=True)

#imbalance data balance
from imblearn.metrics import sensitivity_specificity_support

# +
# Set the required global options

# To display all the columns in dataframe
pd.set_option( "display.max_columns", None)
pd.set_option( "display.max_rows", None)
# +
# from google.colab import drive
# drive.mount('/content/drive')
# -


# ## 2. Reading the Input data (csv) file

# +
tel = pd.read_csv('./telecom_churn_data.csv')

# G Colab
# tel = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/telecom_churn_data.csv')
# -

tel.head()

# ## 3. Data Analysis & Cleaning

# Checking rows and columns - shape 
tel.shape

# Getting the overview of Data types and Non-Null info
tel.info()

#Rename vbc columns to corresponding monthnumber to keep it consistent
tel.rename(columns = {'jun_vbc_3g':'vbc_3g_6',
                               'jul_vbc_3g':'vbc_3g_7',
                               'aug_vbc_3g':'vbc_3g_8',
                               'sep_vbc_3g':'vbc_3g_9'}, inplace=True)

# +
# create column name list by types of columns
id_cols = ['mobile_number', 'circle_id']

date_cols = ['last_date_of_month_6',
             'last_date_of_month_7',
             'last_date_of_month_8',
             'last_date_of_month_9',
             'date_of_last_rech_6',
             'date_of_last_rech_7',
             'date_of_last_rech_8',
             'date_of_last_rech_9',
             'date_of_last_rech_data_6',
             'date_of_last_rech_data_7',
             'date_of_last_rech_data_8',
             'date_of_last_rech_data_9'
            ]

cat_cols =  ['night_pck_user_6',
             'night_pck_user_7',
             'night_pck_user_8',
             'night_pck_user_9',
             'fb_user_6',
             'fb_user_7',
             'fb_user_8',
             'fb_user_9'
            ]

num_cols = [column for column in tel.columns if column not in id_cols + date_cols + cat_cols]

# print the number of columns in each list
print("#ID cols: %d\n#Date cols:%d\n#Numeric cols:%d\n#Category cols:%d" % (len(id_cols), len(date_cols), len(num_cols), len(cat_cols)))

# check if we have missed any column or not
print(len(id_cols) + len(date_cols) + len(num_cols) + len(cat_cols) == tel.shape[1])
# -

# drop id and date columns
# id columns - as they are moble number - unique for every user, circle_id = 109 corresponds to single circle
# Date columns - does not much information to the analysis
print("Shape before dropping: ", tel.shape)
tel = tel.drop(id_cols, axis=1)
print("Shape after dropping: ", tel.shape)

# #### Impute Categorical values

# +
# Replace NaN values in categorical variables
# We will replace missing values in the categorical values with '-1' where '-1' will be a new category.

# replace missing values with '-1' in categorical columns
tel[cat_cols] = tel[cat_cols].apply(lambda x: x.fillna(-1))
# -

# #### Imputation of Numerical variables

tel[['count_rech_2g_6','count_rech_3g_6','total_rech_data_6']].tail(20)

# ##### But, digging deep into the columns, there is a kind of relation among these variables.
# - total_rech_data - total number of recharges done in a month (= count_rech_2g+count_rech_3g). So, count_rech_2g, count_rech_3g columns can be removed

cols_todel = ['count_rech_2g_6','count_rech_3g_6','count_rech_2g_7','count_rech_3g_7','count_rech_2g_8','count_rech_3g_8','count_rech_2g_9','count_rech_3g_9']
tel = tel.drop(cols_todel,axis=1,errors='ignore')

num_cols = [column for column in tel.columns if column not in date_cols + cat_cols]

#recharge_cols = [col for col in num_cols if (('rech' in col) & ('rech_num' not in col))]
recharge_cols = [col for col in num_cols if (('rech' in col))]

recharge_cols

# some recharge columns have minimum value of 1 while some don't
tel[recharge_cols].describe(include='all')

# It is also observed that the recharge date and the recharge value are missing together which means the customer didn't recharge
tel.loc[tel.total_rech_data_6.isnull() & tel.date_of_last_rech_data_6.isnull(), ["total_rech_data_6", "date_of_last_rech_data_6"]].head(20)

# Date columns - does not much information to the analysis
print("Shape before dropping: ", tel.shape)
tel = tel.drop(date_cols, axis=1)
print("Shape after dropping: ", tel.shape)

# +
# Checking for any Null columns
tel.isnull().sum().any()

tel.shape[0]

# Finding the columns with more than 40% NULLs.
ser = tel.isnull().sum()/len(tel)*100
nulls = ser[ser > 40]
nulls
# -

# Checking the info of the remaining columns with NULLs
tel[nulls.index].info()

# Checking the data of the columns with NULLs
tel[nulls.index[:]].sample(4)

# +
#nulls.index

# creating a list of NULL columns of type float
nullsf = nulls.index[:]
nullsf

# +
# let's impute the columns av_rech_amt_data_* , total_rech_data_* for furthur usage

tel[nullsf]=tel[nullsf].fillna(0)

# +
# Verifying, whether floating variables are successfully imputed

ser = tel.isnull().sum()/len(tel)*100
nulls = ser[ser > 40]
nulls
# -

# - All the columns with more than 70% missing values are imputed with zero.

# +
# Checking for any Null columns
tel.isnull().sum().any()

# Finding the columns with more than 40% NULLs.
ser = tel.isnull().sum()/len(tel)*100
nullsgtzero = ser[ser > 0]
nullsgtzero
# -

# - All the missing values are for columns corresponding to incoming/outgoing calls within or outside network. This might be the customer has not utilised and can be imputed with 0.
# - And dropping date_of_last_rech_6,_7,_8 as they are dates and cannot be imputed.

# +
# Imputing the rest of the columns with 0

cols = tel[nullsgtzero.index].select_dtypes(exclude = 'object').columns
tel[cols] = tel[cols].fillna(0)

# +
# Finding the columns with NULLs.

ser = tel.isnull().sum()/len(tel)*100
nulls = ser[ser > 0]
nulls.sort_values(ascending=False)
# -

# Shape of Dataframe After dropping 
tel.shape

# Checking for any Null columns
tel.isnull().sum().any()

# ### Revenue is mostly generated by high value customers with 70th percentile of the total amount spent in the good phase

# +
## Derive the total data recharge amount columns for months 6,7,8,9

# mon_list = ['_6','_7'.'_8','_9']

for i in range(6,10):
    count = 'total_rech_data_' + str(i)
    avg_amt = 'av_rech_amt_data_' + str(i)
    tot_amt = 'total_rech_amnt_data_' + str(i)
    
    tel[tot_amt] = tel[avg_amt] * tel[count]

# -

# Verifying the newly derived features
tel.iloc[:,-4:].head()

# +
# Lets compute the average recharge amount for the month 6 & 7. This total amount is equal to the sum of talk time recharge 
# and data recharge amounts for the respective months.

avg_recharge_amnt_months_6_7 = tel[['total_rech_amnt_data_6','total_rech_amnt_data_7','total_rech_amt_6',
                                             'total_rech_amt_7']].mean(axis=1)
# type(avg_recharge_amnt_months_6_7)

amount_70th_percentile = np.percentile(avg_recharge_amnt_months_6_7, 70)

print("70th percentile of the average recharge amount in the first two months is - ", amount_70th_percentile)
# -

tel.shape
df_highvalue_cust = tel[avg_recharge_amnt_months_6_7 >= amount_70th_percentile]
df_highvalue_cust.shape

# #### Now to tag churned customers (churn=1, else 0) based on the churn month as follows: Those who have not made any calls (either incoming or outgoing) and have not used mobile internet even once in the churn phase. The attributes i am using to tag churners are:
#
# - total_ic_mou_9
# - total_og_mou_9
# - vol_2g_mb_9
# - vol_3g_mb_9
#
# ###### We will create a temporary dataset that stores all the parameters/features related to the tagging a customer as churn. We will use the above attributes mentioned.

churn_parameters_data = df_highvalue_cust[['total_ic_mou_9', 'total_og_mou_9', 'vol_2g_mb_9', 'vol_3g_mb_9']]
churn_parameters_data.head()

churn_parameters_data.isnull().sum()

# ##### Check if the customer has used any mobile calls or data in the fourth month.
# - If not used any services: Churn(1)
# - else: non-Churn(0)

df_highvalue_cust['Churn'] = df_highvalue_cust.apply(lambda x:1 if ((x.total_ic_mou_9 == 0)&
                                                                    (x.total_og_mou_9 == 0)&
                                                                   (x.vol_2g_mb_9 == 0)&
                                                                   (x.vol_3g_mb_9 == 0)) else 0,axis=1)

# #### Now, that churners are tagged from churn phase. We can remove all the columns corresponding to 9th month

cols_9_todel = [col for col in df_highvalue_cust.columns if col.endswith('_9')]
print('Total cols ending with _9 : ', len(cols_9_todel))
cols_9_todel

# Dropping the columns related to churn month, here the 9th month
df_highvalue_cust.drop(columns = cols_9_todel, inplace = True, errors = 'ignore')

# +
# Checking whether the columns are dropped successfully

cols_9_todel = [col for col in df_highvalue_cust.columns if col.endswith('_9')]
print('Total cols ending with _9 : ', len(cols_9_todel))

df_highvalue_cust.shape
# -

# ### Checking Variance of the features
#
# - Dropping featurs with  0 & 1 variance.

# +
# Creating a list of columns having entirely unique or entirely constant values

col_list = []
for i in df_highvalue_cust.columns:
    if df_highvalue_cust[i].nunique() in (1, len(df_highvalue_cust)):
        i, df_highvalue_cust[i].nunique()
        col_list.append(i)
print('Column number with zero variance : ',len(col_list))
# -

df_highvalue_cust[col_list].sample(5)

# - As we can see that All the above columns have either unique or constant values, So i would drop these features.

# Dropping the entirely unique or entirely constant variables
df_highvalue_cust.shape
df_highvalue_cust.drop(columns = col_list, inplace = True, errors='ignore')
df_highvalue_cust.shape

# #### ARPU - Average revenue per user. One of the important parameter to network procider.
# - The total revenue generated during the standard time period should then be divided by the number of units or users.ref-https://www.investopedia.com/terms/a/arpu.asp
# - ARPU can never be negative as it is Total revenue per total number of subscribers. So, let remove those rows as this might be data entry issue and so this rows cannot be trusted

df_highvalue_cust[['arpu_6','arpu_7','arpu_8']].describe()

cols_neg_arpu = df_highvalue_cust.columns[(df_highvalue_cust < 0).any()].tolist()

(df_highvalue_cust[cols_neg_arpu]<0).sum(axis=0)

for col in cols_neg_arpu[0:3]:
    #print(type(col))
    #print(df_highvalue_cust[col].head())
    df_highvalue_cust = df_highvalue_cust[(df_highvalue_cust[col] >= 0)]

df_highvalue_cust.shape #--(30001, 149)

# #### Now, let's understand the remaining columns and remove if there are any dependent columns

# - There ar emany columns which represents incoming/outgoing calls made the within operator, outside operator & to customer care, which are totally represented by minutes of usage columns.
# - ARPU columns corresponding to 3g/2g for every  months are aggregatively represented in arpu_mon* columns.
# - average recharge amount is part of new derived column - total recharge amount
# - So, dropping this redundant columns.

# +
# Let's drop individual columns whose totals are available as a different attribute

redundant_cols = ['loc_ic_t2t_mou_6', 'loc_ic_t2t_mou_7', 'loc_ic_t2t_mou_8',
                   'loc_ic_t2m_mou_6', 'loc_ic_t2m_mou_7', 'loc_ic_t2m_mou_8',
                   'loc_ic_t2f_mou_6', 'loc_ic_t2f_mou_7', 'loc_ic_t2f_mou_8',
                   'std_ic_t2t_mou_6', 'std_ic_t2t_mou_7', 'std_ic_t2t_mou_8',
                   'std_ic_t2m_mou_6', 'std_ic_t2m_mou_7', 'std_ic_t2m_mou_8',
                   'std_ic_t2f_mou_6', 'std_ic_t2f_mou_7', 'std_ic_t2f_mou_8',
                   'loc_og_t2t_mou_6', 'loc_og_t2t_mou_7', 'loc_og_t2t_mou_8',
                   'loc_og_t2m_mou_6', 'loc_og_t2m_mou_7', 'loc_og_t2m_mou_8',
                   'loc_og_t2f_mou_6', 'loc_og_t2f_mou_7', 'loc_og_t2f_mou_8',
                   'std_og_t2t_mou_6', 'std_og_t2t_mou_7', 'std_og_t2t_mou_8',
                   'std_og_t2m_mou_6', 'std_og_t2m_mou_7', 'std_og_t2m_mou_8',
                   'std_og_t2f_mou_6', 'std_og_t2f_mou_7', 'std_og_t2f_mou_8',
                   'last_day_rch_amt_6','last_day_rch_amt_7','last_day_rch_amt_8',
                   'arpu_3g_6', 'arpu_3g_7', 'arpu_3g_8',
                   'arpu_2g_6', 'arpu_2g_7', 'arpu_2g_8',
                   'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']

df_highvalue_cust.drop(redundant_cols, axis = 1, inplace = True)

df_highvalue_cust.shape
# -

# ### Deriving new Features

# Let's derive some variables. The most important feature, in this situation, can be the difference between the 8th month and the previous months. The difference can be in patterns such as usage difference or recharge value difference. Let's calculate difference variable as the difference between 8th month and the average of 6th and 7th month.

# +
# Deriving columns detail after substracting the action phase i.e 8th columns with the 6th and 7th columns

cols = ['arpu_','onnet_mou_','offnet_mou_','roam_ic_mou_','roam_og_mou_','loc_og_mou_','std_og_mou_','isd_og_mou_','spl_og_mou_','total_og_mou_','loc_ic_mou_','std_ic_mou_','isd_ic_mou_','spl_ic_mou_','total_ic_mou_','total_rech_num_','total_rech_amt_','max_rech_amt_','total_rech_data_','max_rech_data_','vol_2g_mb_','vol_3g_mb_']

for i in cols:
    col1 = i + str('6')
    col2 = i + str('7')
    col3 = i + str('8')
    col4 = i + str('diff_avg')
    
    ## -ve shows the user is not using the services actively as before
    df_highvalue_cust[col4] = df_highvalue_cust[col3] - ((df_highvalue_cust[col1] + df_highvalue_cust[col2])/2)

# +
# Dropping the set of columns as the _diff_avg features are derived from these columns.

# for i in range(6,9):
#     for j in cols:
#         col = j + str(i)
#         df_highvalue_cust.drop(columns = [col], inplace = True)

# +
# Deriving aggregated columns from _6th and _7th months as _goodphase

cols = [ 'og_others_', 'ic_others_','night_pck_user_','monthly_2g_','monthly_3g_','sachet_2g_','sachet_3g_','fb_user_','vbc_3g_','total_rech_amnt_data_']

for i in cols:
    col1 = i + str('6')
    col2 = i + str('7')
    col3 = i + str('goodph')
    col4 = i + str('8')
    col5 = i + str('actionph')
    col6 = i + str('drop')
    
    df_highvalue_cust[col3] = ((df_highvalue_cust[col1] + df_highvalue_cust[col2])/2)
    df_highvalue_cust[col5] = df_highvalue_cust[col4]
    #df_highvalue_cust[col6] = [(df_highvalue_cust[col3] - df_highvalue_cust[col5]) > 0 ] = 1 else 0
    #df_highvalue_cust.loc[((df_highvalue_cust[col3] - df_highvalue_cust[col5]) > 0), col6] = 1
    #df_highvalue_cust.loc[((df_highvalue_cust[col3] - df_highvalue_cust[col5]) < 0), col6] = 0
    df_highvalue_cust[col6] = np.where(((df_highvalue_cust[col3] - df_highvalue_cust[col5]) <= 0),0,1)
    
    df_highvalue_cust.drop(col3,axis=1,inplace=True)
    df_highvalue_cust.drop(col5,axis=1,inplace=True)


# +
# Dropping the set of columns as the _diff_avg features are derived from these columns.

for i in range(6,9):
    for j in cols:
        col = j + str(i)
        df_highvalue_cust.drop(columns = [col], inplace = True)
# -

# cols = ['arpu_','total_og_mou_','total_ic_mou_','total_rech_num_','total_rech_amt_','total_rech_data_','total_rech_amnt_data_']
# cols = [ 'og_others_', 'ic_others_','night_pck_user_','monthly_2g_','monthly_3g_','sachet_2g_','sachet_3g_','fb_user_','vbc_3g_']


# +
# Deriving usage/revenue drop at subsequent months
# cols = ['arpu_','onnet_mou_','offnet_mou_','roam_ic_mou_','roam_og_mou_','loc_og_mou_','std_og_mou_','isd_og_mou_','spl_og_mou_','total_og_mou_','loc_ic_mou_','std_ic_mou_','isd_ic_mou_','spl_ic_mou_','total_ic_mou_','total_rech_num_','total_rech_amt_','max_rech_amt_','total_rech_data_','max_rech_data_','vol_2g_mb_','vol_3g_mb_']

cols = ['loc_og_t2c_mou_','arpu_','onnet_mou_','offnet_mou_','roam_ic_mou_','roam_og_mou_','loc_og_mou_','std_og_mou_','isd_og_mou_','spl_og_mou_','total_og_mou_','loc_ic_mou_','std_ic_mou_','isd_ic_mou_','spl_ic_mou_','total_ic_mou_','total_rech_num_','total_rech_amt_','total_rech_data_','vol_2g_mb_','vol_3g_mb_']

for i in cols:
    col1 = i + str('6')
    col2 = i + str('7')
    col3 = i + str('8')
    col4 = i + str('drop_1')
    col5 = i + str('drop_2')
    col6 = i + str('drop_ovrall')
    
    ## -ve shows the user is not using the services actively as before
    df_highvalue_cust[col4] = df_highvalue_cust[col1] - df_highvalue_cust[col2]
    df_highvalue_cust[col5] = df_highvalue_cust[col2] - df_highvalue_cust[col3]
    df_highvalue_cust.loc[(df_highvalue_cust[col4] > 0) & (df_highvalue_cust[col5] > 0), col6] = 1
    df_highvalue_cust.loc[~(df_highvalue_cust[col4] > 0) | ~(df_highvalue_cust[col5] > 0), col6] = 0
    df_highvalue_cust.drop(col4,axis=1,inplace=True)
    df_highvalue_cust.drop(col5,axis=1,inplace=True)
#     df_highvalue_cust[col6] = df_highvalue_cust.loc[(df_highvalue_cust[col4] > 0) & (df_highvalue_cust[col5] > 0)]


#df_highvalue_cust.drop(columns = [col4 ])


# +
# Dropping the set of columns as the _diff_avg features are derived from these columns.

cols = ['loc_og_t2c_mou_','arpu_','onnet_mou_','offnet_mou_','roam_ic_mou_','roam_og_mou_','loc_og_mou_','std_og_mou_','isd_og_mou_','spl_og_mou_','total_og_mou_','loc_ic_mou_','std_ic_mou_','isd_ic_mou_','spl_ic_mou_','total_ic_mou_','total_rech_num_','total_rech_amt_','max_rech_amt_','total_rech_data_','max_rech_data_','vol_2g_mb_','vol_3g_mb_']

for i in range(6,9):
    for j in cols:
        col = j + str(i)
        df_highvalue_cust.drop(columns = [col], inplace = True)
# -

df_highvalue_cust.info(verbose=True)

df_highvalue_cust.shape


# ## 4. Data Visualization

# +
## Show labels in bar plots

def showLabels(ax, d=None):
    plt.margins(0.2, 0.2)
    rects = ax.patches
    i = 0
    locs, labels = plt.xticks() 
    counts = {}
    if not d is None:
        for key, value in d.items():
            counts[str(key)] = value

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if d is None:
            label = "{:.1f}".format(y_value)
        else:
            try:
                label = "{:.1f}".format(y_value) + "\nof " + str(counts[str(labels[i].get_text())])
            except:
                label = "{:.1f}".format(y_value)
        
        i = i+1

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# -

def default_rate_per_var(var, df = df_highvalue_cust, sort_flg=True, head=0):
    
    plt.subplot(1, 2, 1)
    if head == 0:
        ser = (df[var].value_counts(normalize=True)*100)
    else:
        ser = (df[var].value_counts(normalize=True).head(head)*100)
    #ser
    if sort_flg:
        ser = ser.sort_index()
    ax = ser.plot.bar(color=sns.color_palette("Paired", 10))
    ax.set_ylabel('% count in data', fontsize=16)
    ax.set_xlabel(var, fontsize=12)
    showLabels(ax)
    plt.subplot(1, 2, 2)
    if head == 0:
        ser = (df.loc[df['Churn'] == 1][var].value_counts(normalize=True)*100)
    else:
        ser = (df.loc[df['Churn'] == 1][var].value_counts(normalize=True).head(head)*100)
    #ser
    if sort_flg:
        ser = ser.sort_index()
    ax = ser.plot.bar(color=sns.color_palette("Paired", 10))
    ax.set_ylabel('% in Churners', fontsize=16)
    ax.set_xlabel(var, fontsize=12)
    showLabels(ax)
    plt.show()


cols = [x for x in df_highvalue_cust.columns if x.endswith('ovrall')]

for i in cols:
    plt.figure(figsize=(12,7));
    default_rate_per_var(i);

# - **From the above plots it resembles that arpu,onnet,offnet,total outgoing,incoming,number of recharges and total amount of recarge are highly dependent features for predicting churn.**

cols = [x for x in df_highvalue_cust.columns if x.endswith('diff_avg')]

plt.figure(figsize=(17,13))
sns.heatmap(df_highvalue_cust[cols].corr(), annot = True);


# This user-defined function plots the distribution of target column, and its boxplot against Churn column
def plot_distribution(var):
    plt.figure(figsize=(17,9))
    plt.subplot(1, 2, 1)
    ax = sns.histplot(data=df_highvalue_cust, x=var, kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=var, y= 'Churn', data=df_highvalue_cust)
    plt.show()

# +
# for i in cols:
#     plot_distribution(i)
# -


df_highvalue_cust.head()


df_highvalue_cust.shape
telecom = df_highvalue_cust.copy()
telecom.shape

# ## 5. Data Preparation

# #### The below steps are to be done before Data Modelling
#
# 1. Split the dataset
# 2. Scale the data
# 3. SMOTE + undersampling - to overcome the imbalace in the data

# +
# Importing Machine learning Scikit-learn Libraries

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score
from sklearn import metrics
from imblearn.metrics import sensitivity_specificity_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
# -

# ### Train Test split

# +
# Divide data into train and test

X = telecom.drop("Churn", axis = 1)
y = telecom.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70, test_size = 0.30, random_state = 100, stratify = y)
# -

X_train.shape
y_train.shape
X_test.shape
y_test.shape

# ### Scaling

# +
# Scale the data using MinMaxScaler

#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Scale the data to overcome the outlier impact and bring the data centered to zero median
scaler = RobustScaler(quantile_range=(1, 99))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# -

# ### Class Imbalance Check

# +
# change data type to category
# telecom['Churn'] = telecom['Churn'].astype("category")

telecom.Churn.value_counts()

# print churn ratio
print("Churn Ratio:")
telecom['Churn'].value_counts()*100/len(telecom)
# -

# - **This seems to be imbalanced dataset. So we need to make it balanced before prediction.**

# #### Pre-Processing Techniques
# - https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/
# As a part of pre-processing stage of ML pipelines prior, the following algorithms will be used for handling imbalanced dataset.
#
# - Undersampling
# - Random undersampling
# - Oversampling
# - Random oversampling: generates new samples by random resampling with replacement of under represented class
# - Synthetic Minority Oversampling (SMOTE)
# - Combined over and under sampling
# - SMOTEENN
# - SMOTETomek
# ##### Training techniques
# Number of learning models themselves do provide some built in support to deal with imbalance data.
#
# Sample weighting
#
# #### Fact:
#
# SMOTE allows to generate samples. However, this method of over-sampling does not have any knowledge regarding the underlying distribution. Therefore, some noisy samples can be generated, e.g. when the different classes cannot be well separated. Hence, it can be beneficial to apply an under-sampling algorithm to clean the noisy samples. Imbalanced-learn provides two ready-to-use combined samplers:
#
# SMOTETomek
# SMOTEENN
# Both the methods are good but in general, SMOTEENN cleans more noisy data than SMOTETomek.
#
# #### Note:
#
# It is not possible to check different sampling techniques on very cost sensitive Machine Learning models like SVM, Decision Trees, Random Forest. For this Case Study, we will particularily use SMOTEENN sampling technique to handle imbalanced dataset as it is uses both over-sampling and under-sampling method and helps in cleaning noisy samples.

# +
# Oversample with SMOTE and random undersample for imbalanced dataset

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

# counter = Counter(y_train)
# print('before : ' , counter)

# # define pipeline
# over = SMOTE(sampling_strategy=0.5)
# under = RandomUnderSampler(sampling_strategy=0.5)
# steps = [('o', over), ('u', under)]
# pipeline = Pipeline(steps=steps)

# # transform the dataset
# X_train_bal, y_train_bal = pipeline.fit_resample(X_train_scaled, y_train)

# # summarize the new class distribution
# counter = Counter(y_train_bal)
# print('after  : ' , counter)
# -

# ### Handling Class Imbalance using SMOTE

# +
#### Implement SMOTE to balance the imbaance in the data

counter = Counter(y_train)
print('before : ' , counter)

over = SMOTE(random_state=100,n_jobs=-1)
X_train_bal, y_train_bal = over.fit_sample(X_train_scaled, y_train)

# summarize the new class distribution
counter = Counter(y_train_bal)
print('after  : ' , counter)

print("X_train_bal: \n", X_train_bal.shape)
print("y_train_bal: \n", y_train_bal.shape)
# -

X_train_resampled = pd.DataFrame(data = X_train_bal)
y_train_resampled = pd.DataFrame(data = y_train_bal)
print("X_train_resampled: \n", X_train_resampled.shape)
print("y_train_resampled: \n", y_train_resampled.shape)


# ### Dimensionality Reduction

def perform_PCA(X_train):
    pca = PCA(svd_solver = 'randomized',random_state=100)
    
    #fit the data
    pca.fit(X_train)
    
    #plot cummulative variance against no. of components
    var_cumu = np.cumsum(pca.explained_variance_ratio_)
    fig = plt.figure(figsize=[8,4])
    #plt.vlines(x=15, ymax=1, ymin=0, colors="r", linestyles="--")
    plt.hlines(y=0.95, xmax=30, xmin=0, colors="g", linestyles="--")
    plt.plot(var_cumu)
    plt.ylabel("Cumulative variance explained")
    plt.show()


def perform_increpca(X_train,X_test,no_comp):
    pca_final = IncrementalPCA(n_components=no_comp)
    X_train_pca = pca_final.fit_transform(X_train)
    X_test_pca = pca_final.transform(X_test)

    X_train_pca = pd.DataFrame(data = X_train_pca)
    X_test_pca = pd.DataFrame(data = X_test_pca)

    print('X_train_data',X_train_pca.shape)
    #print(y_train_bal.shape)
    print('X_test_data',X_test_pca.shape)
    #print(y_test.shape)
    
    # create a correlation map for principal components derived from PCA
    corrmat = np.corrcoef(X_train_pca.transpose())
    
    #plotting the correlation matrix
    plt.figure(figsize = (20,10))
    sns.heatmap(corrmat, annot = True)
    plt.show()
    
    return X_train_pca, X_test_pca


perform_PCA(X_train_bal)
#perform_PCA(X_train_scaled)

# - For 95% variance the number of components to be choosen is 30

X_train_pca,X_test_pca = perform_increpca(X_train_bal,X_test_scaled,30)
#X_train_pca,X_test_pca = perform_increpca(X_train_scaled,X_test_scaled,40)

# - From above heatmap it shows that data is nicely spearated from each other features i.e. no multicollinearity

# ### User-defined functions for repetitive tasks for training & evaluation

def get_churnprob(df_train_pca):
    y_train_pred = model_pca.predict_proba(X_train_pca)[:,1]
    y_train_pred_final = pd.DataFrame({'Churn':y_train_bal, 'Churn_Prob':y_train_pred})
    # Let's create columns with different probability cutoffs 
    numbers = [float(x)/10 for x in range(10)]
    for i in numbers:
        y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)
    y_train_pred_final.head()
    # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','specificity'])        
    
    # TP = confusion[1,1] # true positive 
    # TN = confusion[0,0] # true negatives
    # FP = confusion[0,1] # false positives
    # FN = confusion[1,0] # false negatives
    num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in num:
        cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
        total1=sum(sum(cm1))
        accuracy = (cm1[0,0]+cm1[1,1])/total1
        speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
    print(cutoff_df)
    # Let's plot accuracy sensitivity and specificity for various probabilities.
    cutoff_df.plot.line(x='prob', y=['accuracy','sensi','specificity'])
    plt.show()


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


def get_scores(scores,model_pca,X_test_pca):
    #if prob_churn > 0:
        #y_test_pred_probs = model_pca.predict_proba(X_test_pca)[:,1]
        #y_test_pred = np.where(model_pca.predict_proba(X_test_pca)[:,1] > prob_churn, 1, 0)
        #y_test_df=pd.DataFrame(y_test)
        #y_pred_df=pd.DataFrame(y_test_pred_probs)
        #y_test_df.reset_index(drop=True, inplace=True)
        #y_pred_df.reset_index(drop=True, inplace=True)
        #y_test_pred_final=pd.concat([y_test_df, y_pred_df],axis=1)
        # Renaming the column
        #y_test_pred_final = y_test_pred_final.rename(columns={ 0 : 'Churn_prob'})
        #y_test_pred_final['final_predicted'] = y_test_pred_final.Churn_prob.map(lambda x: 1 if x > prob_churn else 0)
    #else:
    y_test_pred_probs = model_pca.predict_proba(X_test_pca)[:,1]
    y_test_pred = model_pca.predict(X_test_pca)
    test_confusion = confusion_matrix(y_test, y_test_pred)       
    TP = test_confusion[1,1] # true positive
    TN = test_confusion[0,0] # true negatives
    FP = test_confusion[0,1] # false positives
    FN = test_confusion[1,0] # false negatives
    
    # Let's see the parameters of our logistic regression model
    model_Accuracy = accuracy_score(y_test,y_test_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_probs, drop_intermediate = False )
    p,r,t = precision_recall_curve(y_test,y_test_pred)
    model_Recall = recall_score(y_test,y_test_pred)
    model_f1_score = f1_score(y_test,y_test_pred)
    model_Precision = precision_score(y_test,y_test_pred)
    model_auc_score = auc(fpr,tpr)
    model_roc_area = roc_auc_score(y_test,y_test_pred_probs)
    model_FalsePositiveRate = FP / float(FP + TN)
    model_Specificity = TN / float(TN + FP)
    model_FalseNegativeRate = FN / float(FN + TP)
    model_auc_roc = auc(fpr, tpr)
    model_auc_pr = auc(p,r)
    
    print('model_Accuracy-',model_Accuracy)
    print('model_Recall/Sensitivity-',model_Recall)
    print('model_Precision/TPR-',model_Precision)
    print('model_f1_score-',model_f1_score)
    print('model_auc_score-',model_auc_score)
    print('model_roc_area-',model_roc_area)
    print('FPR-',model_FalsePositiveRate)
    print('Specificity/TNR-',model_Specificity)
    print('FNR-',model_FalseNegativeRate)
    
    scores.append((model_f1_score,model_Precision,model_Recall,model_Accuracy,model_auc_score, model_auc_pr,test_confusion))  
    # Plot ROC and PR curves using all models and test data
   
    fig, axes = plt.subplots(1, 2, figsize = (14, 6))
    axes[0].plot(fpr, tpr, label = f"auc_roc = {model_auc_roc:.3f}")
    axes[1].plot(r, p, label = f"auc_pr = {model_auc_pr:.3f}")

    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].legend(loc = "lower right")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("AUC ROC curve")

    axes[1].legend(loc = "lower right")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("PR curve")

    plt.tight_layout()
    plt.show()
    #draw_roc(y_test,y_test_pred)
    #precision_recall_curve(y_test,y_test_pred)
    #plt.plot(thresholds, p, "g-")
    #plt.plot(thresholds, r, "r-")
    plt.show()
    return scores


def hypertuning_plot(scores, parameter):
    
    col = "param_" + parameter
    
    plt.figure()
    
    plt.plot(scores[col], scores["mean_train_score"], label = "training accuracy")
    plt.plot(scores[col], scores["mean_test_score"], label = "test accuracy")
    
    plt.xlabel(parameter)
    plt.ylabel("Accuracy")
    
    plt.legend()
    plt.show()


# ## 6. Model Building

# ### Logistic Regression

lr_pca = LogisticRegression(class_weight='balanced')
model_pca = lr_pca.fit(X_train_pca, y_train_bal)
#get_churnprob(X_train_pca)

scores = []
scores = get_scores(scores,model_pca,X_test_pca)
# Tabulate results
sampling_results = pd.DataFrame(scores, columns = ['f1', 'precision', 'recall', 'accuracy',
                                                   'auc_roc','auc_pr', 'confusion_matrix'])
sampling_results

# +
##Logistic Regression - HyperTuning Penalty

# GridSearchCV to find best penalty

lr = LogisticRegression(class_weight={0:0.1, 1: 0.9})

parameter = ['penalty','C']

# parameters to build the model on
param_grid = {'penalty': ['l1', 'l2','none'],
              'C': [0.1,0.5,1,5,10,50,100,400,500,1000]
             }
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 100)

gc = GridSearchCV(estimator = lr, param_grid = param_grid, scoring = 'recall',n_jobs = -1, cv = folds, verbose = 2,return_train_score=True)   
gc.fit(X_train_pca,y_train_bal)

#print(scores)
# Plot the scores
#for param in parameter:
   # print(param)
    #hypertuning_plot(scores, param)
# scores of GridSearch CV
cv_results = pd.DataFrame(gc.cv_results_)
# Get the best value
gc.best_params_
# -

cv_results[cv_results['rank_test_score']==1].head()

print("Best AUC: ", gc.best_score_)
print("Best hyperparameters: ", gc.best_params_)

# +
# predict churn on test data
y_pred = gc.predict(X_test_pca)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = gc.predict_proba(X_test_pca)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))

# +
# Logistic with best parameters obtained from grid search

lr = LogisticRegression(penalty = 'l2', C = 0.1, class_weight={0:0.4, 1: 0.6})

lrf = lr.fit(X_train_pca,y_train_bal)
#lef.predict(X_test_pca.values)[:, 1:]
# Get the Score Metrics and plots
scores = []

scores = get_scores(scores, lrf, X_test_pca)

# Tabulate results
sampling_results = pd.DataFrame(scores, columns = ['f1', 'precision', 'recall', 'accuracy',
                                                   'auc_roc', 'auc_pr', 'confusion_matrix'])
sampling_results
# -

print(classification_report(y_test, gc.predict(X_test_pca), target_names=['0','1']))

# ### Decision Trees

perform_PCA(X_train_bal)

X_train_dt,X_test_dt = perform_increpca(X_train_bal,X_test,30)

# +
from sklearn.tree import DecisionTreeClassifier

score = make_scorer('auc_score', greater_is_better=True)
param_grid={'max_depth':[5,10,20,None],'max_features':['sqrt','log2',None],'class_weight':['balanced']}
gc = GridSearchCV(DecisionTreeClassifier(),cv=5,refit=True,param_grid=param_grid,scoring='recall')
gc.fit(X_train_dt,y_train_bal)
print('best estimator',gc.best_estimator_)
print('best score',gc.best_score_)
# -

print("Best AUC: ", gc.best_score_)
print("Best hyperparameters: ", gc.best_params_)

# +
# predict churn on test data
y_pred = gc.predict(X_test_dt)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = gc.predict_proba(X_test_dt)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))
# -

# ### Random Forest

perform_PCA(X_train_bal)
X_train_dt,X_test_dt = perform_increpca(X_train_bal,X_test,30)

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# RandomizedSearchCV - HPT 

n_estimators = [100, 200, 500, 700, 900, 1000] # no of tress [200,210,220,230.....20000]
max_features = ['auto', 'sqrt']
max_depth = [4,5,6,7,8]
min_samples_split = [2, 5, 8, 10]
min_samples_leaf = [1, 2, 3, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth, 
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'criterion' :['gini', 'entropy'],
               'bootstrap': bootstrap}

rand_forest = RandomForestClassifier(class_weight={0:0.1, 1: 0.9},random_state=100)

rf_random = RandomizedSearchCV(estimator=rand_forest, param_distributions=random_grid, n_iter=100, cv=3, 
                               verbose=2, random_state = 100, n_jobs=-1)
rf_random.fit(X_train_dt, y_train_bal)
# -

print("Best AUC: ", gc.best_score_)
print("Best hyperparameters: ", gc.best_params_)

# +
# predict churn on test data
y_pred = gc.predict(X_test_dt)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = gc.predict_proba(X_test_dt)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))
# -

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity:", round(sensitivity, 2))
print("Specificity:", round(specificity, 2))
# check area under curve
#y_pred_prob = gc.predict_proba(X_test_dt)[:, 1]
print("AUC:", round(roc_auc_score(y_test, y_pred_prob),2))
print('model_Recall:',round(recall_score(y_test,y_pred),2))
print('model_f1_score:',round(f1_score(y_test,y_pred),2))
print('model_Precision:',round( precision_score(y_test,y_pred),2))

# ### XGBOOST

from xgboost import XGBClassifier

xgb_cfl = XGBClassifier(use_label_encoder=False,n_jobs = -1,objective = 'binary:logistic',eval_metric='error')
# Fit the model to our train and target
xgb_cfl.fit(X_train_dt, y_train_bal)  # default 
# Get our predictions
xgb_predictions = xgb_cfl.predict(X_test_dt)
#xgb_predictions_prob = 

# +
# predict churn on test data
y_pred = xgb_cfl.predict(X_test_dt)
y_pred_prob = xgb_cfl.predict_proba(X_test_dt)[:,1]
# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
#y_pred_prob = gc.predict_proba(X_test_dt)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))
# -

# ## Model to choose best features

df =df_highvalue_cust.copy()

df_highvalue_cust.head()

# +
# Divide data into train and test

X = df.drop("Churn", axis = 1)
y = df.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70, test_size = 0.30, random_state = 100, stratify = y)
# -

scaler_feat = RobustScaler()
X_train_scaled = scaler_feat.fit_transform(X_train)
X_test_scaled = scaler_feat.transform(X_test)

X_train = pd.DataFrame(data = X_train_scaled, index = X_train.index, columns = X_train.columns)
X_test = pd.DataFrame(data = X_test_scaled, index = X_test.index, columns = X_test.columns)

# ### RFE

# +

import statsmodels.api as sm
from sklearn.feature_selection import RFE
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# +
n_features_list = list(range(30, 40)) #checking for optimal number of features between 20 to 60
train_adjusted_r2 = []
train_r2 = []
test_r2 = []
train_RMSE=[]
test_RMSE=[]

for n_features in range(30, 40):

    # RFE with n features
    lm = LogisticRegression()

    # specifying number of features
    rfe_n = RFE(estimator=lm, n_features_to_select=n_features)

    # fit with n features
    rfe_n.fit(X_train, y_train)

    # selecting features selected by rfe_n
    col_n = X_train.columns[rfe_n.support_] #rfe_n.support_: returns an array with boolean values to indicate whether 
    #an attribute was selected using RFE

    # training & test data for n selected columns
    X_train_rfe_n = X_train[col_n]
    X_test_rfe_n = X_test[col_n]


    # add a constant to the model
    X_train_rfe_n = sm.add_constant(X_train_rfe_n)


    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')

    
    
    # fitting the model with n featues
    lm_sm = sm.OLS(y_train, X_train_rfe_n).fit()
    
    
    # # Making predictions
    y_pred_test = lm_sm.predict(X_test_rfe_n)
    y_pred_train = lm_sm.predict(X_train_rfe_n)
    
    
    #Calculating evaluation metrics
    
    #R-square
    train_adjusted_r2.append(lm_sm.rsquared_adj)
    train_r2.append(lm_sm.rsquared)
    test_r2.append(r2_score(y_test, y_pred_test))
    
    #RMSE/stan. error
    error_test=y_pred_test-y_test
    error_train=y_pred_train-y_train
    
    test_RMSE.append(((error_test**2).mean())**0.5)
    train_RMSE.append(((error_train**2).mean())**0.5)

# +
# plotting r2 and RMSE against n_features
#reference from web and modified accordingly
import matplotlib.ticker as plticker

fig,ax=plt.subplots(2,1,figsize=(13, 9))
ax[0].plot(n_features_list, train_r2,'b', label="r2_train data")
ax[0].plot(n_features_list, test_r2,'g', label="r2_test data")
ax[0].set_xlabel('Features Count')
#method 1 of ticks
ax[0].legend(loc='upper left')
loc = plticker.MultipleLocator(base=1)
ax[0].xaxis.set_major_locator(loc)
#plt.show()

ax[1].plot(n_features_list, train_RMSE, 'b',label="RMSE_train data")
ax[1].plot(n_features_list, test_RMSE, 'g',label="RMSE_test data")
ax[1].set_xlabel('Features Count')
#method 2 of ticks
ax[1].legend(loc='upper left')
plt.xticks(np.arange(0, 51, step=1))

plt.show()
# -

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 35)             # running RFE with 35 variables as output
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))

X_train.columns[rfe.support_]

X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_test_rfe = X_test[X_train.columns[rfe.support_]]

X_train_rfe.head()

# #### create a heatmap to check correlation

corr_val = X_train_rfe.corr()
corr_val.loc[:,:] = np.tril(corr_val, k=-1)
corr_val = corr_val.stack()
val = corr_val[(corr_val >= 0.60) | (corr_val <= -0.60)].sort_values()

corre_values = val.index.tolist()
corre_values

cols_todrop = []
for i in range(len(corre_values)):
    cols_todrop.append(corre_values[i][1])
cols_todrop

plt.figure(figsize=(17,13))
sns.heatmap(X_train_rfe.corr(), annot = True);
plt.show()

# Drop columns with high correlation from above heatmap
#cols_todrop = ['night_pck_user_drop','loc_ic_mou_drop_ovrall','total_rech_amt_drop_ovrall','spl_ic_mou_drop_ovrall','total_og_mou_drop_ovrall']
X_train_rfe = X_train_rfe.drop(cols_todrop,axis=1)
X_test_rfe = X_test_rfe.drop(cols_todrop,axis=1)

# #### Assessing model using rfe columns

X_train_sm = sm.add_constant(X_train_rfe)
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# - based on above summary it make confident enogh that there is not much possibility of multi-collinearity among features.
#  - X_train_rfe - no of features to train the model

# +
# logistic regression
steps = [
         ("logistic", LogisticRegression(class_weight='balanced'))]

# compile pipeline
logistic = Pipeline(steps)

# hyperparameter space
params = {'logistic__C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'logistic__penalty': ['l1', 'l2']}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 100)

# create gridsearch object
model = GridSearchCV(estimator=logistic, cv=folds, param_grid=params, scoring='recall', n_jobs=-1, verbose=1)
# -

# fit model
model.fit(X_train_rfe, y_train)

# print best hyperparameters
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)

# +
# predict churn on test data
y_pred = model.predict(X_test_rfe)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = model.predict_proba(X_test_rfe)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))
# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity:", round(sensitivity, 2))
print("Specificity:", round(specificity, 2))
# check area under curve
#y_pred_prob = gc.predict_proba(X_test_dt)[:, 1]
print("AUC:", round(roc_auc_score(y_test, y_pred_prob),2))
print('model_Recall:',round(recall_score(y_test,y_pred),2))
print('model_f1_score:',round(f1_score(y_test,y_pred),2))
print('model_Precision:',round( precision_score(y_test,y_pred),2))
# -

model.best_estimator_



# +
# Logistic with best parameters obtained from grid search

lr = LogisticRegression( C = 0.1,class_weight='balanced', n_jobs = -1, random_state = 100)

lrf = lr.fit(X_train_rfe,y_train)
#lef.predict(X_test_pca.values)[:, 1:]
# Get the Score Metrics and plots
scores = []

print('Test Data Evaluataion Scores')
scores = get_scores(scores, lrf, X_test_rfe)

# Tabulate results
sampling_results = pd.DataFrame(scores, columns = ['f1', 'precision', 'recall', 'accuracy',
                                                   'auc_roc', 'auc_pr', 'confusion_matrix'])
sampling_results
# -

# #### Get feature wise importance

model_parameter = lr.coef_.tolist()
model_parameter = model_parameter[0]
#model_parameter.append()
model_parameter.insert(0,lr.intercept_[0])
cols = X_train_rfe.columns
cols = cols.insert(0,'constant')
lr_coef = pd.DataFrame(list(zip(cols,model_parameter)))
lr_coef.columns = ['Feature','Coef']

lr_coef = lr_coef.sort_values(by='Coef',ascending=False)
lr_coef.Feature

plt.figure(figsize=(45,25))
splot = sns.barplot(x='Feature',y='Coef',data=lr_coef.head(15),orient='v')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0,5), 
                   textcoords = 'offset points')
plt.title('Best fitline coefficients')
plt.xlabel('Predictor Variable')
plt.ylabel('Coefficients')
plt.show()

# ## Summary

# #### Top features affecting the customers to churn
# -                  fb_user_drop
# -       roam_og_mou_drop_ovrall
# -      total_ic_mou_drop_ovrall
# -         onnet_mou_drop_ovrall
# -                   vbc_3g_drop
# -        std_ic_mou_drop_ovrall
# -              arpu_drop_ovrall
# -        std_og_mou_drop_ovrall
# -        isd_ic_mou_drop_ovrall
# -    loc_og_t2c_mou_drop_ovrall

# #### Business Insights
#
# - Customers who are facebook users and use fb packs tend to churn more, if their facebook recharge packs drop gradually. 
#
# - The telecom company must focus on the roaming rates. They could provide good offers to customer using roaming services. Moreover it is also possible that the roaming network of the telecom company might be weak in some areas due to which customers might be churning.
#
# - Company must also focus on STD and ISD rates. Perhaps, the rates are too high. Provide them with some kind of STD and ISD packages.
#
# - Volume  Based Cost - is always blind fold to customer. So if there is any slight change in price or speed. Customer dissatisfaction leads to churn.
