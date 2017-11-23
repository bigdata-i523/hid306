#************************************************************
# module: analyis - exploratory data analysis
#***********************************************************

#Loading libraries
import numpy as np 
import pandas as pd
from fancyimpute import KNN

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import skew
from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import imputer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import plots

#****************************************************************
# method: analyze_missing_values(df)
# purpose: fill NA values of numeric variables using MICE package
#*****************************************************************
def _analyze_missing_values(df, saveAs = ''):

    #check for any null values?
    hasNullValues = df.columns[df.isnull().any()]
    print('df has any null values?', hasNullValues)
    
    #missing value counts in each of these columns
    miss = df.isnull().sum()/len(df)
    miss = miss[miss > 0]
    miss.sort_values(inplace=True)
    print(miss)

    #visualising missing values
    miss = miss.to_frame()
    miss.columns = ['Count']
    miss.index.names = ['Name']
    miss['Name'] = miss.index

    #plot the missing value count
    plots.bar_plot(miss, 'Name', 'Count', saveAs)

#end analyze_missing_values

#****************************************************************
# method: _numeric_imputation(df_numeric)
# purpose: fill the null values of numeric variables using FancyImpute package
#*****************************************************************
def _numeric_imputation(df_numeric):
           
    df_filled = pd.DataFrame(KNN(k=3).complete(df_numeric.as_matrix()))
    df_filled.columns = df_numeric.columns
    df_filled.index = df_numeric.index     
        
    df_numeric = df_filled
        
#end numeric_imputation


#****************************************************************
# method: _analyze_numeric_variables(df)
# purpose: 
#*****************************************************************
def _analyze_numeric_variables(df_numeric):

    plot = sns.FacetGrid(df_numeric, col = 'variable', col_wrap = 2, sharex = False, sharey = False)
    
    plot = plot.map(sns.distplot, 'value')
    
    plots.save_plot(plot, "num_variables")
    
    
#end _analyze_numeric_variables
    
#****************************************************************
# method: _analyze_correlations(df)
# purpose: 
#*****************************************************************
def _analyze_correlations(df_numeric, target_variable):
    
    print('..analyzing correlations...')
    corr = df_numeric.corr()
    plot = sns.heatmap(corr)
    
    plots.save_plot(plot, "correlations")
    
    #top 15 highly positively correlated variables with target
    print(corr[target_variable].sort_values(ascending = False)[:15])
    
    #bottom 5 highly positively correlated variables with target
    print(corr[target_variable].sort_values(ascending = False)[-5:])
#end _analyze_correlations
    
#****************************************************************
# method: adjust_skewed_data(df, numeric_features)
# purpose: replace skewed numeric data with corresponding log() values
#*****************************************************************
def _adjust_skewed_data(df, numeric_features, target_col):

    #before applying log(target_col)
    #sns.distplot(df[target_col])
    #print("the skewness of {0} is {1}, target_col, format(df[target_col].skew()))

    #after applying log(target_col)
    #log_target = np.log(df[target_col])
    #print('log({0}) skewness is {1}, target_col, log_target.skew())
    #sns.displot(target)

    skewed = df[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
    skewed = skewed[skewed > 0.75]
    skewed = skewed.index
    df[skewed] = np.log1p(df[skewed])
    df[skewed] = np.log1p(df[skewed])

#end adjust_skewed_data

#****************************************************************
# method: adjust_skewed_data(data)
# purpose: replace skewed data with corresponding log() values
#*****************************************************************
def _factorize_category(df, cat_variable, fill_na = None):

    if fill_na is not None:
        df[cat_variable].fillna(fill_na, inplace=True)
    #end if
     
    le.fit(df[cat_variable])
    df[cat_variable] = le.transform(df[cat_variable])

#end adjust_skewed_data


#****************************************************************
# method: hot_encoding(data)
# purpose: hot encode category variables to numeric factors
#*****************************************************************
def _hot_encoding(onehot_df, df, column_name, fill_na):
    
    onehot_df[column_name] = df[column_name]
    
    if fill_na is not None:
        onehot_df[column_name].fillna(fill_na, inplace=True)
    #end if

    dummies = pd.get_dummies(onehot_df[column_name], prefix="_"+column_name)
    onehot_df = onehot_df.join(dummies)
    onehot_df = onehot_df.drop([column_name], axis=1)

    return onehot_df

#end hot_encoding

#****************************************************************
# method: fix_outliers(data)
# purpose: handle outlier data
#*****************************************************************
def _fix_outliers(df, condition):
    
    #removing outliers
    df.drop(df[condition].index, inplace=True)
    df.shape 

#end fix_outliers

#****************************************************************
# method: analyse_category_variable(data)
# purpose: 
#*****************************************************************
def _analyse_category_variable(df):
    print('hello')
#end analyse_category_variable

#****************************************************************
# method: handle_zero_variance(data)
# purpose: analyze and remove zero variance rows
#*****************************************************************
def _handle_zero_variance(df):
    print('hello')
#end handle_zero_variance

#****************************************************************
# method: cateogry_anova(df)
# purpose: ANOVA test is a statistical technique
#          used to determine if there exists a significant 
#          difference in the mean of groups.
#*****************************************************************
def _cateogry_anova(df, cat_variables, target_variable):

    anv = pd.DataFrame()
    anv['features'] = cat_variables
    pvals = []

    for c in cat_variables:
       samples = []
       for cls in df[c].unique():
              s = df[df[c] == cls][target_variable].values
              samples.append(s)
      #end for
              
     #pval = stats.f_oneway(*samples)[1]
     #vals.append(pval)
    #end for

    anv['pval'] = pvals

    return anv.sort_values('pval')

#end cateogry_anova

#****************************************************************
# method: rmse(y_test,y_pred)
# purpose: root mean square error
#*****************************************************************
def rmse(y_test,y_pred):

      return np.sqrt(mean_squared_error(y_test,y_pred))

#end cateogry_anova


#****************************************************************
# method: explore_and_preprocess
# purpose: 
#*****************************************************************
def explore_and_preprocess(df, num_features, cat_features, target_col, row_id_col):

    print('..start exploratory analysis')
    
    print('..check and analyze columns having null values...')
    _analyze_missing_values(df, 'missing_values')
    
    print('fix null values of numeric variables using imputation techniques..')
         
    _numeric_imputation(df[num_features])
    
    print('.. render some graphs analyzing numeric variables...')      
    #_analyze_numeric_variables(df[num_features])        
        
    print('...end exploratory analysis')

#end explore_and_preprocess