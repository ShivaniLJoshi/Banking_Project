#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[2]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ## Loading and cleaning the data

# In[3]:


data = pd.read_csv('train.csv')

data.drop('Id', axis = 1, inplace = True)
print("Shape of the data is:", data.shape)
data.head()


# ## check numeric and categorical features

# In[4]:


numeric_data = data.select_dtypes(include = np.number) 
numeric_col = numeric_data.columns

print('Numeric Features:')
print('====='*20)
numeric_data.head()


# In[5]:


categorical_data = data.select_dtypes(exclude = np.number)
categorical_col = categorical_data.columns

print('Categorical Features:')
print('====='*20)
categorical_data.head()


# In[6]:


data.dtypes


# ## Checking the missing values

# In[7]:


#To identify the missing values in every feature

total = data.isnull().sum() # gives the values which are missing in the form of True

percent = (total/data.isnull().count()) #total gives total missing values in each feature and count is the total of all true values of the missing values


# In[8]:


total


# In[9]:


percent


# ## Dropping the missing values

# In[10]:


#dropping features having the missing values more than 60%
data = data.drop(percent[percent > 0.6].index, axis = 1)
print(data.isnull().sum())


# In[11]:


#imputing values with mean
for column in numeric_col:
    mean = data[column].mean()
    data[column].fillna(mean, inplace = True)
    
# imputing with median (median can be used when there are outliers in the data)
#for column in numeric_col:
#    median = data[column].median()
#    data[column].fillna(median, inplace = True)


# ## Check for class imbalance

# In[12]:


#class imbalance is only checked on the target variable
#(normalize = True)*100 means in percentage
class_values = (data['y'].value_counts(normalize = True)*100).round(2)

print(class_values)


# In[13]:


sns.countplot(data['y'])


# ### Detect outliers in the continuous columns

# Outliers are the data points defined beyond (third quartile + 1.5*IQR) and below (first quartile -1.5*IQR) in bar plot

# In[14]:


# Illustration of detecting the outliers; percentile function divides the list in ascending order
# and finds the percentile 
num_ls = [1, 2, 3, 4, 5, 6, 7, 10, 100]
# Finding the 25th and 75th percentile:
pc_25 = np.percentile(num_ls, 25) 
pc_75 = np.percentile(num_ls, 75)
#Finding the IQR:
iqr = pc_75 - pc_25

print(f'25th percentile - {pc_25}, 75th percentile - {pc_75}, IQR - {iqr}')

#Calculating outliers:To calculating the outliers we need to set a threshold value beyond which 
#we can consider the values as outliers
upper_threshold = pc_75 + 1.5*iqr
lower_threshold = pc_25 - 1.5*iqr

print(f'Upper - {upper_threshold}, lower - {lower_threshold}')

#plotting the outliers
plt.boxplot(num_ls)# plt.boxplot(num_ls, showfliers = False) gives the plot without outliers


# In[15]:


#Another example with more outliers
num_ls1 = [1, 2, 3, 4, 5, 6, 7, 10, 100, 120, -5, -11 ]
plt.boxplot(num_ls1)


# In[16]:


cols = list(data)# A list of all features

outliers = pd.DataFrame(columns = ['Feature', 'Number of outliers']) # Creating a new dataframe to show the outliers

for column in numeric_col:#Iterating through each feature
        #First quartile (Q1)
    q1 = data[column].quantile(0.25)
    
        #third quartile
    q3 = data[column].quantile(0.75)
    
        #IQR
    iqr = q3 - q1
    
    fence_low = q1 - (1.5*iqr)
    
    fence_high = q3 + (1.5*iqr)
        #finding the number of outliers using 'and(|) condition.
    total_outlier = data[(data[column] < fence_low) | (data[column] < fence_high)].shape[0]
    
    outliers = outliers.append({'Feature': column, 'Number of outliers': total_outlier}, ignore_index = True)
    
outliers


# ## Exploratory data analysis (EDA) and Data Visualization
# Explotatory data analysis is an approach to analyse data sets by summerizing their main characteristics with visualizations. 

# ### Univeriate analysis on categorical variable

# In[17]:


#plotting the frequency of all the values in the categorical variables.

#Selecting the categorical columns
categorical_col = data.select_dtypes(include = ['object']).columns

#plotting a bar chart for each of the categorical variable
for column in categorical_col:
    plt.figure(figsize = (20, 4))
    plt.subplot(121)
    data[column].value_counts().plot(kind = 'bar')
    plt.title(column)
    


# #### Observations
# From the above visuals, we can make the following observations:
# - The top three profession that our customers belong to are - administration, blue-collar jobs and technicians.
# - A huge number of the customers are married.
# - Majority of the customers do not have a credit in default
# - Many of our past customers have applied for a housing loan but very few have applied for a personal loan.
# - Cell-phones seem to the most favoured method of reaching out to the customers. 
# - The plot for the target variable shows heavy imbalance in the target variable.
# - The missing values in some columns have been represented as unknown- Unknown represents missing data. In the next task, we will treat these values.

# ## Univeriate analysis on Continuous/Numeric Columns

# In[18]:


for column in numeric_col:
    plt.figure(figsize = (20, 4))
    plt.subplot(121)
    plt.hist(data[column])
    plt.title(column)


# #### Imputing unknown values of categorical columns

# One method of imputing unknown values is to directly impute them with the mode value of respective columns.

# In[19]:


#Impute missing values of categorical columns
for column in categorical_col:
    mode = (data[column]).mode()[0]
    data[column] = data[column].replace('unknown', mode)


# In[20]:


data['job'].value_counts()


# In[21]:


data['job'].mode()


# In[22]:


for column in numeric_col:
    plt.figure(figsize = (20, 4))
    plt.subplot(121)
    plt.boxplot(data[column])
    plt.title(column)


# #### Observation :
# 1. As we see from the histogram, the features age, duration and campaign are heavily skewed and this is due to the presence of outliers as seen on the boxplot for these features.
# 2. Looking at the plot for pdays, we can infer that majority of the customers were being contacted for the first time because as per the feature descriptionf for pdays and previous consist majority only of single value, their variance is quite less and hence we can drop them since technically will be of no help in prediction.

# In[23]:


data['pdays'].value_counts(normalize = True)


# In[24]:


data['previous'].value_counts(normalize = True)


# ### Dropping the columns pdays and previous

# In[25]:


data.drop(['pdays', 'previous'], 1, inplace = True)#here 1 is the axis i.e the column


# ### Bivartate Analysis - Categorical Columns

# In[26]:


for column in categorical_col:
    plt.figure(figsize = (20,4))
    plt.subplot(121)
    sns.countplot(x = data[column], hue = data['y'], data = data )
    plt.title(column)
    plt.xticks(rotation = 90)


# ### Observation:
# The common traits seen for the customers who subscribed for term depostit are-
#    - Customers having administrative jobs from the majority amongst those who have subscribed to the term deposite with technicians being the second majority.
#    - They are married.
#    - They hold a university degree. 
#    - They dont hold a credit in default.
#    - Housing loan doesn't seem a priority to check for since an equal number of customers who have not subscribed to it seem to have subscribed to the term deposit.
#    - Cell-phones should be the prefered mode of contact for contacting customers.

# ### Treating the outliers in continuous column

# upper threshold = 25
# lower threshold = 10
# 
# > (>25 and <10) = outlier
# 
# if x < lower threshold -> replace with 10
# if x > upper threshold -> replace with 25
# 
# #### This process is called as "winsorization". In this method we define a confidence interval of let's say 90% and then repalce all the outliers below the 5th percentile with the value above 95th percentile with the value of 95th percentile.

# ### Example of using winsorization:
# np.percentile([1, 2, 3, 4, 5, 6, 7, 10, 100], 95)
# >output: 63.99999999999997
# 
# np.percentile([1, 2, 3, 4, 5, 6, 7, 10, 100], 5)
# >output: 1.4
# 
# Any value above 63.99 will be replaced by 63.99
# and any value below 1.4 will be replaced by 1.4

# In[27]:


from scipy.stats.mstats import winsorize


# In[28]:


numeric_col = data.select_dtypes(include = np.number).columns

for col in numeric_col:
    data[col] = winsorize(data[col], limits = [0.05, 0.1], inclusive = (True, True))


# In[29]:


numeric_col


# #### Machine learning models do not accept string as input hence we convert the categories into numbers this process is called as "Label Encoding"
# #### And the process in which we create seperate column for each category of the categorical variable is called as "One-hot Encoding"

# In[30]:


#Illustration of label encoding:
data1 = data.copy()


# In[31]:


marital_dict = {'married': 0, 'divorced': 1, 'single': 2}


# In[32]:


data1['marital_num'] = data1.marital.map(marital_dict)


# In[33]:


data1[['marital', 'marital_num']]


# In[34]:


#Illustration of One-hot Encoding:

pd.get_dummies(data1, columns = ['marital'])


# In[35]:


data1.drop('marital_num', axis = 1, inplace = True)


# In[36]:


data1


# ## Applying vanilla models on the data
# 
# Since we have prerformed preprocessing on our data and also done with the EDA part, it is now time to apply vanilla machinelearning models on the data and check their performance
# 
# ### Function to label encode categorical variables
# 
# Before applying ml algo, we need to recollect that any algo can only read numerical values. It is therefore, essential to encode categorical features into numerical values. Encoding of categorical variables can be performed in two ways:
# 
# - Label Encoding
# - One-Hot Encoding

# In[37]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


# In[38]:


#Initialize the label encoder
le = LabelEncoder()

#Iterating through each of the categorical columns and label encoding them
for feature in categorical_col:
    try:
        data[feature] = le.fit_transform(data[feature])
    except:
        print('Error encoding'+feature)


# In[39]:


data


# ### Fit vanilla classifier models
# 
# There are many classifier algorithms:
# - Logistic Regression
# - DecisionTree Classifier
# - RandomForest Classifier
# - XGBClassifier
# - GradientBoostingClassifier
# 
# The code below splits the data into training data and validation data. It then fits the classification model on the train data and then makes a prediction on the validation data and outputs the roc_auc_score and roc_curve for this prediction.

# #### Preparing the train and test data

# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[41]:


#Predictors
X = data.iloc[:,:-1]

#Target 
y = data.iloc[:,-1]

#Dividing the data into train and test subsets
x_train,x_val,y_train,y_val = train_test_split(X, y, test_size = 0.2, random_state= 42)


# #### FITTING THE MODEL AND PREDICTING THE VALUES

# In[42]:


#Run Logistic Regression model
model = LogisticRegression()

#fitting the model
model.fit(x_train, y_train)

#predicting the values
y_scores = model.predict(x_val)


# #### GETTING THE METRICS TO CHECK OUR MODEL PERFORMANCE

# In[43]:


from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, classification_report, roc_curve, confusion_matrix


# In[44]:


#GETTING THE AUC ROC CURVE
auc = roc_auc_score(y_val, y_scores)
print('Classification Report:')
print(classification_report(y_val, y_scores))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is', roc_auc_score(y_val, y_scores))

#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])

plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()


# In[45]:


from sklearn.tree import DecisionTreeClassifier
#Run on Decision Tree Classifier
model = DecisionTreeClassifier()

model.fit(x_train, y_train)

y_score = model.predict(x_val)

auc = roc_auc_score(y_val, y_score)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)

print('ROC_AUC_SCORE is', roc_auc_score(y_val, y_scores))

#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])

plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()


# In[46]:


#Run random Forest classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(x_train, y_train)

y_scores = model.predict(x_val)

auc = roc_auc_score(y_val, y_scores)
print('Classification Report:')
print(classification_report(y_val, y_scores))
false_positive_rate, true_positive_rate, threshold = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is', roc_auc_score(y_val, y_scores))
#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])

plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[47]:


#Run on XGBoost model
from xgboost import XGBClassifier 
model = XGBClassifier()

model.fit(x_train, y_train)

y_scores = model.predict(x_val)

auc = roc_auc_score(y_val, y_scores)
print('Classification Report is:')
print(classification_report(y_val, y_scores))

false_positive_rate, true_positive_rate, threshold = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))

plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[48]:


#Run Gradient Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier 
model = GradientBoostingClassifier()

model.fit(x_train, y_train)

y_scores = model.predict(x_val)

auc = roc_auc_score(y_val, y_scores)
print('Classification Report is:')
print(classification_report(y_val, y_scores))

false_positive_rate, true_positive_rate, threshold = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))

plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# ## Feature Selection
# Now that we have applied vanilla models on our data we now have a basic understanding of what our predictions look like. Let's now use feature selection methods for identifying the best of features for each model.
# 
# ## Using RFE(Recursive Feature Elimination ) for feature selection
# In this task let's use Recursive Feature Elimination for selecting the best features. RFE is a wrapper method that uses the model to identify the best features.
# 
# - For the below task, we have inputted a feature. We can change the value and input the number of features you want to retain for your model

# In[49]:


from sklearn.feature_selection import RFE
#Selecting 8 number of features
#Selecting models

models = LogisticRegression()

#using rfe and selecting 8 features

rfe = RFE(models, 8)

#fitting the model

rfe = rfe.fit(X, y)

#Ranking features

feature_ranking = pd.Series(rfe.ranking_, index = X.columns)
plt.show()
print('Features to be selected for Logistic Regression model are:')
print(feature_ranking[feature_ranking.values == 1].index.tolist())
print('===='*30)


# In[50]:


#Selecting 8 number of features
#Random Forest Classifier model

models = RandomForestClassifier()

#using rfe and selecting 8 features
rfe = RFE(models, 8)

rfe = rfe.fit(X, y)

feature_ranking = pd.Series(rfe.ranking_, index = X.columns)
plt.show()
print('Features to be selected for Logistic Regression model are:')
print(feature_ranking[feature_ranking.values == 1].index.tolist())
print('===='*30)


# In[51]:


#Selecting 8 number of features
#XGBoost Classifier model

models = XGBClassifier()

#using rfe and selecting 8 features
rfe = RFE(models, 8)

rfe = rfe.fit(X, y)

feature_ranking = pd.Series(rfe.ranking_, index = X.columns)
plt.show()
print('Features to be selected for Logistic Regression model are:')
print(feature_ranking[feature_ranking.values == 1].index.tolist())
print('===='*30)


# ### Feature Selection using Random Forest
# Random Forests are often used for feature selection in a data science workflow. This is because the tree based stratergies that random forests use, rank the features based on how well they improve the purity of the node. This nodes having a very low impurity get split at the start of the tree while the nodes having a very high impurity get split towards the end of the tree. Hence, by pruning the tree after desired amount of splits, we can create a subset of the most important features. 

# In[52]:


#Splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

#Selecting the data
rfc = RandomForestClassifier(random_state = 42)

#Fitting the data
rfc.fit(X_train, y_train)

#Predicting the data 
y_pred = rfc.predict(X_test)

#Feature importances
rfc_importances = pd.Series(rfc.feature_importances_, index = X.columns).sort_values().tail(10)

#plotting bar chart according to feature importance
rfc_importances.plot(kind = 'bar')

plt.show()


# #### Observation:
# We can test the features obtained from both the selection techniques by inserting the model and depending on which set of features perform better, we can retain them from the model.
# 
# The feature selection techniques can differ from problem to problem and the techniques applied for an algorithm may or may not work for the other problems. In those cases, feel free to try out other methods like PCA, SelectKBest(), SelectPercentile(), ISNE etc.

# ## Grid - Search and Hyperparameter Tuning
# Hyperparameters are function attributes that we have to specify for an algorithm. By now, you should be knowing that grid search is done to find out the best set of hyperparameters for your model.
# 
# #### Grid Search for Random- Forest
# In the below task, we write a code that performs hyperparameter tuning for a random forest classifier. We have used the hyperparameters max_features, max_depth and criterion for this task. Feel free to play around with this function by introducing a few more hyperparameters and changing their values.

# In[53]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#splitting the data
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

#selecting the classifier
rfc = RandomForestClassifier()

#Selecting the parameters
param_grid = {
    'max_features':['auto', 'sqrt', 'log2'],
    'max_depth':[4, 5, 6, 7, 8],
    'criterion':['gini', 'entropy']
}

#using grid search with respective parameters
grid_search_model = GridSearchCV(rfc, param_grid = param_grid)

#fitting the model
grid_search_model.fit(x_train, y_train)

#printing the best parameters
print('Best parameters are:', grid_search_model.best_params_)


# ### Applying the best parameters obtained using Grid Search on Random Forest model
# In the task below, we fit a random forest model using the best parameters obtained using Grid Search. Since the target is imbalanced, we apply Synthetic Minority Oversampling (SMOTE) for undersampling and oversampling the majority and minority classes in the target respectively.
# 
# #### Kindly note that SMOTE should always be applied only on the training data and not on the validation and test data.

# In[58]:


from sklearn.metrics import roc_auc_score,roc_curve,classification_report
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Ridge,Lasso
from yellowbrick.classifier import roc_auc


def grid_search_random_forrest_best(dataframe,target):
    
    
    x_train,x_val,y_train,y_val = train_test_split(dataframe,target, test_size=0.3, random_state=42)
    
    # Applying Smote on train data for dealing with class imbalance
    smote = SMOTE(random_state = 42)
    X_sm, y_sm =  smote.fit_sample(x_train, y_train)
    
    
    rfc = RandomForestClassifier(n_estimators=11, max_features='auto', max_depth=8, criterion='entropy',random_state=42)
    rfc.fit(X_sm, y_sm)
    y_pred = rfc.predict(x_val)

    
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    visualizer = roc_auc(rfc, X_sm, y_sm, x_val, y_val)

grid_search_random_forrest_best(X,y)


# ### Applying the grid search function for random forest only on the best features obtained using RFE

# In[64]:


#Random Forest Classifier

grid_search_random_forrest_best(X[['age', 'job', 'education', 'day_of_week', 'duration', 'campaign', 'euribor3m', 'nr.employed']], y)


# ### Applying the grid search function for Logistic Regression

# In[67]:


def grid_search_log_reg(dataframe,target):
    
    
    x_train,x_val,y_train,y_val = train_test_split(dataframe, target, test_size=0.3, random_state=42)

    smote = SMOTE(random_state = 42)
    X_sm, y_sm =  smote.fit_sample(x_train, y_train)
    
    
    log_reg = LogisticRegression()
    
    param_grid = { 
        'C' : np.logspace(-5, 8, 15)
    }
    grid_search = GridSearchCV(log_reg, param_grid=param_grid)
    
    grid_search.fit(X_sm, y_sm)
    y_pred = grid_search.predict(x_val)
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    visualizer = roc_auc(rfc, X_sm, y_sm, x_val, y_val)
    
grid_search_log_reg(X, y)


# ### Applying the grid search function for XGBoost

# In[69]:


def xgboost(dataframe,target):
    #X = dataframe
    #y = target

    x_train,x_val,y_train,y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    smote = SMOTE(random_state = 42)
    X_sm, y_sm =  smote.fit_sample(x_train, y_train)

    model = XGBClassifier(n_estimators=50, max_depth=4)
    model.fit(pd.DataFrame(X_sm,columns=x_train.columns), y_sm)
    y_pred = model.predict(x_val)
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    visualizer = roc_auc(rfc, X_sm, y_sm, x_val, y_val)
    
xgboost(X, y)


# ### Ensembling
# Ensemble learning uses multiple machine learning models to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. In the below task, we have used an ensemble of three models - RandomForestClassifier(), GradientBoostingClassifier(), LogisticRegression().

# In[73]:


from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier

#splitting the data
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)

#using smote
smote = SMOTE(random_state = 42)
X_sm, y_sm = smote.fit_sample(x_train, y_train)

model1 = RandomForestClassifier()
model2 = GradientBoostingClassifier()
model3 = LogisticRegression()

#Fitting the model
model = VotingClassifier(estimators = [('rf', model1), ('lr', model2), ('xgb', model3)], voting = 'soft')
model.fit(X_sm, y_sm)

#Predicting values and getting the metrics
y_pred = model.predict(x_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
visualizer = roc_auc(model, X_sm, y_sm, x_val, y_val)


# ### Prediction on the test data

# In[74]:


# Actual Test File
test = pd.read_csv('test.csv')

# Storing the Id column
Id = test[['Id']]

# Preprocessed Test File
test = pd.read_csv('test_preprocessed.csv')
test.drop('Id',1,inplace=True)
test.head()


# In[77]:


def grid_search_log_reg(dataframe,target):


    x_train,x_val,y_train,y_val = train_test_split(dataframe, target, test_size=0.3, random_state=42)

    smote = SMOTE(random_state = 42)
    X_sm, y_sm =  smote.fit_sample(x_train, y_train)


    log_reg = LogisticRegression()

    param_grid = { 
        'C' : np.logspace(-5, 8, 15)
    }
    grid_search = GridSearchCV(log_reg, param_grid=param_grid)

    grid_search.fit(X_sm, y_sm)
    
    # Predict on the preprocessed test file
    y_pred = grid_search.predict(test)
    return y_pred

    
prediction = pd.DataFrame(grid_search_log_reg(X,y),columns=['y'])
submission = pd.concat([Id,prediction['y']],1)

submission.to_csv('submission.csv',index=False)


# In[ ]:




