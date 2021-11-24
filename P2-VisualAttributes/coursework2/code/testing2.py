
import pandas as pd
import numpy as numpy
import pandas as panda
from pandas import DataFrame
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from sklearn.pipeline import Pipeline
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer

# read data
# step1 data Preprocessing
# scikit-learn Preprocessing staring

t_train = pd.read_csv('./data/data_train.csv')
print(t_train.groupby(['color']).size())

'''fullfill the null data'''
t_train.fillna(t_train.mean(), inplace=True)


y_train_output_texture = t_train['texture']  # output values
y_train_output_color = t_train['color']  # output values
t_train_real = t_train.drop(['id', 'image', 'color', 'texture'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(t_train_real, y_train_output_texture, test_size=0.25,
                                                    stratify=y_train_output_texture, random_state=1)

# input features

# print(t_train_real.dtypes)

# Build a transformer for numeric features
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Build a one-hot encoding transformer for categorical features
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

selector_num_transformer = list(X_train.select_dtypes('float64', 'int64').columns)
print(f"Categorical columns are: {selector_num_transformer}")
selector_cat_transformer = list(X_train.select_dtypes('object').columns)
print(f"Numerical columns are: {selector_cat_transformer}")

# Combine both transformers into one

preprocessor = ColumnTransformer(transformers=[('num', num_transformer, selector_num_transformer),
                                               ('cat', cat_transformer, selector_cat_transformer)])



''' ###################          classification      ###################        '''

model = RandomForestClassifier(
n_estimators=7000, class_weight='balanced',
oob_score=True,
n_jobs=-1,
verbose=1,
max_features=10,
max_depth=20,
)



#  scikit-learn: Training : Combine preprocessors and a neural network into a Pipeline:
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
# deal with unsampling data



'''   #######################################################################################  
     ################################   getting texture result     ##########################  
      #######################################################################################  
'''

''' ##########             over sampling      ##########        '''
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

'''     ####################         trainging      ####################   '''
clf.fit(X_resampled, y_resampled)

#clf.fit(X_train, y_train)
# get the predicted probability values
# example output: [0.9825, 0.0175]
h = clf.predict_proba(t_train_real)[0]


# calculate the predictions
y_pred = clf.predict(X_test)
# print accuracy
from sklearn.metrics import accuracy_score

'''  ###############################  evalution ###############################  '''

#report
print(metrics.classification_report(y_test,y_pred, zero_division=0))
print("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
print(y_pred)


'''   ###############################      getting real testing data    ##########################       '''

t_realtesting = pd.read_csv('./data/data_test.csv')
t_realtesting .fillna(t_train.mean(), inplace=True)
t_realtesting = t_realtesting .drop(['id', 'image'], axis=1)
y_pred_real = clf.predict(t_realtesting)

y_pred_df=pd.DataFrame(y_pred_real)

'''   ###############################       save texture result     ##########################       '''
y_pred_df.to_csv('texture_test.csv',  header=False,index=False)
print(y_pred_df)




'''   #######################################################################################  
     ################################   getting color result     ##########################  
      #######################################################################################  
'''


X_train, X_test, y_train, y_test = train_test_split(t_train_real, y_train_output_color, test_size=0.25,
                                                    stratify=y_train_output_color, random_state=1)
clf2 = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])


''' ##########             over sampling      ##########        '''
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

clf2.fit(X_resampled, y_resampled)

y_pred = clf2.predict(X_test)

'''  ###############################  evalution ###############################  '''

#report
print(metrics.classification_report(y_test,y_pred, zero_division=0))
print("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))



'''   ###############################      getting real testing data    ##########################       '''

t_realtesting = pd.read_csv('./data/data_test.csv')
t_realtesting .fillna(t_train.mean(), inplace=True)
t_realtesting = t_realtesting .drop(['id', 'image'], axis=1)
y_pred_real = clf.predict(t_realtesting)

y_pred_df=pd.DataFrame(y_pred_real)

'''   ###############################       save color result     ##########################       '''
y_pred_df.to_csv('color_test.csv',  header=False,index=False)
print(y_pred_df)

