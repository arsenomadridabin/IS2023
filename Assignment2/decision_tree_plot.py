import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from sklearn.tree import plot_tree


# Function to perform training with giniIndex.
def train_using_gini(X_train, Y_train):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=4, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, Y_train)
    return clf_gini

# Function to split the dataset
def splitdataset(balance_data):
  
    # Separating the target variable
    X = balance_data.values[:, 2:-2]
    Y = balance_data.values[:, -2]
  
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
      
    return X, Y, X_train, X_test, y_train, y_test


def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))

df = pd.read_csv('Kaggle-data.csv')
#Remove inconsistent data
df = df.loc[df['Machine'] != "3ab1aa9785d0681434766bb0ffc4a13c"]
for column in df.columns:
  if df[column].isnull().values.any():
    if column == 'MajorLinkerVersion':
      df2 = df.loc[~df[column].isnull()]

X, Y, X_train, X_test, Y_train, y_test = splitdataset(df2)

print("shap=",y_test.shape)

index = []
Y_train_str = []
for i in Y_train:
  Y_train_str.append(str(i))

Y_train = np.array(Y_train_str)
gini_val = train_using_gini(X_train,Y_train )


pd.set_option('display.max_columns', None)
df3 = df2.iloc[:,2:-2]
columns = df3.columns
feat = gini_val.feature_importances_

sorted_features = np.argsort(feat)[::-1]
sorted_val = sorted(feat,reverse=True)
zipped_val = zip(sorted_features,sorted_val)

for feature_index, feature_score in zipped_val:
  print(columns[feature_index],feature_score)


columns_list = list(columns)
len(columns_list)
plt.figure(figsize=(60,60))
plot_tree(gini_val, max_depth=None, feature_names=columns_list)