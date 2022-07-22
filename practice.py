import pandas as pd
df=pd.read_csv('creditcard.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df['Class'].value_counts())
#print(df.info)
# Train & Test - StratifiedShuffleSplit for imbalance Data set. according to 'Class' feature.
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(df,df['Class']):
    strat_train_set=df.loc[train_index]
    strat_test_set=df.loc[test_index]
print("Train values :\n ",strat_train_set['Class'].value_counts())
print("Test values : \n",strat_test_set['Class'].value_counts())
x_train = strat_train_set.drop("Class",axis=1)
y_train = strat_train_set["Class"].copy()
x_test = strat_test_set.drop("Class",axis=1)
y_test = strat_test_set["Class"].copy()

from imblearn.combine import SMOTETomek
os=SMOTETomek(random_state=42)
x_train_ns,y_train_ns=os.fit_resample(x_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))
# Independent & dependent feature (Input & Output)
print(x_train.head())
print(y_train.head())
class_weight=dict({0:1,1:100})

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from collections import Counter
#print(Counter(y_train))
classifier=RandomForestClassifier(class_weight=class_weight)

#classifier=RandomForestClassifier()
#classifier.fit(x_train,y_train)
classifier.fit(x_train_ns,y_train_ns)
y_pred=classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))