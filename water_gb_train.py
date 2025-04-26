import pandas as pd
import numpy as np
import pickle
import mlflow
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import mlflow.sklearn




import dagshub
dagshub.init(repo_owner='RajeshB-0699', repo_name='MLOps_Exp_Dagshub', mlflow=True)

mlflow.set_experiment('gbc_water')
mlflow.set_tracking_uri('https://dagshub.com/RajeshB-0699/MLOps_Exp_Dagshub.mlflow')




data = pd.read_csv('D:\exp-mlflow\data\water_potability.csv')

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df


train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

from sklearn.ensemble import GradientBoostingClassifier

X_train = train_processed_data.drop(columns = ['Potability'], axis=1)
y_train = train_processed_data['Potability']

n_estimators = 800

with mlflow.start_run():
    clf = GradientBoostingClassifier(n_estimators = n_estimators)

    clf.fit(X_train, y_train)

    pickle.dump(clf,open("model.pkl","wb"))

    X_test = test_processed_data.drop(columns = ['Potability'], axis = 1)
    y_test = test_processed_data['Potability']

    from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

    model = pickle.load(open('model.pkl','rb'))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    mlflow.log_metric("acc",acc)
    mlflow.log_metric("prec",prec)
    mlflow.log_metric("f1",f1)
    mlflow.log_metric("recall",recall)

    

    mlflow.log_param("n_estimators", n_estimators)

    
    cm = confusion_matrix(y_pred, y_test)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot = True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(clf,"GradientBoostingClassifier")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("author","Rajesh B")
    mlflow.set_tag("model", "GB")

    print("acc",acc)
    print("prec",prec)
    print("f1",f1)
    print("recall",recall)

