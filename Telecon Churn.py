# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:43:50 2021

@author: inf88
"""
import pandas as pd 
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFpr, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor


###################資料前置處理#####################################
#載入資料集
df=pd.read_csv('IBM_sample_data.csv')

#查看資料屬性
df.info()

#檢查是否有遺漏值,重複值
df.isna().describe()
df.duplicated().sum()

#計算 df 中每組的唯一值
u = df.nunique() 

#移除不需要的欄位及遺漏值過多的欄位
df1 = df.drop(['customerID','Churn','TotalCharges'],axis=1)

data = pd.get_dummies(df1,columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod'],drop_first = True)
#再次檢查遺漏值
print(data.isnull().any())

#將Churn提出為y，並處理類別(categorical)變數
y=df[['Churn']]
y=y.replace({"Yes":1,"No":0})

#轉換型態，以便後續分析順利
data = data.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
data.fillna(0, inplace=True)
y.fillna(0, inplace=True)

data.corr()


# 使用 FPR測試 (False Positive Rate) 篩選出 p-value < 0.05 的特徵
selector = SelectFpr(f_regression, alpha=0.05)
data_new = selector.fit_transform(data, y) 
# 檢查哪一個特徵被刪除？
mask = selector.get_support() #list of booleans
new_features = data.columns[mask]
print (new_features)

# 1. 將資料拆成成兩個部分：訓練資料與測試資料。80%是用來訓練電腦, 20%用來測試
# X_train: 訓練資料集(特徵), y_train: 訓練資料集 (label)
# X_test:  測試資料集(特徵), y_test:  測試資料集 (label)
X = data[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'Partner_Yes',
       'Dependents_Yes', 'MultipleLines_Yes', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No internet service',
       'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#2. 利用訓練樣本訓練模型並建立模型 - 本範例使用 LogisticRegression 演算法        
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)  
print(clf.score(X_test,y_test))     
#0.79
                         
# 使用另外一個 迴歸演算法        
rfc = RandomForestRegressor()
rfc.fit(X_train, y_train)
print(rfc.score(X_test,y_test))
#0.18

y_pred = rfc.predict(X_test)
r2=metrics.r2_score(y_test, y_pred)
print('R2 Score：{0:.3f}'.format(r2))
#R2 Score：0.187

# 使用另外一個 迴歸演算法  
from xgboost import XGBClassifier
xgbc = XGBClassifier()

xgbc.fit(X_train, y_train)

print(xgbc.score(X_test,y_test))
#0.78

#混淆矩陣
predictions = xgbc.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report,plot_confusion_matrix

print(classification_report(y_test,predictions ))

from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,predictions)

plot_confusion_matrix(xgbc, X_test, y_test)
predy = xgbc.predict(X_test)
print(classification_report(y_test,predy))