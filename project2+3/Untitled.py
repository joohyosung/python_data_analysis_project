#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Gulim'

member = pd.read_csv('유아용품데이터/Member_data02.csv')
product = pd.read_csv('유아용품데이터/Product_data.csv')
sales = pd.read_csv('유아용품데이터/Sales_data02.csv')


# In[3]:


sales.rename(columns = {'고객번호' : 'ID'}, inplace = True)
sales['ID'] = sales['ID'].astype('float')


# In[5]:


sm_merge = sales.merge(member, on = 'ID', how='inner')

data = sm_merge.drop(['주문번호'],axis=1)
data['구매일'] = pd.to_datetime(data['구매일'], format = '%Y-%m-%d')
data['구매월'] = data['구매일'].dt.month


# In[6]:


def age(x):
    if x > 100:
        return round(x / 30,0)
    else:
        return x
data['구매시월령(수정2)'] = data['구매시월령(수정)'].apply(lambda x: age(x))

data['구매시월령(수정2)'].max()


# In[7]:


import math

data['구매시월령(수정2)'] = data['구매시월령(수정2)'].fillna(
    math.floor(data[(data['연령'] >= 30) & (data['연령'] <= 36)]['구매시월령(수정2)'].mean()))
data.head()


# In[8]:


data['상품명'] = data['상품명'].str.replace('？', ' ')


# In[9]:


data['구매년월'] = data['구매일'].dt.strftime('%Y-%m')
data['구매년월'] = pd.to_datetime(data['구매년월'])


# In[11]:


def month_div(df, start_date, end_date):
    select_data = df[(df.loc[:,'구매일'] >= start_date ) & (df.loc[:,'구매일'] <= end_date)]
    grouped_select_data = select_data.groupby('ID').agg(sum_purchase = ('구매금액','sum'))
    return grouped_select_data


# In[12]:


user_grade = month_div(data, '2019-01', '2020-08')


# In[13]:


def grade(x):
    if x > 100000:
        result = 1
    elif x > 70000:
        result = 2
    elif x > 50000:
        result = 3
    else:
        result = 0
    return result


# In[15]:


user_grade['등급'] = user_grade['sum_purchase'].apply(grade)
grade_merge = data.merge(user_grade, on = 'ID', how='outer')

grade_merge['등급'] = grade_merge['등급'].fillna('0')
grade_merge = grade_merge.drop(['sum_purchase'], axis=1)


# In[19]:


grade_merge['등급'] = grade_merge['등급'].astype('int')
grade_merge['할인가격'] = data['구매금액'] - data['결제금액']
grade_merge['구매시월령'] = data['구매시월령(수정2)']


# # 등급에 관한 모델 적용

# ## 레그레이션

# In[21]:


# 레그레이션 모델 적용

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

target = ['구매시월령','구매금액','결혼유무','연령','등급']

dataset = grade_merge[target]
dataset = pd.get_dummies(dataset)

model = LinearRegression()

X = dataset.drop('등급', axis = 1)
Y = dataset['등급']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)

model = LinearRegression()
model.fit(X_train, Y_train)

model.score(X_test, Y_test)


# ## 의사결정나무

# In[22]:


# 의사결정나무

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = dataset.drop('등급', axis = 1)
Y = dataset['등급']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

model = DecisionTreeClassifier(max_depth = 8)
model.fit(X_train, Y_train)

train_accuracy = model.score(X_train, Y_train)
test_accuracy = model.score(X_test, Y_test)
print(f'훈련 정확도 : {train_accuracy}')
print(f'테스트 정확도 : {test_accuracy}')


# In[23]:


train_accuracy - test_accuracy


# In[24]:


train_score=[]
test_score=[]
for i in range(1,21):
    model=DecisionTreeClassifier(max_depth=i,random_state=0)
    model.fit(X_train,Y_train)
    train_score.append(model.score(X_train,Y_train))
    test_score.append(model.score(X_test,Y_test))
plt.figure()
plt.title('score for depths')
plt.plot(range(1,21),train_score)
plt.plot(range(1,21),test_score)
plt.xticks(range(1,21))
plt.show()


# In[25]:


importance = pd.DataFrame({'feature_names':X.columns, "특성 중요도":
                          model.feature_importances_})
# importance.sort_values(by = "특성 중요도",ascending = False)

importance[importance['특성 중요도']!=0.00].sort_values(by = "특성 중요도",ascending = False)


# In[26]:


# 각 모델의 특성 중요도 시각화 (내림차순되어있지 않음)
import numpy as np
def plot_feature_importances_(model):
    n_features = X.shape[1]
    plt.figure(figsize=(10,5))
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_(model)


# In[27]:


구매금액 = 20000
구매시월령 = 10
연령 = 33
결혼유무_미혼 = 0
결혼유무_기혼 = 0

input_data = [구매금액,연령,구매시월령,결혼유무_미혼,결혼유무_기혼]


# In[28]:


print(model.predict([input_data]))
print(model.predict_proba([input_data]))

pred_dt = model.predict(X_test)


# In[29]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(Y_test, pred_dt)
print(conf_matrix)


# In[30]:


from sklearn.metrics import classification_report
class_report = classification_report(Y_test, pred_dt)
print(class_report)

# Accuracy (정확도)
# Confusion Matrix (오차행렬)
# Precision (정밀도)
# Recall (재현율)
# F1 Score (F1스코어)


# ## 랜덤 포레스트 적용

# In[33]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

X = dataset.drop('등급', axis = 1)
Y = dataset['등급'].ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

scaler = StandardScaler()
scaler.fit(X_train)

x_train_std = scaler.transform(X_train)
x_test_std = scaler.transform(X_test)


# In[34]:


rf_clf = RandomForestClassifier(max_depth = 7.2,
                               n_estimators=120)

rf_clf.fit(x_train_std, Y_train)

rf_train_score = rf_clf.score(x_train_std, Y_train)
rf_test_score = rf_clf.score(x_test_std, Y_test)
print(f'랜덤포레스트 훈련 정확도는 {round(rf_train_score,3)} 입니다.')
print(f'랜덤포레스트 테스트 정확도는 {round(rf_test_score,3)} 입니다.')


# In[35]:


round(rf_train_score,3) - round(rf_test_score,3)


# In[36]:


train_score=[]
test_score=[]
for i in range(1,10):
    model=RandomForestClassifier(max_depth=i,random_state=0)
    model.fit(X_train,Y_train)
    train_score.append(model.score(X_train,Y_train))
    test_score.append(model.score(X_test,Y_test))
plt.figure()
plt.title('score for depths')
plt.plot(range(1,10),train_score)
plt.plot(range(1,10),test_score)
plt.xticks(range(1,10))
plt.show()


# In[37]:


importance = pd.DataFrame({'feature_names':X.columns, "특성 중요도":
                          model.feature_importances_})
# importance.sort_values(by = "특성 중요도",ascending = False)

importance[importance['특성 중요도']!=0.00].sort_values(by = "특성 중요도",ascending = False)


# In[38]:


# 각 모델의 특성 중요도 시각화 (내림차순되어있지 않음)
import numpy as np
def plot_feature_importances_(model):
    n_features = X.shape[1]
    plt.figure(figsize=(10,5))
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_(model)


# In[39]:


pred_rf = rf_clf.predict(x_test_std)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(Y_test, pred_rf)
print(conf_matrix)


# In[40]:


from sklearn.metrics import classification_report
class_report = classification_report(Y_test, pred_rf)
print(class_report)


# ## 그레디언트 부스팅 적용

# In[41]:


from sklearn.ensemble import GradientBoostingClassifier


gbrt = GradientBoostingClassifier(max_depth = 2, learning_rate = 0.11)
gbrt.fit(x_train_std, Y_train)

gbrt_train_score = gbrt.score(x_train_std, Y_train)
gbrt_test_score = gbrt.score(x_test_std, Y_test)
print(f'그레디언트부스팅 훈련 정확도는 {round(gbrt_train_score,2)} 입니다.')
print(f'그레디언트부스팅 테스트 정확도는 {round(gbrt_test_score,2)} 입니다.')


# In[ ]:


train_score=[]
test_score=[]
for i in range(1,10):
    model=GradientBoostingClassifier(max_depth=i,random_state=0)
    model.fit(X_train,Y_train)
    train_score.append(model.score(X_train,Y_train))
    test_score.append(model.score(X_test,Y_test))
plt.figure()
plt.title('score for depths')
plt.plot(range(1,10),train_score)
plt.plot(range(1,10),test_score)
plt.xticks(range(1,10))
plt.show()


# In[ ]:


importance = pd.DataFrame({'feature_names':X.columns, "특성 중요도":
                          model.feature_importances_})
# importance.sort_values(by = "특성 중요도",ascending = False)

importance[importance['특성 중요도']!=0.00].sort_values(by = "특성 중요도",ascending = False)


# In[ ]:


# 각 모델의 특성 중요도 시각화 (내림차순되어있지 않음)
import numpy as np
def plot_feature_importances_(model):
    n_features = X.shape[1]
    plt.figure(figsize=(10,5))
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_(model)


# In[ ]:


pred_gb = gbrt.predict(x_test_std)

conf_matrix = confusion_matrix(Y_test, pred_gb)
print(conf_matrix)


# In[ ]:


class_report = classification_report(Y_test, pred_gb)
print(class_report)


# # 3개월마다 구매 예측하기

# In[ ]:


user_purchas = pd.pivot_table(data = grade_merge, index = 'ID', columns = '구매년월', values = '구매금액', aggfunc='count', fill_value=0)
user_purchas[user_purchas.values > 1]


# In[ ]:


total_column_count = len(user_purchas.columns)    # 20
set_column_count = 3
result_df = pd.DataFrame()

for i in range(total_column_count - set_column_count):
    selected_df = user_purchas.iloc[:,i:i+4]
    print(selected_df.columns)
    selected_df.columns = ['3달전','2달전','1달전','구매여부']
    result_df = pd.concat([result_df, selected_df])


# In[ ]:


predict_data = pd.merge(result_df.reset_index(),
                        grade_merge['ID'],
                        on = 'ID')
predict_data.head()


# ## 레그리언트

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

model = LinearRegression()

X = predict_data.drop(['구매여부','ID'], axis = 1)
Y = predict_data['구매여부']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)

model = LinearRegression()
model.fit(X_train, Y_train)

model.score(X_test, Y_test)


# ## 의사결정나무

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = predict_data.drop('구매여부', axis = 1)
Y = predict_data['구매여부']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 100)

model = DecisionTreeClassifier(max_depth = 8)
model.fit(X_train, Y_train)

train_accuracy = model.score(X_train, Y_train)
test_accuracy = model.score(X_test, Y_test)
print(f'훈련 정확도 : {train_accuracy}')
print(f'테스트 정확도 : {test_accuracy}')


# ### 랜덤 포레스트

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


X = predict_data.drop('구매여부', axis = 1)
Y = predict_data['구매여부']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 50)

scaler = StandardScaler()
scaler.fit(X_train)

x_train_std = scaler.transform(X_train)
x_test_std = scaler.transform(X_test)

model = RandomForestClassifier(max_depth = 4,
                               n_estimators=150)
model.fit(x_train_std, Y_train)

# model = RandomForestClassifier(max_depth = 2)
# model.fit(X_train, Y_train)

train_accuracy = model.score(x_train_std, Y_train)
test_accuracy = model.score(x_test_std, Y_test)

print(f'훈련 정확도 : {train_accuracy}')
print(f'테스트 정확도 : {test_accuracy}')


# In[ ]:


round(train_accuracy,3) - round(test_accuracy,3)


# In[ ]:


train_score=[]
test_score=[]
for i in range(1,10):
    model=RandomForestClassifier(max_depth=i,random_state=0)
    model.fit(x_train_std,Y_train)
    train_score.append(model.score(x_train_std,Y_train))
    test_score.append(model.score(x_test_std,Y_test))
plt.figure()
plt.title('score for depths')
plt.plot(range(1,10),train_score)
plt.plot(range(1,10),test_score)
plt.xticks(range(1,10))
plt.show()


# In[ ]:


importance = pd.DataFrame({'feature_names':X.columns, "특성 중요도":
                          model.feature_importances_})
# importance.sort_values(by = "특성 중요도",ascending = False)

importance[importance['특성 중요도']!=0.00].sort_values(by = "특성 중요도",ascending = False)


# In[ ]:


# 각 모델의 특성 중요도 시각화 (내림차순되어있지 않음)

import numpy as np
def plot_feature_importances_(model):
    n_features = X.shape[1]
    plt.figure(figsize=(10,5))
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_(model)


# # K-군집화

# In[ ]:


user_grade = month_div(data, '2019-01', '2020-01')

grade_merge['등급'].unique()
grade_merge['등급'].astype('int')
grade_merge.columns
aa = grade_merge.groupby('ID').agg(count = ('구매금액','count'))
aa
grade_merge2 = pd.merge(grade_merge,aa,on='ID',how='inner')
grade_merge2['구매횟수'] = grade_merge2['count']
grade_merge2


# In[ ]:


# 군집화
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

clustering = grade_merge2[['구매횟수','구매금액']]
scaler = MinMaxScaler()
data_scale = scaler.fit_transform(clustering)
k = 3
model = KMeans(n_clusters = k, random_state = 10)
model.fit(data_scale)
grade_merge2['cluster'] = model.fit_predict(data_scale)
plt.figure(figsize = (8, 8))
for i in range(k):
    plt.scatter(grade_merge2.loc[grade_merge2['cluster'] == i, '구매횟수'],
                grade_merge2.loc[grade_merge2['cluster'] == i, '구매금액'],
                label = 'cluster ' + str(i))
plt.legend()
plt.title('K = %d results'%k , size = 15)
plt.xlabel('구매횟수', size = 12)
plt.ylabel('구매금액', size = 12)
plt.show()


# In[ ]:




