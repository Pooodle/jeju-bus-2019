#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from collections import Counter # count 용도

import matplotlib.pyplot as plt # 시각화
import seaborn as sns #시각화

import folium # 지도 관련 시각화
from folium.plugins import MarkerCluster #지도 관련 시각화
import geopy.distance #거리 계산해주는 패키지 사용

import random #데이터 샘플링
from sklearn.model_selection import GridSearchCV #모델링
from sklearn.ensemble import RandomForestRegressor #모델링


# In[2]:


os.chdir("./data/")


# In[3]:


train = pd.read_csv("train_ansi.csv", encoding='CP949')
test = pd.read_csv("test_ansi.csv", encoding='CP949')


# In[4]:


#check the numbers of samples and features
print("The train data size is : {} ".format(train.shape))
print("The test data size is : {} ".format(test.shape))


# * 외부데이터(기간: 2019.09.01 ~ 10.16)
# * 06~11시에 해당되는 사항들만 편집

# In[5]:


#데이터 불러오기
raining=pd.read_csv("weather.csv",engine='python')

#외부데이터에서 나오는 지점명들을 변경
raining['지점'] = [ str(i) for i in raining['지점'] ]

raining['지점'] = ['jeju' if i=='184' else i for i in raining['지점'] ]  # 위도 : 33.51411 경도 : 126.52969
raining['지점'] = ['gosan' if i=='185' else i for i in raining['지점'] ]  # 위도 : 33.29382 경도 : 126.16283
raining['지점'] = ['seongsan' if i=='188' else i for i in raining['지점'] ]  # 위도 : 33.38677 경도 : 126.8802
raining['지점'] = ['po' if i=='189' else i for i in raining['지점'] ]  # 위도 : 33.24616 경도 : 126.5653

raining.head()


# In[6]:


# bts = pd.read_csv("./input/exdata/train_cate_onoff.csv",engine='python')
# bts.info()


# ## 2. Features engineering
# ### 1) target feature 분리

# In[7]:


ntrain = train.shape[0]
ntest = test.shape[0]
# y_train = train["18~20_ride"].values
y_train = train["18~20_ride"]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['18~20_ride'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ### 2) missing data 처리

# In[8]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# #### "Do not need to imput missing value"

# ### 3) Data Correlation

# In[9]:


#Correlation map to see how features are correlated with 18~20_ride
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[10]:


corrmat


# ### 4) Data Columns Classification

# In[11]:


# all_data.index


# In[12]:


# all_data.index.dtype


# In[13]:


# all_data.index.astype(np.float64, copy=False)


# In[14]:


# all_data.index.dtype


# In[15]:


all_data.info()


# In[16]:


cate_feature = [col for col in all_data.columns if all_data[col].dtypes=="object"]
cate_feature=list(set(cate_feature))
num_feature = list(set(all_data.columns)-set(cate_feature))


# In[17]:


# cate_feature


# In[18]:


# num_feature


# In[19]:


# for i in cate_feature:
#     vals = set(all_data[i].values)
#     cnt = all_data[i].value_counts().sort_index(ascending=True)
#     print(i, "\n", cnt, "\n""\n")
# #     val_map = map({'cate_feature':i,'val':vals, 'val_cnt':len(vals)})


# ### 4)-A. categorical 변수

# #### A-1. date 변수 변환

# In[20]:


all_data['date'] = pd.to_datetime(all_data['date'])


# In[21]:


all_data['weekday'] = all_data['date'].dt.weekday
# Monday is 0 and Sunday is 6.


# In[22]:


all_data = pd.get_dummies(all_data, columns=['weekday'])


# #### A-2. in_out 변수 변환

# In[23]:


# all_data['in_out'].value_counts()


# In[24]:


all_data['in_out'] = all_data['in_out'].map({'시내':0,'시외':1})


# ### 4)-B. numeric 변수

# #### B-1. 승차, 하차시간: 승차, 하차 시간대 통합 작업

# In[25]:


all_data['68_ride']=all_data['6~7_ride']+all_data['7~8_ride'] # 6 ~ 8시 승차인원
all_data['810_ride']=all_data['8~9_ride']+all_data['9~10_ride']
all_data['1012_ride']=all_data['10~11_ride']+all_data['11~12_ride']

all_data['68_takeoff']=all_data['6~7_takeoff']+all_data['7~8_takeoff'] # 6 ~ 8시 하차인원
all_data['810_takeoff']=all_data['8~9_takeoff']+all_data['9~10_takeoff']
all_data['1012_takeoff']=all_data['10~11_takeoff']+all_data['11~12_takeoff']

all_data['ride']=all_data['68_ride']+all_data['810_ride']+all_data['1012_ride']
all_data['takeoff']=all_data['68_takeoff']+all_data['810_takeoff']+all_data['1012_takeoff']


# In[26]:


# train22=all_data[['6~7_ride', '7~8_ride', "8~9_ride", '9~10_ride', '10~11_ride', '11~12_ride', '68ride', "810_ride", "1012_ride"]]
# train22["ride"]=y_train

# cor=train22.corr()


# In[27]:


# cor


# In[28]:


# all_data.columns


# * 차트로 target feature와의 상관관계 확인용 train 분리

# In[29]:


# train = all_data[:ntrain] #ntrain = train.shape[0]
# # test = all_data[ntrain:]


# In[30]:


# train22=train[['ride', 'takeoff']]
# train22["18~20_ride"]=y_train

# cor=train22.corr()
# # sns.set(style="white")
# # mask=np.zeros_like(cor,dtype=np.bool)
# # mask[np.triu_indices_from(mask)]=True

# # f,ax=plt.subplots(figsize=(20,15))
# # cmap=sns.diverging_palette(200,10,as_cmap=True)
# # sns.heatmap(cor, mask=mask,cmap=cmap,center=0,square=True,linewidths=0.5,cbar_kws={"shrink":1},annot=True); #히트맵 생성
# # plt.xticks(size=20)
# # plt.yticks(size=20,rotation=0)
# # plt.title("arrive and leave correlation graph",size=30);


# In[31]:


# cor


# * 동일한 혹은 더 높은 상관관계 실현

# In[32]:


drop_columns1 = ['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride']
drop_columns2 = ['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']
all_data.drop(drop_columns1, axis=1, inplace=True)
all_data.drop(drop_columns2, axis=1, inplace=True)
all_data.columns


# #### C-1. 좌표 데이터
# - 제주도의 인구는 서귀포시와 제주시에 밀집
# - 해당 지역 및 서쪽 동쪽 지역의 위치 활용, 해당 지역과의 거리를 각각 feature로 추가

# * 제주 측정소의 위.경도: 33.51411, 126.52969
# * 고산 측정소의 위.경도: 33.29382, 126.16283
# * 성산 측정소의 위.경도: 33.38677, 126.880
# * 서귀포 측정소의 위.경도: 33.24616, 126.5653

# In[33]:


# 해당 주요 장소의 임의 지역 위도, 경도

jeju=(33.51411, 126.52969) # 제주 측정소 근처
gosan=(33.29382, 126.16283) #고산 측정소 근처
seongsan=(33.38677, 126.8802) #성산 측정소 근처
po=(33.24616, 126.5653) #서귀포 측정소 근처
uni=(33.458564, 126.561722) #제주대학교 

#제주도 지역이 보일 수 있는 위치의 위도, 경도를 표시한 뒤, folium.Map에 변수로 넣고, map_osm에 할당
map_osm= folium.Map((33.399835, 126.506031),zoom_start=9)
mc = MarkerCluster()

mc.add_child( folium.Marker(location=jeju,popup='제주 측정소',icon=folium.Icon(color='red',icon='info-sign') ) ) #제주 측정소 마커 생성
map_osm.add_child(mc) #마커를 map_osm에 추가

mc.add_child( folium.Marker(location=gosan,popup='고산 측정소',icon=folium.Icon(color='red',icon='info-sign') ) )
map_osm.add_child(mc) 

mc.add_child( folium.Marker(location=seongsan,popup='성산 측정소',icon=folium.Icon(color='red',icon='info-sign') ) )
map_osm.add_child(mc) 

mc.add_child( folium.Marker(location=po,popup='서귀포 측정소',icon=folium.Icon(color='red',icon='info-sign') ) )
map_osm.add_child(mc)

mc.add_child( folium.Marker(location=uni,popup='제주대 측정소',icon=folium.Icon(color='red',icon='info-sign') ) )
map_osm.add_child(mc)


# In[34]:


# #정류장의 위치만 확인하기 위해 groupby를 실행함
# data=all_data[['latitude','longitude','station_name']].drop_duplicates(keep='first')

# data2=data.groupby(['station_name'])['latitude','longitude'].mean()

# data2.to_csv("folium.csv")

# data2=pd.read_csv("folium.csv")

# #정류장의 대략적인 위치를 확인하기 위하여, folium map에 해당 정류장을 표시
# for row in data2.itertuples():
#     mc.add_child(folium.Marker(location=[row.latitude,  row.longitude], popup=row.station_name)) #마커 생성
#     map_osm.add_child(mc) #마커를 map_osm에 추가
    
# map_osm


# #### C-2. 측정소와 정류장 사이 거리 계산 적용

# geopy.distance.vincenty를 이용, m/km 단위 계산 (현재 km 단위 사용)
# * dis_jeju : 버스정류장과 제주 측정소와의 거리
# * dis_gosan : 버스정류장과 고산 측정소시와의 거리
# * dis_seongsan : 버스정류장과 성산 측정소와의 거리
# * dis_po : 버스정류장과 서귀포 측정소와의 거리

# In[35]:


t1 = [geopy.distance.vincenty( (i,j), jeju).km for i,j in list( zip( all_data['latitude'],all_data['longitude'] )) ]
t2 = [geopy.distance.vincenty( (i,j), gosan).km for i,j in list( zip( all_data['latitude'],all_data['longitude'] )) ]
t3 = [geopy.distance.vincenty( (i,j), seongsan).km for i,j in list( zip( all_data['latitude'],all_data['longitude'] )) ]
t4 = [geopy.distance.vincenty( (i,j), po).km for i,j in list( zip( all_data['latitude'],all_data['longitude'] )) ]
t5 = [geopy.distance.vincenty( (i,j), uni).km for i,j in list( zip( all_data['latitude'],all_data['longitude'] )) ]

all_data['dis_jeju']=t1
all_data['dis_gosan']=t2
all_data['dis_seongsan']=t3
all_data['dis_po']=t4
all_data['dis_uni']=t5


# In[36]:


total=pd.DataFrame( list(zip( t1,t2,t3,t4,t5)),columns=['jeju','gosan','seongsan','po','jeju'] )
total


# - 변수 생성(dist_name): 해당 정류소에서 가장 가까운 측정소(jeju, gosan, seongsan, po)

# In[37]:


all_data['dist_name'] = total.apply(lambda x: x.argmin(), axis=1)

data22=all_data[['station_name','latitude','longitude','dist_name']].drop_duplicates(keep='first')

# 전체 정류장별 어느 측정소와 가장 가까운지 Counter를 통해 확인
Counter(data22['dist_name'])


# ### 5) 외부데이터 활용
# #### 5)-A. 외부 날씨 측정 데이터
# * 일시와 시간대를 분리

# In[38]:


raining['time'] = [ int( i.split(' ')[1].split(':')[0] ) for i in raining['일시']] 
raining['일시'] = [ i.split(' ')[0] for i in raining['일시'] ] 

# 실제 측정 데이터이기 때문에, 12시 이전의 시간대만 사용
raining = raining[ (raining['time']>=6) & (raining['time']<12)  ]


# In[39]:


# raining


# #### A-1. (Feature Engineering) 새로운 변수 생성: 해당 시간대 평균 기온 및 강수량(groupby)

# In[40]:


rain = raining.groupby(['지점','일시'])[['기온(°C)','강수량(mm)']].mean()

rain.to_csv("rain.csv")

rain=pd.read_csv("rain.csv")


# * 변수명 동일하게 변경, 결측 처리 (NaN == 0.0000)

# In[41]:


# train, test의 변수명과 통일시키고, NaN의 값은 0.0000으로 변경
rain = rain.rename(columns={"일시":"date","지점":"dist_name"})
rain= rain.fillna(0.00000)


# * all_data에 merge

# In[42]:


# all_data.info()


# In[43]:


# rain.info()


# In[44]:


rain['date'] = pd.to_datetime(rain['date'])


# In[45]:


all_data=pd.merge(all_data, rain, how='left',on=['dist_name','date'])


# ### 6) 외부데이터 결합 후 dist_name dummy화

# In[46]:


all_data = pd.get_dummies(all_data, columns=['dist_name'])


# ### *) holiday categorical 변수 추가 

# In[47]:


# from korean_lunar_calendar import KoreanLunarCalendar


# In[48]:


# holiday_list = []

# #추석
# calendar = KoreanLunarCalendar()
# calendar.setLunarDate(lunarYear = 2019, lunarMonth=8, lunarDay=15, isIntercalation=False)
# holiday_list.append(calendar.SolarIsoFormat())
# calendar.setLunarDate(lunarYear = 2019, lunarMonth=8, lunarDay=14, isIntercalation=False)
# holiday_list.append(calendar.SolarIsoFormat())
# calendar.setLunarDate(lunarYear = 2019, lunarMonth=8, lunarDay=16, isIntercalation=False)
# holiday_list.append(calendar.SolarIsoFormat())

# #추석 연결 일요일
# holiday_list.append('2019-09-15')

# #한글날
# holiday_list.append('2019-10-09')

# #개천절
# holiday_list.append('2019-10-03')

# print(len(holiday_list))


# In[49]:


# all_data['holiday']=0

# ind = all_data['id'][all_data['date'].isin(holiday_list)==True]
# lis = ind.tolist()
# # type(lis)
# # len(lis)

# for i in lis:
#     all_data['holiday'][i]=1


# In[50]:


# all_data.columns


# ### 7) 불필요한 col 삭제

# In[51]:


drop_columns3 = ['bus_route_id', 'date', 'station_name', 'station_code']
all_data.drop(drop_columns3, axis=1, inplace=True)


# ### 8) col 이름 변경(모델 학습시 인코딩 문제 발생)

# In[52]:


all_data.rename(columns = {'기온(°C)' : 'temperatures'}, inplace = True)
all_data.rename(columns = {'강수량(mm)' : 'precipitation'}, inplace = True)


# ### *) latitude 이상치(섬) 데이터 삭제

# In[53]:


# ind2 = all_data['latitude'][all_data['latitude']>33.7].index

# all_data.drop(index=ind2, inplace=True)


# In[54]:


# all_data.shape


# In[55]:


# cor_test=all_data[['latitude','longitude']]
# cor_test["18~20_ride"]=y_train

# cor=cor_test.corr()
# cor


# ### *) 

# In[56]:


all_data.loc[all_data['latitude'] < 33.36, 'i_lat'] = 0
all_data.loc[all_data['latitude'] >= 33.36, 'i_lat'] = 1

all_data.loc[all_data['longitude'] < 126.52, 'i_long'] = 0
all_data.loc[all_data['longitude'] >= 126.52, 'i_long'] = 1

all_data.loc[all_data['dis_jeju'] < 20, 'i_jeju'] = 0
all_data.loc[all_data['dis_jeju'] >= 20, 'i_jeju'] = 1

all_data.loc[all_data['dis_gosan'] < 40, 'i_go'] = 0
all_data.loc[all_data['dis_gosan'] >= 40, 'i_go'] = 1

all_data.loc[all_data['dis_seongsan'] < 35, 'i_seong'] = 0
all_data.loc[all_data['dis_seongsan'] >= 35, 'i_seong'] = 1

all_data.loc[all_data['dis_po'] < 10, 'i_po'] = 0
all_data.loc[all_data['dis_po'] >= 10, 'i_po'] = 1

all_data.loc[all_data['dis_uni'] < 15, 'i_uni'] = 0
all_data.loc[all_data['dis_uni'] >= 15, 'i_uni'] = 1


# In[57]:


all_data['latitude1'] = all_data['latitude'] - 33.36
all_data['latitude2'] = all_data['latitude1']*all_data['i_lat']

all_data['longitude1'] = all_data['longitude'] - 126.52
all_data['longitude2'] = all_data['longitude1']*all_data['i_long']

all_data['dis_jeju1'] = all_data['dis_jeju'] - 20
all_data['dis_jeju2'] = all_data['dis_jeju1']*all_data['i_jeju']
all_data['dis_jeju'] = np.log(all_data['dis_jeju'])

all_data['dis_gosan1'] = all_data['dis_gosan'] - 40
all_data['dis_gosan2'] = all_data['dis_gosan1']*all_data['i_go']

all_data['dis_seongsan1'] = all_data['dis_seongsan'] - 35
all_data['dis_seongsan2'] = all_data['dis_seongsan1']*all_data['i_seong']

all_data['dis_po1'] = all_data['dis_po'] - 10
all_data['dis_po2'] = all_data['dis_po1']*all_data['i_po']

all_data['dis_uni1'] = all_data['dis_uni'] - 15
all_data['log_dis_uni'] = np.log(all_data['dis_uni1'])
all_data = all_data.fillna(0.0)
all_data['dis_uni2'] = all_data['log_dis_uni'] * all_data['i_uni']


# ### 9) 데이터 분할

# In[58]:


train = all_data[:ntrain] #ntrain = train.shape[0]
test = all_data[ntrain:]


# In[59]:


train.shape


# In[60]:


# train.info()


# In[61]:


# train.index


# In[62]:


test.shape


# In[63]:


# test.info()


# ## 3. Modelling - Random Forest
# ### *) 평가

# In[64]:


from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# * 단순 평가

# In[65]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# * Define a cross validation strategy

# In[66]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# ### 1) RF hyper parameter: GridSearch
# * 전체 데이터로 best hyper paramter 구하는 데에 긴 시간 소요
# * train data의 1% 데이터만으로 best parameter 찾고, 전체 train data에 학습

# In[67]:


input_var=['in_out','latitude', 'latitude2', 'longitude', 'longitude2', 
            '68_ride', '810_ride', '1012_ride', '68_takeoff', '810_takeoff','1012_takeoff',
           'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
           'dis_uni', 'dis_uni2', 'dis_jeju', 'dis_jeju2', 'dis_po', 'dis_po2', 'precipitation', 
           'dist_name_jeju','dist_name_po',  'dist_name_gosan', 'dist_name_seongsan', 
           'ride', 'takeoff']


# In[68]:


X_train=train[input_var]
random.seed(333) #동일한 샘플링하기 위한 시드번호
train_list=random.sample(list(range(train.shape[0])), int(round(train.shape[0]*0.01,0)) )

X_train=train[input_var]
X_train=X_train.iloc[train_list,:]
y_train2=y_train.iloc[train_list]

X_test=test[input_var]

X_train.shape, y_train.shape


# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_features': [2,3,5],
    'min_samples_leaf': [2,3],
    'min_samples_split': [2,4,6],
    'n_estimators': [100, 200,500]
}

rf = RandomForestRegressor(random_state=1217) # 랜덤포레스트 모델을 정의한다.

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid) # GridSearchCV를 정의한다.

grid_search.fit(X_train, y_train2)

grid_search.best_params_

#해당 코드 실행시간 2분 ~ 3분 소요


# In[ ]:


#전체 데이터로 적용
X_train=train[input_var]
X_test=test[input_var]

X_train.shape, y_train.shape, X_test.shape


# ### 2) best_params_ 값 대입

# In[ ]:


rf = RandomForestRegressor(max_features=5,min_samples_leaf=2,min_samples_split=2,n_estimators=100,random_state=333)

rf.fit(X_train,y_train) #학습 
test['18~20_ride'] = rf.predict(X_test) #예측값 생성 후, test['18~20_ride']에 집어 넣는다.
#해당 코드 소요 시간 5분


# In[ ]:


# 단순 평가
print("(S)rf score: ", rmsle(y_train, rf.predict(X_train.values)))


# In[ ]:


# score = rmsle_cv(rf)
# print("(KF) rf score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# * 대회 제출용 저장

# In[ ]:


test[['id','18~20_ride']].to_csv("dacon_base_middle.csv",index=False) # id와 18~20_ride만 선택 후 csv 파일로 내보낸다
test.drop('18~20_ride', axis=1, inplace=True)


# ## 4. Modelling
# <!-- ### 1) hyper parameter: GridSearch
# * 전체 데이터로 best hyper paramter 구하는 데에 긴 시간 소요
# * train data의 1% 데이터만으로 best parameter 찾고, 전체 train data에 학습 -->

# ### 1) Base models

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(random_state=3))
# KRR = KernelRidge(kernel='rbf')


# In[ ]:


lasso.fit(X_train.values, y_train)
print("lasso fitting DONE===============")

ENet.fit(X_train.values, y_train)
print("ENet fitting DONE===============")

# KRR.fit(X_train.values, y_train)
# print("KRR fitting DONE===============")


# In[ ]:


# 단순평가
print("(S)lasso score: ", rmsle(y_train, lasso.predict(X_train.values)))
print("(S)ENet score: ", rmsle(y_train, ENet.predict(X_train.values)))
# print("(S)KRR score: ", rmsle(y_train, KRR.predict(X_train.values)))


# In[ ]:


# # 교차검증
# score = rmsle_cv(lasso)
# print("\n (KF)Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = rmsle_cv(ENet)
# print("\n (KF)ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# # score = rmsle_cv(KRR)
# # print("\n (KF)Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# #### 2)-C. Boosting Models

# In[ ]:


GBoost = GradientBoostingRegressor(random_state =5)
model_xgb = xgb.XGBRegressor(random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor()


# In[ ]:


GBoost.fit(X_train.values, y_train)
print("GBoost fitting DONE===============")

# model_xgb.fit(X_train.values, y_train)
# print("model_xgb fitting DONE===============")

# model_lgb.fit(X_train.values, y_train)
# print("model_lgb fitting DONE===============")


# In[ ]:


gboost_train_pred = GBoost.predict(X_train.values)
# xgb_train_pred = model_xgb.predict(X_train.values)
# lgb_train_pred = model_lgb.predict(X_train.values)


# In[ ]:


# 단순평가
print("(S)GBoost score: ", rmsle(y_train, gboost_train_pred))
# print("(S)xgb score: ", rmsle(y_train, xgb_train_pred))
# print("(S)lgb score: ", rmsle(y_train, lgb_train_pred))


# In[ ]:


# # 교차검증
# score = rmsle_cv(GBoost)
# print("\n (KF)GBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = rmsle_cv(model_xgb)
# print("\n (KF)xgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = rmsle_cv(model_lgb)
# print("\n (KF)lgb Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# ### 2)  Stacking models
# #### 2)-A. Averaged base models class: 모델의 단순 rmse 평균 비교

# In[ ]:


# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, models):
#         self.models = models
        
#     # we define clones of the original models to fit the data in
#     def fit(self, X, y):
#         self.models_ = [clone(x) for x in self.models]
        
#         # Train cloned base models
#         for model in self.models_:
#             model.fit(X, y)

#         return self
    
#     #Now we do the predictions for cloned models and average them
#     def predict(self, X):
#         predictions = np.column_stack([
#             model.predict(X) for model in self.models_
#         ])
#         return np.mean(predictions, axis=1)  


# In[ ]:


# averaged_models = AveragingModels(models = (ENet, GBoost, rf, lasso))


# In[ ]:


# averaged_models.fit(X_train.values, y_train)
# # 단순평가
# print("(S)averaged_models score: ", rmsle(y_train, averaged_models.predict(X_train.values)))


# In[ ]:


# # 교차검증
# score = rmsle_cv(averaged_models)
# print("\n (KF)Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# #### 2)-B. Stacking averaged Models Class

# In[ ]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[ ]:


# stacked_averaged_models1 = StackingAveragedModels(base_models = (ENet, GBoost, rf),
#                                                  meta_model = lasso)
# stacked_averaged_models2 = StackingAveragedModels(base_models = (lasso, GBoost, rf),
#                                                  meta_model = ENet)
stacked_averaged_models3 = StackingAveragedModels(base_models = (lasso, ENet, rf),
                                                 meta_model = GBoost)
# stacked_averaged_models4 = StackingAveragedModels(base_models = (lasso, ENet, GBoost),
#                                                  meta_model = rf)


# In[ ]:


# stacked_averaged_models1.fit(X_train.values, y_train)
# print("stacked averaged models (lasso) fitting DONE===============")

# stacked_averaged_models2.fit(X_train.values, y_train)
# print("stacked averaged models (ENet) fitting DONE===============")

stacked_averaged_models3.fit(X_train.values, y_train)
print("stacked averaged models (GBoost) fitting DONE===============")

# stacked_averaged_models4.fit(X_train.values, y_train)
# print("stacked averaged models (lasso) fitting DONE===============")


# In[ ]:


# s1_pred = stacked_averaged_models1.predict(X_train)
# s2_pred = stacked_averaged_models2.predict(X_train)
s3_pred =  stacked_averaged_models3.predict(X_train)


# In[ ]:


# 단순평가
# print("(S)stacked averaged models (lasso) score: ", rmsle(y_train, s1_pred))
# print("(S)stacked averaged models (ENet) score: ", rmsle(y_train, s2_pred))
print("(S)stacked averaged models (GBoost) score: ", rmsle(y_train, s3_pred))
# print("(S)stacked averaged models (rf) score: ", rmsle(y_train, stacked_averaged_models4.predict(X_train)))


# * (S)stacked averaged models (GBoost) score가 가장 낮음

# ## 5. Final Training and Prediction

# In[ ]:


# test.columns


# In[ ]:


# test.drop('18~20_ride', axis=1, inplace=True)


# In[ ]:


# model_xgb.fit(X_train, y_train)
# xgb_pred = model_xgb.predict(X_test)


# In[ ]:


# model_lgb.fit(X_train, y_train)
# lgb_pred = model_lgb.predict(X_test.values)


# In[ ]:


stacked_averaged_models3.fit(X_train.values, y_train)
stacked3_pred = stacked_averaged_models3.predict(X_test.values)


# In[ ]:


test['18~20_ride'] = stacked3_pred
test[['id','18~20_ride']].to_csv("dacon_mid_line_sam_gboost3.csv",index=False)
test.drop('18~20_ride', axis=1, inplace=True)


# In[ ]:


# stacked_averaged_models1.fit(X_train.values, y_train)
# stacked1_pred = stacked_averaged_models1.predict(X_test.values)


# In[ ]:


# test['18~20_ride'] = stacked1_pred
# test[['id','18~20_ride']].to_csv("dacon_mid_line_sam_gboost1.csv",index=False)
# test.drop('18~20_ride', axis=1, inplace=True)


# In[ ]:


# stacked_averaged_models2.fit(X_train.values, y_train)
# stacked2_pred = stacked_averaged_models2.predict(X_test.values)


# In[ ]:


# test['18~20_ride'] = stacked2_pred
# test[['id','18~20_ride']].to_csv("dacon_mid_line_sam_gboost2.csv",index=False)
# test.drop('18~20_ride', axis=1, inplace=True)


# In[ ]:


# sss = pd.read_csv("dacon_mid_line_sam_gboost.csv")
# sss.head()


# In[ ]:


# ensemble = stacked3_pred*0.90 + xgb_pred*0.05 + lgb_pred*0.05


# In[ ]:


# test['18~20_ride'] = ensemble
# test[['id','18~20_ride']].to_csv("dacon_mid_line_ensemble2.csv",index=False)
# test.drop('18~20_ride', axis=1, inplace=True)


# In[ ]:


# sss = pd.read_csv("dacon_mid_line_ensemble2.csv")
# sss.head()


# In[ ]:


# test.drop('18~20_ride', axis=1, inplace=True)


# https://inspiringpeople.github.io/data%20analysis/Ensemble_Stacking/
# https://swalloow.github.io/bagging-boosting
# https://blogs.sas.com/content/subconsciousmusings/2017/05/18/stacked-ensemble-models-win-data-science-competitions/

# In[ ]:




