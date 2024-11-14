#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

import numpy as np
import pandas as pd
import os

#設定LSTM往前看的筆數和預測筆數
LookBackNum = 12 #LSTM往前看的筆數
ForecastNum = 48 #預測筆數

#載入訓練資料
DataName = os.getcwd()+'\ExampleTrainData(AVG)\AvgDATA_17.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')

#迴歸分析 選擇要留下來的資料欄位
#(溫度,濕度,光照度)
#(發電量)
Regression_X_train = SourceData[['Temperature(°C)','Humidity(%)','Sunlight(Lux)']].values
Regression_y_train = SourceData[['Power(mW)']].values

#LSTM 選擇要留下來的資料欄位
#(溫度,濕度,光照度)
AllOutPut = SourceData[['Temperature(°C)','Humidity(%)','Sunlight(Lux)']].values

#正規化
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

X_train = []
y_train = []

#設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum,len(AllOutPut_MinMax)):
  X_train.append(AllOutPut_MinMax[i-LookBackNum:i, :])
  y_train.append(AllOutPut_MinMax[i, :])


X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
#(samples 是訓練樣本數量,timesteps 是每個樣本的時間步長,features 是每個時間步的特徵數量)
X_train = np.reshape(X_train,(X_train.shape [0], X_train.shape [1], 3))


#%%
#============================建置&訓練「LSTM模型」============================
#建置LSTM模型

regressor = Sequential ()

regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], 3)))

regressor.add(LSTM(units = 64))

regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 3))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#開始訓練
regressor.fit(X_train, y_train, epochs = 100, batch_size = 128)

#保存模型
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
regressor.save('WheatherLSTM_'+NowDateTime+'.h5')
print('Model Saved')


#%%
#============================建置&訓練「回歸模型」========================

#開始迴歸分析(對發電量做迴歸)
RegressionModel = LinearRegression()
RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

#儲存回歸模型
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
joblib.dump(RegressionModel, 'WheatherRegression_'+NowDateTime)

#取得截距
print('截距: ',RegressionModel.intercept_)

#取得係數
print('係數 : ', RegressionModel.coef_)

#取得R平方
print('R squared: ',RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))


#%%
#============================對模型微調============================

from tensorflow.keras.optimizers import Adam

#為不同站點微調模型
def FineTune(LocationCode):
  # 格式化LocationCode為兩位數
  LocationCode = f'{int(LocationCode):02d}'

  # 載入預訓練模型
  regressor = load_model('WheatherLSTM_2024-11-15T00_11_08Z.h5')
  # 檢查並建立模型存放資料夾
  model_dir = './models'
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  # 載入各站點的數據進行微調
  DataName = os.path.join(os.getcwd(), 'ExampleTrainData(AVG)', f'AvgDATA_{LocationCode}.csv')
  SourceData = pd.read_csv(DataName, encoding='utf-8')

  AllOutPut = SourceData[['Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values

  # 正規化
  AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

  # LSTM訓練資料構建
  X_train, y_train = [], []
  for i in range(LookBackNum, len(AllOutPut_MinMax)):
    X_train.append(AllOutPut_MinMax[i - LookBackNum:i, :])
    y_train.append(AllOutPut_MinMax[i, :])

  X_train, y_train = np.array(X_train), np.array(y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 3))

  # 凍結基礎層，只解凍最後一層 LSTM 和輸出層
  for layer in regressor.layers[:-2]: 
    layer.trainable = False
  # 編譯模型並設定較低的學習率
  regressor.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error')
  # 使用新的數據進行微調
  regressor.fit(X_train, y_train, epochs=10, batch_size=128)

  # 保存微調後的模型
  regressor.save(os.path.join(model_dir, f'WheatherLSTM_{LocationCode}.h5'))
  print(f'Model {LocationCode} Saved')

for i in range(1,18):
    FineTune(i)


#%%

#載入回歸模型
Regression = joblib.load('WheatherRegression_2024-11-15T00_11_17Z')

#載入測試資料
DataName = os.getcwd()+r'\ExampleTestData\upload.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values

inputs = [] #存放參考資料
PredictOutput = [] #存放預測值(天氣參數)
PredictPower = [] #存放預測值(發電量) 

count = 0
while(count < len(EXquestion)):
  print('count : ',count)
  LocationCode = int(EXquestion[count])
  strLocationCode = f'{LocationCode:02d}'[-2:]

  # 根據編號載入指定的模型
  regressor = load_model(f'./models/WheatherLSTM_{strLocationCode}.h5')

  # 根據編號載入指定的資料集
  DataName = os.getcwd()+'\ExampleTrainData(IncompleteAVG)\IncompleteAvgDATA_'+ strLocationCode +'.csv'
  SourceData = pd.read_csv(DataName, encoding='utf-8')
  ReferTitle = SourceData[['Serial']].values
  ReferData = SourceData[['Temperature(°C)','Humidity(%)','Sunlight(Lux)']].values
  
  inputs = []#重置存放參考資料

  #找到相同的一天，把12個資料都加進inputs
  for DaysCount in range(len(ReferTitle)):
    if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]):
      TempData = ReferData[DaysCount].reshape(1,-1)
      TempData = LSTM_MinMaxModel.transform(TempData)
      inputs.append(TempData)

  #用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
  for i in range(ForecastNum) :

    #print(i)
    
    #將新的預測值加入參考資料(用自己的預測值往前看)
    if i > 0 :
      inputs.append(PredictOutput[i-1].reshape(1,3))

    #切出新的參考資料12筆(往前看12筆)
    X_test = []
    X_test.append(inputs[0+i:LookBackNum+i])
    
    #Reshaping
    NewTest = np.array(X_test)
    NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 3))
    
    predicted = regressor.predict(NewTest)
    PredictOutput.append(predicted)
    PredictPower.append(np.round(Regression.predict(predicted),2).flatten())
  
  #每次預測都要預測48個，因此加48個會切到下一天
  #0~47,48~95,96~143...
  count += 48

#寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame(PredictPower, columns=['答案'])

# 將 DataFrame 寫入 CSV 檔案
df.to_csv('output.csv', index=False) 
print('Output CSV File Saved')

# %%