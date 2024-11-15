import os  
import joblib  
import numpy as np  
import pandas as pd 
from tensorflow.keras.models import load_model  
from sklearn.preprocessing import MinMaxScaler

#載入訓練資料
DataName = os.getcwd()+'\ExampleTrainData(AVG)\AvgDATA_17.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
AllOutPut = SourceData[['Temperature(°C)','Humidity(%)','Sunlight(Lux)']].values

#正規化
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)

#載入模型
regressor = load_model('WheatherLSTM_2024-11-14T19_55_11Z.h5')
Regression = joblib.load('WheatherRegression')

# 載入測試資料
DataName = os.getcwd()+r'\ExampleTestData\upload.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values

inputs = [] #存放參考資料
PredictOutput = [] #存放預測值(天氣參數)
PredictPower = [] #存放預測值(發電量) 
LookBackNum = 12  # LSTM往前看的筆數
ForecastNum = 48 #預測筆數

count = 0
while(count < len(EXquestion)):
  print('count : ',count)
  LocationCode = int(EXquestion[count])
  strLocationCode = str(LocationCode)[-2:]
  if LocationCode < 10 :
    strLocationCode = '0'+LocationCode

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

  