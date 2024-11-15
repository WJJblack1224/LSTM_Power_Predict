from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

LookBackNum = 12 #LSTM往前看的筆數

#載入訓練資料
DataName = os.getcwd()+'\ExampleTrainData(AVG)\AvgDATA_17.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')

#(溫度,濕度,光照度)
AllOutPut = SourceData[['Temperature(°C)','Humidity(%)','Sunlight(Lux)']].values

#正規化
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)

# 檢查並建立模型存放資料夾
model_dir = './models'
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

#為不同站點微調模型
def FineTune(LocationCode):
  # 格式化LocationCode為兩位數
  LocationCode = f'{int(LocationCode):02d}'

  # 載入預訓練模型
  regressor = load_model('WheatherLSTM_2024-11-14T19_55_11Z.h5')

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