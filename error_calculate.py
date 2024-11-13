import pandas as pd
import numpy as np

df = pd.read_csv('./ExampleTestData/upload.csv')
df_predict = pd.read_csv('./output.csv')

# 計算每筆資料的誤差
df['Error'] = df['答案'] - df_predict['答案']

# 計算 MSE
mean_absolute_error = np.mean(df['Error'].abs())
print(f"Mean Absolute Error (MSE): {mean_absolute_error}")
total_absolute_error = df['Error'].abs().sum()
print(f"Total Absolute Error (MSE): {total_absolute_error}")

# 儲存誤差紀錄
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
output_file = 'output'+ NowDateTime + '.txt'

# 將結果寫入文字檔
with open(output_file, 'w') as file:
    file.write(f"Mean Absolute Error (MAE): {mean_absolute_error}\n")
    file.write(f"Total Absolute Error (TAE): {total_absolute_error}\n")

print(f"結果已儲存至 {output_file}")