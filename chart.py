import pandas as pd
import matplotlib.pyplot as plt

upload = pd.read_csv("./ExampleTestData/upload.csv")
output = pd.read_csv("./output.csv")

# 設定圖表大小
plt.figure(figsize=(16, 6))

# 繪製 upload 資料的 "答案" 欄位
plt.plot(upload.index, upload["答案"], label="Upload Data", marker='o',markersize=4, linewidth=1 )

# 繪製 output 資料的 "答案" 欄位
plt.plot(output.index, output["答案"], label="Output Data", marker='x',markersize=4, linewidth=1.5, alpha=0.7)

# 加入圖表標題和軸標籤
plt.title("Comparison of '答案' between Upload and Output Data")
plt.xlabel("Index")
plt.ylabel("答案")

# 加入圖例
plt.legend()

# 顯示圖表
plt.show()
