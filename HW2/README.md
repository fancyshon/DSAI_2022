# DSAI-HW2-2022

## Execution
```
python trader.py
```
執行後會抓取training.csv訓練，並輸出output.csv，裡面有最終的執行結果

## 想法
以LSTM模型作為基礎的架構，因為這次作業的test data只有20天，因此我只有取兩天去預測下一天開盤價，希望可以減少因為等待2天的資料所損失可以執行動作的次數。