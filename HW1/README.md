# DSAI-HW1-2022

## Introduction

### Overview
> **Electricity Forecasting**  
In this HW, we will implement an algorithm to predict the operating reserve (備轉容量) of electrical power. Given a time series electricity data to predict the value of the operating reserve value of each day during 2022/03/30 ~ 2022/04/13. 

### Goal
> Predict the operating reserve (備轉容量) value from 2022/03/30 to 2022/04/13.

## Data
- 台灣電力公司_過去電力供需資訊
- 台灣電力公司_本年度每日尖峰備轉容量率

## Idea
> 透過LSTM模型進行預測

### Model
```
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
```