# DSAI-HW3-2022

## Introduction

- You will act as a house owner.
- You'll generate and consume power.
- You can decide to buy or sell the power from the microgrid through a local power trading platform.
- **Goal** : Design an agent for bidding power to minimize your electricity bill.

## Data
1. 50 households synthetic data
2. Hourly Power consumption (kWh)
3. Hourly Solar Power generation (kWh)
4. Hourly Bidding records
5. 12 months of data
   - 8 months for training
   - 1 months for validation
   - 3 months for testing

## Idea

### Data preprocess
將得到的50戶發電用電量先平均，在用標準差與平均去做標準化
```
for i in range(50):
    path='training_data/target{}.csv'.format(i)
    x = pd.read_csv(path)
    x = x.drop('time', axis=1)
    temp.append(x)
    if i == 0:
        train_data = x
    else:
        train_data += x

# train_data = pd.concat(temp,ignore_index=True)
train_data/=50
normalized_data=train_data.apply(lambda x:(x-x.mean())/ x.std())
```

### Model
```
class LSTM(torch.nn.Module):
    def __init__(self, input=24, hidden_size=8, output=24):
        # Input of LSTM : batch, sequence_len, input_size
        super(LSTM, self).__init__()

        self.rnn = torch.nn.LSTM(
            input, hidden_size, num_layers=2, dropout=0.05, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)
        a, b, c = hidden.shape
        # print(hidden.reshape(a*b,c).shape)
        # out = self.linear(hidden.reshape(a*b,c))
        out = self.linear(out[:, -1, :])

        return out
```

### Prediction
根據這次作業的需求，利用前7天的數據去預測第8天的產電量與耗電量，訓練是也是照個7:1的資料下去訓練。

下圖是訓練的結果

**Generation**

![img](https://github.com/fancyshon/DSAI_2022/blob/main/HW3/readme_img/generation.png?raw=true)

**Consumption**

![img](https://github.com/fancyshon/DSAI_2022/blob/main/HW3/readme_img/consumption.png?raw=true)

可以看到在generation的部分大致上是準確的，因此generation直接使用訓練結果的模型，而在consumption的部分則是可以預測到波動，低點大致符合，但在高點的地方預測出來的值要比實際的數據來的小，因此我決定在當數據高過一定的值後將其乘上倍數放大來當最後結果。

### Strategy
+ 產電量 > 耗電量
   + 將多餘的電用比台電低價個賣出
+ 產電量 < 耗電量
   - 將產的電用比台電高一些的價格掛賣單
   + 將耗電量減去產電量並以低台電一些的價格掛買單
   + 盡量減少損失
+ 產電量 = 耗電量
   + 不做任何操作
