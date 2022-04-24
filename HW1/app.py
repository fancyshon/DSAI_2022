import pandas as pd
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def getData(df, column, train_size):

    train_data = np.array(df[column][:-train_size])
    test_data = np.array(df[column][-train_size:])

    return train_data, test_data

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='train_data.csv')
    parser.add_argument('--output',default='submission.csv')
    args = parser.parse_args()

    EPOCH = 100
    LR = 0.0001
    REF_DAY = 15

    device = 'cpu'
    if torch.cuda.is_available(): # 若想使用 cuda 且可以使用 cuda
        device = 'cuda'

    data=pd.read_csv('train_data.csv')
    # print(data.shape)
    # data.head()

    train_data, test_data = getData(data,'備轉容量(萬瓩)',REF_DAY)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Normalization
    train_data = scaler.fit_transform(train_data.reshape(-1,1))
    train_data = torch.FloatTensor(train_data).view(-1)
    # test_data = scaler.fit_transform(test_data.reshape(-1,1))
    # test_data = torch.FloatTensor(test_data).view(-1)
    # print(test_data)

    input=[]
    for i in range(len(train_data)-REF_DAY):
        input.append((train_data[i:REF_DAY+i],train_data[i+REF_DAY:i+REF_DAY+1]))

    model=LSTM()
    # model.to(device)
    loss=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    # print(model)

    for i in range(EPOCH):
    
        model.train()

        for seq, labels in input:
            # seq, labels = seq.to(device), labels.to(device)
            
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)
            # print(y_pred)

            single_loss = loss(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            # break

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    test_input=train_data[-REF_DAY:].tolist()
    # print(model(test_input[-REF_DAY:]).item())
    model.eval()

    for i in range(REF_DAY):
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_input.append(model(torch.FloatTensor(test_input[-REF_DAY:])).item())

    actual_result=scaler.inverse_transform(np.array(test_input[-REF_DAY:]).reshape(-1,1))
    # print(test_data)
    # print(actual_result.reshape(REF_DAY,))

    # plt.title('Comparison')
    # plt.plot(test_data)
    # plt.plot(actual_result)
    # plt.show()

    date=[n for n in range(20220330,20220332)]
    date.extend([n for n in range(20220401,20220414)])

    # print(date)
    # print(actual_result.reshape(REF_DAY,)[:len(date)])
    submission=pd.DataFrame({
        'date':date,
        'operating_reserve(MW)':actual_result.reshape(REF_DAY,)[:len(date)]
    })
    submission.head()
    submission.to_csv('submission.csv')