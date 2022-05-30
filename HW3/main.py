# You should not modify this part.
import datetime
import torch
import pandas as pd
import numpy as np
import argparse


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


def config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv",
                        help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv",
                        help="input the generation data path")
    parser.add_argument(
        "--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv",
                        help="output the bids path")

    return parser.parse_args()


def output(path, data):

    df = pd.DataFrame(
        data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()

    g_model = torch.load('Generation.pt', map_location=torch.device('cpu'))
    c_model = torch.load('Consumption.pt', map_location=torch.device('cpu'))
    # 若想使用 cuda 且可以使用 cuda
    device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # print(device)

    generation = pd.read_csv(args.generation, usecols=['generation'])
    consumption = pd.read_csv(args.consumption, usecols=['consumption'])
    bidresult = pd.read_csv(args.bidresult)

    g_input = torch.from_numpy(generation.apply(lambda x: (
        x-x.mean()) / x.std()).to_numpy(dtype=np.float32).reshape(1, 7, 24))
    c_input = torch.from_numpy(consumption.apply(lambda x: (
        x-x.mean()) / x.std()).to_numpy(dtype=np.float32).reshape(1, 7, 24))

    g_mean, g_std = generation.mean()[0], generation.std()[0]
    c_mean, c_std = consumption.mean()[0], consumption.std()[0]

    # print(g_input.size())

    g_model.eval()
    c_model.eval()
    with torch.no_grad():
        g_pred = g_model(g_input.to(device)).squeeze().tolist()
        c_pred = c_model(c_input.to(device)).squeeze().tolist()

    g_pred = [i*g_std+g_mean for i in g_pred]
    c_pred = [i*c_std+c_mean for i in c_pred]

    # print(g_pred)

    # print(bidresult.empty)

    date = np.array(pd.read_csv(args.consumption, header=None))[-1,0]
    lastdate = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    date = lastdate + datetime.timedelta(days=1)
    date = date.strftime("%Y-%m-%d")
    data=[]
    # print(date)
    for i in range(24):
        # if c_pred[i]>=1.75:
        #     c_pred[i]*=1.5

        if g_pred[i] > c_pred[i]:
            # Generation more
            temp=[date+" "+"%02d:00:00"%i, "sell", 2.2, g_pred[i]-c_pred[i]]
        elif g_pred[i] < c_pred[i]:
            temp=[date+" "+"%02d:00:00"%i, "sell", 2.6, g_pred[i]]
            temp2=[date+" "+"%02d:00:00"%i, "buy", 2.5, c_pred[i]-g_pred[i]]
            data.append(temp2)
        else:
            temp=[date+" "+"%02d:00:00"%i, "buy", 0.0, 0.0]

        # temp=[str(tomorrow)+" "+"%02d:00:00"%i, "", 0.0, 0.0]
        data.append(temp)


    # data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
    #         ["2018-01-01 01:00:00", "sell", 3, 5]]
    output(args.output, data)
