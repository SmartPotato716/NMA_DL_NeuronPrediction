""" Created on Fri Jul 19 15:09:42 2024
    @author: dcupolillo """

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(
            self,
            ncomp,
            NN1,
            NN2,
            bidi=True
    ) -> None:

        super(Net, self).__init__()

        self.rnn = nn.RNN(NN1, ncomp, num_layers=1, dropout=0,
                          bidirectional=bidi, nonlinearity='tanh')
        self.fc = nn.Linear(ncomp, NN2)

    def forward(self, x):

        y = self.rnn(x)[0]

        if self.rnn.bidirectional:
            q = (y[:, :, :ncomp] + y[:, :, ncomp:])/2

        else:
            q = y

        z = F.softplus(self.fc(q), 10)

        return z, q

# CNN that first predicts basal from visual and then has another layer
# that predicts mouse's choice
# from basal ganlgia data
# Conv2D
# plotting the loss