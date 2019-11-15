from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        # an affine operation: y = Wx +b
        self.fc1 = nn.Linear(in_features=16*6*6, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # max pool with a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))

        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        # flatten x
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    net = Net()
    print(net)
    input = torch.randn(1, 1, 32, 32)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()

    import torch.optim as optim

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in training loop:
    optimizer.zero_grad()
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)
    output = net(input)
    loss = criterion(output, target=target)
    loss.backward()
    optimizer.step()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)