import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ising import load_configurations_bin
from preprocessing import preprocess

from matplotlib import pyplot as plt

class IsingDataset(Dataset):

    def __init__(self, train, N=10, J=1, B=0, dir="ising_ml_data"):
        filename_data = f"{dir}/N_{N}_J_{J}_B_{B}_data_" + ("train" if train else "test") + ".ising"
        filename_labels = f"{dir}/N_{N}_J_{J}_B_{B}_labels_" + ("train" if train else "test") + ".csv"

        self.labels = np.loadtxt(filename_labels, delimiter=",")
        self.labels = self.labels.reshape((self.labels.size, 1))
        self.data = load_configurations_bin(filename_data, flatten=True)
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class IsingNNModel(nn.Module):

    def __init__(self, N, preprocessing=False):
        super(IsingNNModel, self).__init__()

        self.architecture = nn.Sequential(
            nn.Linear(N**2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        self.N = N
        self.preprocessing = preprocessing

    def forward(self, x):
        if self.preprocessing:
            x = preprocess((x.numpy().reshape((self.N, self.N)) + 1) / 2)
            x = torch.tensor(x.flatten()*2 - 1, dtype=torch.float32)
        return self.architecture(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f} [{batch+100:5d}/{size:>5d}]")

    model.eval()

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += torch.all(((pred - y)/y).abs() < torch.tensor([[0.05]])).item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Correct: {100*correct:0.2f}%, Avg loss: {test_loss:>8f}\n")

def main():
    N = 10
    J = 1
    B = 0

    training_data = IsingDataset(train=True, N=N, J=J, B=B)
    testing_data = IsingDataset(train=False, N=N, J=J, B=B)

    train_data_loader = DataLoader(training_data, batch_size=1000, shuffle=True)
    test_data_loader = DataLoader(testing_data, batch_size=1, shuffle=True)

    model = IsingNNModel(N=N)

    learning_rate = 1e-3

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 500
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------------")
        train_loop(train_data_loader, model, loss_fn, optimizer)
        if t % 25 == 0:
            test_loop(test_data_loader, model, loss_fn)

    print("Done.\nSaving...")
    
    filename = f"ising_nn_models/N_{N}_J_{J}_B_{B}"

    torch.save(model.state_dict(), filename + ".pth")

    with torch.no_grad():
        temp_init, temp_pred = [], []
        for X, y in test_data_loader:
            pred = model(X)

            temp_init.append(y.numpy().flatten()[0])
            temp_pred.append(pred.numpy().flatten()[0])

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        ax.plot(temp_init, temp_init, color="red", label="f(x) = x")
        ax.scatter(temp_init, temp_pred, label="Results")

        ax.set_xlabel("Correct temperature")
        ax.set_ylabel("Predicted temperature")
        ax.set_title("Confusion of model")

        ax.legend()


        np.savetxt(f"{filename}.csv", np.vstack([temp_init, temp_pred]).T, delimiter=",", header="temp_init,temp_pred")
        
        plt.show()

if __name__ == "__main__":
    main()
