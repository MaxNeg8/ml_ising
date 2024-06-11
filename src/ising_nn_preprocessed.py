import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ising import load_configurations_bin
from preprocessing import preprocess

class IsingPreprocessedDataset(Dataset):

    def __init__(self, train, N=10, J=1, B=0, dir="ising_ml_data"):
        filename_data = f"{dir}/N_{N}_J_{J}_B_{B}_data_" + ("train" if train else "test") + ".ising"
        filename_labels = f"{dir}/N_{N}_J_{J}_B_{B}_labels_" + ("train" if train else "test") + ".csv"

        self.labels = np.loadtxt(filename_labels, delimiter=",")
        self.labels = self.labels.reshape((self.labels.size, 1))
        print("Loading configurations...")
        self.data = load_configurations_bin(filename_data, flatten=False)
        
        print("Preprocessing configurations...")
        configs = []
        done = 0
        for config in self.data:
            configs.append(preprocess(config).flatten())
            done += 1
            print(f"Progress: {done/self.data.shape[0]*100:0.2f}%\r", end="")
        self.data = np.array(configs)
        print("\nDone.")
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class IsingNNModel(nn.Module):

    def __init__(self, N):
        super(IsingNNModel, self).__init__()

        self.architecture = nn.Sequential(
            nn.Linear(N**2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        self.N = N

    def forward(self, x):
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

def main():
    N = 25
    J = 1
    B = 0

    model = IsingNNModel(N=N)

    training_data = IsingPreprocessedDataset(train=True, N=N, J=J, B=B)
    train_data_loader = DataLoader(training_data, batch_size=1000, shuffle=False)

    learning_rate = 1e-3

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 500
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------------")
        train_loop(train_data_loader, model, loss_fn, optimizer)

    print("Done.\nSaving as new model...")
    
    filename = f"ising_nn_models_preprocessed/N_{N}_J_{J}_B_{B}"

    torch.save(model.state_dict(), filename + ".pth")

    testing_data = IsingPreprocessedDataset(train=False, N=N, J=J, B=B)
    test_data_loader = DataLoader(testing_data, batch_size=1, shuffle=True)

    with torch.no_grad():
        temp_init, temp_pred = [], []
        for X, y in test_data_loader:
            pred = model(X)

            temp_init.append(y.numpy().flatten()[0])
            temp_pred.append(pred.numpy().flatten()[0])

        np.savetxt(f"{filename}.csv", np.vstack([temp_init, temp_pred]).T, delimiter=",", header="temp_init,temp_pred")


if __name__ == "__main__":
    main()
