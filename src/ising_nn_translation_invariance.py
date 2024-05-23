import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ising import load_configurations_bin

class IsingTranslationDataset(Dataset):

    def __init__(self, N=10, J=1, B=0, dir="ising_ml_data"):
        filename_data = f"{dir}/N_{N}_J_{J}_B_{B}_data_train.ising"
        filename_labels = f"{dir}/N_{N}_J_{J}_B_{B}_labels_train.csv"

        self.N = N
        self.labels = np.loadtxt(filename_labels, delimiter=",")
        self.labels = self.labels.reshape((self.labels.size, 1))
        self.data = load_configurations_bin(filename_data, flatten=True)

        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def roll_data(self):
        self.data = self.data.numpy()
        self.data = self.data.reshape((self.data.shape[0], self.N, self.N))
        
        for i in range(self.data.shape[0]):
            shift_row, shift_col = np.random.randint(0, self.N), np.random.randint(0, self.N)
            self.data[i] = np.roll(np.roll(self.data[i], shift_row, axis=0), shift_col, axis=1)
        
        self.data = self.data.reshape((self.data.shape[0], self.N**2))
        self.data = torch.tensor(self.data, dtype=torch.float32)

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
    model.load_state_dict(torch.load(f"ising_nn_models/N_{N}_J_1_B_0.pth"))

    training_data = IsingTranslationDataset(N=N, J=J, B=B)
    train_data_loader = DataLoader(training_data, batch_size=1000, shuffle=False)

    learning_rate = 1e-3

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 300
    for t in range(epochs):
        training_data.roll_data()
        print(f"Epoch {t+1}\n---------------------------------")
        train_loop(train_data_loader, model, loss_fn, optimizer)

    print("Done.\nSaving as new model...")
    
    filename = f"ising_nn_models_translation/N_{N}_J_{J}_B_{B}"

    torch.save(model.state_dict(), filename + ".pth")


if __name__ == "__main__":
    main()
