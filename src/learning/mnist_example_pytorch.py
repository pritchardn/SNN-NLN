"""
A very simple MNIST example (end to end) in order to play with the basics
"""
import optuna
import torch
import torch.nn as nn
import wandb
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
WANDB_ACTIVE = False


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def load_data(batch_size=64) -> (DataLoader, DataLoader):
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def create_model():
    return SimpleNetwork().to(device)


def train_model(model, dataloader: DataLoader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(X)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return correct / len(dataloader.dataset), running_loss / len(dataloader.dataset)


def test_model(model, dataloader, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def train(trial: optuna.Trial, model, train_data, test_data, loss_fn, optimizer, epochs=5):
    return_accuracy = 0.0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_acc, train_loss = train_model(model, train_data, loss_fn, optimizer)
        test_acc, test_loss = test_model(model, test_data, loss_fn)
        return_accuracy = test_acc
        if WANDB_ACTIVE:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss,
                       "test_acc": test_acc})
        trial.report(test_acc, t)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return return_accuracy


def run_trial(trial: optuna.Trial):
    config_vals = {"learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
                   "epochs": trial.suggest_int("epochs", 10, 20, 2),
                   "batch_size": trial.suggest_int("batch_size", 32, 128, 32),
                   "optimizer": trial.suggest_categorical("optimizer",
                                                          ["Adam", "SGD", "RMSprop"])}
    if WANDB_ACTIVE:
        wandb.init(project="mnist-example",
                   config=config_vals)
    train_data, test_data = load_data(batch_size=config_vals["batch_size"])
    model = create_model()
    loss_fn = nn.CrossEntropyLoss()
    if WANDB_ACTIVE:
        wandb.watch(model, log="all")
    optimizer = getattr(torch.optim, config_vals["optimizer"])(model.parameters(),
                                                               lr=config_vals["learning_rate"])
    accuracy = train(trial, model, train_data, test_data, loss_fn, optimizer,
                     epochs=config_vals["epochs"])
    if WANDB_ACTIVE:
        wandb.finish()
    return accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(run_trial, n_trials=10)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
