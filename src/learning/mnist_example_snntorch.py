import numpy as np
import optuna
import snntorch as snn
import torch
import torch.nn as nn
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float
WANDB_ACTIVE = True


def load_data(batch_size=64) -> (DataLoader, DataLoader):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))
    ])
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)
    return train_dataloader, test_dataloader


class SimpleNetwork(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def print_batch_accuracy(model, batch_size, data, targets, train=False):
    output, _ = model(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")


def train_printer(model, batch_size, epoch, iter_counter, counter, loss_hist, test_loss_hist, data,
                  targets, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(model, batch_size, data, targets, train=True)
    print_batch_accuracy(model, batch_size, test_data, test_targets, train=False)
    print("\n")


def old_main():
    batch_size = 64

    train_data, test_data = load_data()
    model = SimpleNetwork(28 * 28, 1000, 10, 0.95, 25).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0

    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_data)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            model.train()
            spk_rec, mem_rec = model(data.view(batch_size, -1))

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(model.num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                model.eval()
                test_data_example, test_targets = next(iter(test_data))
                test_data_example = test_data_example.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = model(test_data_example.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(model.num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer(model, batch_size, epoch, iter_counter, counter, loss_hist,
                                  test_loss_hist, data, targets, test_data_example, test_targets)
                counter += 1
                iter_counter += 1


def train_model(model, train_data, loss_fn, optimizer, batch_size, dtype):
    running_loss = 0.0
    running_accuracy = 0.0
    for batch, (x, y) in enumerate(train_data):
        x = x.to(device)
        y = y.to(device)

        # forward pass
        model.train()
        spk_rec, mem_rec = model(x.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(model.num_steps):
            loss_val += loss_fn(mem_rec[step], y)
        running_loss += loss_val.item() * len(x)
        output, _ = model(x.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        running_accuracy += np.mean((y == idx).detach().cpu().numpy())

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss_val.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_data.dataset):>5d}]")

    return running_accuracy / len(train_data.dataset), running_loss / len(train_data.dataset)


def test_model(model, test_data, loss_fn, batch_size):
    size = len(test_data.dataset)
    num_batches = len(test_data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            test_spk, test_mem = model(x.view(batch_size, -1))
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(model.num_steps):
                test_loss += loss_fn(test_mem[step], y).item()
            output, _ = model(x.view(batch_size, -1))
            _, idx = output.sum(dim=0).max(1)
            correct += np.mean((y == idx).detach().cpu().numpy())
    test_loss = test_loss.detach().cpu().numpy()[0] / num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def train(trial: optuna.Trial, model, train_data, test_data, loss_fn, optimizer, epochs,
          batch_size):
    return_accuracy = 0.0
    for t in range(epochs):
        train_acc, train_loss = train_model(model, train_data, loss_fn, optimizer, batch_size,
                                            dtype)
        test_acc, test_loss = test_model(model, test_data, loss_fn, batch_size)
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
        wandb.init(project="mnist-example-snntorch",
                   config=config_vals)
    train_data, test_data = load_data(batch_size=config_vals["batch_size"])
    model = SimpleNetwork(28 * 28, 1000, 10, 0.95, 25).to(device)
    loss_fn = nn.CrossEntropyLoss()
    if WANDB_ACTIVE:
        wandb.watch(model, log="all")
    optimizer = getattr(torch.optim, config_vals["optimizer"])(model.parameters(),
                                                               lr=config_vals["learning_rate"])
    accuracy = train(trial, model, train_data, test_data, loss_fn, optimizer,
                     batch_size=config_vals["batch_size"], epochs=config_vals["epochs"])
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
