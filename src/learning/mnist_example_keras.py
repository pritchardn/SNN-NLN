import numpy as np
import optuna
import tensorflow as tf
import wandb
from keras import layers
from optuna.trial import TrialState
from tensorflow import keras
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

device = 'cuda' if tf.test.is_gpu_available() else 'cpu'
WANDB_ACTIVE = False


def load_data(batch_size=64):
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)
    train_y = keras.utils.to_categorical(train_y)
    test_y = keras.utils.to_categorical(test_y)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset


def create_model(input_shape):
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])


class OptunaTerminateCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self._trial = trial

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        return_accuracy = logs.get('val_accuracy')
        self._trial.report(return_accuracy, epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned()


def train_model(current_trial: optuna.Trial, model: keras.Model, dataset, test_dataset, loss_fn,
                optimizer, epochs, batch_size):
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    callbacks = [OptunaTerminateCallback(current_trial)]
    if WANDB_ACTIVE:
        callbacks.append(WandbMetricsLogger())
        callbacks.append(WandbModelCheckpoint("models"))
    history = model.fit(dataset, epochs=epochs, batch_size=batch_size, validation_data=test_dataset,
                        callbacks=callbacks, verbose=1)
    return history


def test_model(model: keras.Model, dataset):
    score = model.evaluate(dataset, verbose=0)
    return score[1], score[0]


def train(optuna_trial: optuna.Trial, model, train_data, test_data, loss_fn, optimizer, epochs=5,
          batch_size=64):
    history = train_model(optuna_trial, model, train_data, test_data, loss_fn, optimizer, epochs,
                          batch_size)
    return_accuracy = history.history['val_accuracy'][-1]
    return return_accuracy


def run_trial(trial: optuna.Trial):
    config_vals = {"learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
                   "epochs": trial.suggest_int("epochs", 10, 20, 2),
                   "batch_size": trial.suggest_int("batch_size", 32, 128, 32),
                   "optimizer": trial.suggest_categorical("optimizer",
                                                          ["Adam", "SGD", "RMSprop"])}
    if WANDB_ACTIVE:
        wandb.init(project="mnist-example-keras",
                   config=config_vals)
    train_data, test_data = load_data(batch_size=config_vals["batch_size"])
    model = create_model((28, 28, 1))
    optimizer = keras.optimizers.get(config_vals["optimizer"])
    optimizer.learning_rate.assign(config_vals["learning_rate"])
    accuracy = train(trial, model, train_data, test_data, 'categorical_crossentropy',
                     optimizer, epochs=config_vals["epochs"])
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

    print("Best optuna_trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
