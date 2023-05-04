import json

import nengo
import nengo_dl
import numpy as np
import optuna
import tensorflow as tf
from optuna.trial import TrialState
from tensorflow import keras


def load_data(num_steps=30):
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x.reshape((train_x.shape[0], -1))
    train_x = train_x[:, None, :]
    train_y = train_y[:, None, None]
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_x = np.tile(test_x[:, None, :], (1, num_steps, 1))
    test_y = np.tile(test_y[:, None, None], (1, num_steps, 1))
    return train_x, train_y, test_x, test_y


def create_model(input_shape, num_outputs=10, amplitude=0.01):
    with nengo.Network() as net:
        nengo_dl.configure_settings(stateful=False)
        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None
        neuron_type = nengo.LIF(amplitude=amplitude)
        input_layer = nengo.Node(np.zeros(input_shape[0] * input_shape[1]))

        # add the first convolutional layer
        x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(
            input_layer, shape_in=input_shape
        )
        x = nengo_dl.Layer(neuron_type)(x)

        # add the second convolutional layer
        x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(
            x, shape_in=(26, 26, 32)
        )
        x = nengo_dl.Layer(neuron_type)(x)

        # add the third convolutional layer
        x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(
            x, shape_in=(12, 12, 64)
        )
        x = nengo_dl.Layer(neuron_type)(x)

        # linear readout
        out = nengo_dl.Layer(tf.keras.layers.Dense(units=num_outputs))(x)

        # we'll create two different output probes, one with a filter
        # (for when we're simulating the network over time and
        # accumulating spikes), and one without (for when we're
        # training the network using a rate-based approximation)
        out_p = nengo.Probe(out, label="out_p")
        out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
    return net, out_p, out_p_filt


def train(model, train_x, train_y, train_probe, loss_fn, optimizer, epochs, batch_size,
          training=True):
    with nengo_dl.Simulator(model, minibatch_size=batch_size) as sim:
        if training:
            sim.compile(loss={train_probe: loss_fn},
                        optimizer=optimizer)
            history = sim.fit(train_x, {train_probe: train_y}, epochs=epochs)
            sim.save_params("./mnist_params")
            return history.history


def classification_accuracy(y_true, y_pred):
    return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])


def run_trial(trial: optuna.Trial):
    config_vals = {"learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
                   "epochs": trial.suggest_int("epochs", 2, 14, 2),
                   "batch_size": trial.suggest_int("batch_size", 50, 400, 50),
                   "optimizer": trial.suggest_categorical("optimizer",
                                                          ["Adam", "SGD", "RMSprop"]),
                   "num_steps": trial.suggest_int("num_steps", 10, 50, 10),
                   "amplitude": trial.suggest_float("amplitude", 0.01, 0.1)}
    input_shape = (28, 28, 1)
    train_x, train_y, test_x, test_y = load_data(num_steps=config_vals["num_steps"])
    model, train_probe, test_probe = create_model(input_shape, amplitude=config_vals["amplitude"])
    sim = nengo_dl.Simulator(model, minibatch_size=config_vals["batch_size"])
    optimizer = keras.optimizers.get(config_vals["optimizer"])
    optimizer.learning_rate.assign(config_vals["learning_rate"])
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    sim.compile(loss={test_probe: classification_accuracy})
    print(
        "Accuracy before training:",
        sim.evaluate(test_x, {test_probe: test_y}, verbose=1)["loss"],
    )

    history = train(model, train_x, train_y, train_probe, loss_fn, optimizer, config_vals['epochs'],
                    config_vals['batch_size'])
    sim.compile(loss={test_probe: classification_accuracy})
    final_accuracy = sim.evaluate(test_x, {test_probe: test_y}, verbose=1)["loss"]
    print("Final accuracy:", final_accuracy)
    sim.close()
    print(history)
    return final_accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(run_trial, n_trials=20)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned_out = {}
    complete_out = {}
    best_out = {}
    for trial in pruned_trials:
        pruned_out[trial.number] = {k: v for k, v in trial.params.items()}
        pruned_out[trial.number]["value"] = trial.value
    for trial in complete_trials:
        complete_out[trial.number] = {k: v for k, v in trial.params.items()}
        complete_out[trial.number]["value"] = trial.value
    best_out[study.best_trial.number] = {k: v for k, v in study.best_trial.params.items()}
    best_out[study.best_trial.number]["value"] = study.best_trial.value
    with open('nengo_study.json', 'w') as f:
        json.dump({"pruned": pruned_out, "complete": complete_out, "best": best_out}, f)
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
