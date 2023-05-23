import inspect
import os
import shutil

from models import Model

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser


def attempt_conversion(model_name: str):
    path_working_directory = os.path.abspath("./src/conversion")
    configparser = import_configparser()
    config = configparser.ConfigParser()

    config['paths'] = {
        'path_wd': path_working_directory,
        'dataset_path': path_working_directory,
        'filename_ann': model_name,
    }

    config['tools'] = {
        'evaluate_ann': False,  # Test ANN on dataset before conversion.
        'normalize': True  # Normalize weights for full dynamic range.
    }

    config['simulation'] = {
        'simulator': 'INI',  # Chooses execution backend of SNN toolbox.
        'duration': 50,  # Number of time steps to run each sample.
        'num_to_test': 100,  # How many test samples to run.
        'batch_size': 50,  # Batch size for simulation.
        'keras_backend': 'tensorflow'  # Which keras backend to use.
    }

    config['input'] = {
        'model_lib': 'pytorch'  # Input model is defined in pytorch.
    }

    config['output'] = {
        'plot_vars': {  # Various plots (slows down simulation).
            'spiketrains',  # Leave section empty to turn off plots.
            'spikerates',
            'activations',
            'correlation',
            'v_mem',
            'error_t'}
    }

    config_filepath = os.path.join(path_working_directory, 'config')
    with open(config_filepath, 'w') as configfile:
        config.write(configfile)

    source_path = inspect.getfile(Model)
    source_file = os.path.join(path_working_directory, model_name + '.py')
    print(source_path)
    shutil.copyfile(source_path, os.path.join(path_working_directory, model_name + '.py'))
    # The Main Event
    main(config_filepath)


if __name__ == "__main__":
    attempt_conversion("pytorch_autoencoder")
