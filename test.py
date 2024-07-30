import numpy as np
import tensorflow as tf
from fits_data_generator import FITSDataGenerator
import json
import argparse
from astropy.io import fits
import os

def load_config(config_path):
    """
    Load the configuration from a JSON file.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_model(model_path):
    """
    Load a trained TensorFlow model from a given path.
    """
    return tf.keras.models.load_model(model_path)

def prepare_test_data_generator(config):
    """
    Prepare a data generator for the test dataset using the configuration settings.
    Automatically determines the number of test data files.

    Args:
        config (dict): Configuration settings that include paths and data parameters.
    Returns:
        FITSDataGenerator: Data generator for the test dataset.
    """
    test_data_dir = config['test_data_dir']
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.fits')]
    test_indices = np.arange(len(test_files))  # Use the number of files instead of total_test_data
    batch_size = config['batch_size']
    return FITSDataGenerator(test_data_dir, test_data_dir, test_indices, batch_size, shuffle=False)

def save_images(original_data, denoised_data, output_dir, residual_dir, batch_index):
    """
    Save the denoised and residual images to specified directories.
    
    Args:
        original_data (numpy.array): Original noisy FITS data.
        denoised_data (numpy.array): Output from the DAE model.
        output_dir (str): Directory to save denoised images.
        residual_dir (str): Directory to save residual images.
        batch_index (int): Current batch index to name the files uniquely.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(residual_dir):
        os.makedirs(residual_dir)

    for i in range(original_data.shape[0]):
        denoised_image = denoised_data[i].squeeze()
        residual_image = original_data[i].squeeze() - denoised_image

        fits.writeto(os.path.join(output_dir, f'denoised_{batch_index}_{i}.fits'), denoised_image, overwrite=True)
        fits.writeto(os.path.join(residual_dir, f'residual_{batch_index}_{i}.fits'), residual_image, overwrite=True)

def evaluate_and_save(model, data_generator, output_dir, residual_dir):
    """
    Evaluate the model and save the output denoised and residual images.
    
    Args:
        model (tf.keras.Model): The trained model to evaluate.
        data_generator (FITSDataGenerator): Generator providing test data batches.
        output_dir (str): Path to save denoised images.
        residual_dir (str): Path to save residual images.
    """
    for i, (noisy, clean) in enumerate(data_generator):
        denoised = model.predict(noisy)
        save_images(noisy, denoised, output_dir, residual_dir, i)

    results = model.evaluate(data_generator)
    print(f"Test Loss, Test Accuracy: {results}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the trained DAE model on test data and save outputs using a configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)
    model = load_model(config['model_path'])
    test_generator = prepare_test_data_generator(config)
    evaluate_and_save(model, test_generator, config['output_dir'], config['residual_dir'])
