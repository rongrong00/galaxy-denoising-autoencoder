import numpy as np
import tensorflow as tf
from fits_data_generator import FITSDataGenerator
import json
import argparse
from astropy.io import fits
import os

def prepare_test_data_generator(config):
    """
    Prepare a data generator for the test dataset using the configuration settings.
    Automatically determines the number of test data files.

    Args:
        config (dict): Configuration settings that include paths and data parameters.
    Returns:
        FITSDataGenerator: Data generator for the test dataset.
        list: List of filenames in the test dataset.
    """
    test_data_dir = config['test_data_dir']
    test_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.fits')])
    test_indices = np.arange(len(test_files))  # Use the number of files instead of total_test_data
    batch_size = config['batch_size']
    return FITSDataGenerator(test_data_dir, test_data_dir, test_indices, batch_size, shuffle=False), test_files

def save_images(original_data, denoised_data, output_dir, residual_dir, filenames):
    """
    Save the denoised and residual images to specified directories using original filenames.

    Args:
        original_data (numpy.array): Original noisy FITS data.
        denoised_data (numpy.array): Output from the DAE model.
        output_dir (str): Directory to save denoised images.
        residual_dir (str): Directory to save residual images.
        filenames (list): List of original filenames for each image in the batch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(residual_dir):
        os.makedirs(residual_dir)

    for i, filename in enumerate(filenames):
        denoised_image = denoised_data[i].squeeze()
        residual_image = original_data[i].squeeze() - denoised_image

        fits.writeto(os.path.join(output_dir, f'denoised_{filename}'), denoised_image, overwrite=True)
        fits.writeto(os.path.join(residual_dir, f'residual_{filename}'), residual_image, overwrite=True)

def evaluate_and_save(model, data_generator, output_dir, residual_dir, all_filenames):
    """
    Evaluate the model and save the output denoised and residual images using original filenames.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        data_generator (FITSDataGenerator): Generator providing test data batches.
        output_dir (str): Path to save denoised images.
        residual_dir (str): Path to save residual images.
        all_filenames (list): List of all filenames in the test dataset.
    """
    batch_size = data_generator.batch_size
    for i, (noisy, _) in enumerate(data_generator):
        batch_start = i * batch_size
        batch_end = batch_start + noisy.shape[0]
        batch_filenames = all_filenames[batch_start:batch_end]
        denoised = model.predict(noisy)
        save_images(noisy, denoised, output_dir, residual_dir, batch_filenames)

    results = model.evaluate(data_generator)
    print(f"Test Loss: {results}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the trained DAE model on test data and save outputs using a configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration file.')
    args = parser.parse_args()

    # Load config and model directly here
    with open(args.config, 'r') as file:
        config = json.load(file)
    model = tf.keras.models.load_model(config['model_path'])

    test_generator, test_filenames = prepare_test_data_generator(config)
    evaluate_and_save(model, test_generator, config['output_dir'], config['residual_dir'], test_filenames)
