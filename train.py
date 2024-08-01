import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from fits_data_generator import FITSDataGenerator
from dae_model import DAE
import json
import argparse

def load_config(config_path):
    """
    Load the training configuration from a JSON file.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def create_model(config):
    """
    Create and compile the DAE model using settings from the config.
    """
    model = DAE(
        input_shape=config['input_shape'],
        encoder_filters=config['encoder_filters'],
        decoder_filters=config['decoder_filters'],
        encoder_kernel_sizes=config['encoder_kernel_sizes'],
        decoder_kernel_sizes=config['decoder_kernel_sizes'],
        pooling_type=config['pooling_type'],
        pooling_size=config['pooling_size'],
        activation=config['activation'],
        alpha=config['alpha']).model
    optimizer = Adam(learning_rate=config['initial_learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_absolute_error')
    return model

def setup_data_generators(noisy_dir, clean_dir, batch_size, total_data_count, val_split):
    """
    Setup the data generators for training and validation datasets.

    Args:
        noisy_dir (str): Directory with noisy FITS images.
        clean_dir (str): Directory with clean FITS images.
        batch_size (int): Number of images per batch.
        total_data_count (int): Total number of images available.
        val_split (float): Fraction of the data to be used for validation.

    Returns:
        tuple: Returns the training and validation data generators.
    """
    total_indices = np.arange(total_data_count)
    np.random.shuffle(total_indices)
    split_index = int(len(total_indices) * (1 - val_split))
    train_indices = total_indices[:split_index]
    val_indices = total_indices[split_index:]

    train_generator = FITSDataGenerator(noisy_dir, clean_dir, train_indices, batch_size, shuffle=True)
    val_generator = FITSDataGenerator(noisy_dir, clean_dir, val_indices, batch_size, shuffle=False)
    return train_generator, val_generator

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model using configuration file.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration JSON file.')
    args = parser.parse_args()

    config = load_config(args.config)
    train_generator, val_generator = setup_data_generators(
        config['noisy_dir'],
        config['clean_dir'],
        config['batch_size'],
        config['total_data_count'],
        config['val_split'])

    model = create_model(config)

    pretrained_model_path = config.get('pretrained_model_path')
    if pretrained_model_path:
        model.load_weights(pretrained_model_path)
        print(f"Loaded pretrained model weights from {pretrained_model_path}")

    callbacks = []

    if config['use_early_stopping']: 
        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True)
        callbacks.append(early_stopping_monitor)

    if config['save_intermediate_models']: 
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
          filepath=config['checkpoint_dir'] + 'model_{epoch:03d}.h5',
          save_freq='epoch',
          save_best_only=False,
          verbose=1)
        callbacks.append(model_checkpoint_callback)

    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config['checkpoint_dir'] + 'best_model.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1)
    callbacks.append(best_checkpoint_callback)

    # Start model training.
    train_history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['n_epochs'],
        callbacks=callbacks)

    # Save training history for further analysis.
    np.save(config['checkpoint_dir'] + 'training_history.npy', train_history.history)
