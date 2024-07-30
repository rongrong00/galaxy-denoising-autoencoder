import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling2D, Activation, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class DAE:
    """
    Class for creating a Denoising Autoencoder (DAE) model.

        Attributes
        ----------
        input_shape (tuple): The shape of the input data
        encoder_filters (list): Number of filters for conv layers in the encoder.
        decoder_filters (list): Number of filters for conv layers in the decoder.
        encoder_kernel_sizes (list of tuples): Kernel sizes for conv layers in the encoder.
        decoder_kernel_sizes (list of tuples): Kernel sizes for conv layers in the decoder.
        pooling_type (str): Type of pooling to use (either'max' or 'average').
        pooling_size (tuple): The size of the pooling window.
        activation (str): The activation function used after each conv layer
        alpha (float or None): The alpha value for LeakyReLU activation (if used)
    """
    def __init__(self, input_shape, encoder_filters, decoder_filters, encoder_kernel_sizes, decoder_kernel_sizes,
                 pooling_type, pooling_size, activation, alpha):
        self.input_shape = input_shape
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.encoder_kernel_sizes = encoder_kernel_sizes
        self.decoder_kernel_sizes = decoder_kernel_sizes
        self.pooling_type = pooling_type
        self.pooling_size = pooling_size
        self.activation = activation
        self.alpha = alpha
        self.model = self.build_model()

    def get_activation_layer(self, activation, alpha=None):
        """
        Create an activation layer based on the specified type and parameters.
        """
        if activation == 'leaky_relu':
            return LeakyReLU(alpha=alpha if alpha is not None else 0.1)
        else:
            return Activation(activation)

    def add_encoder_block(self, model, filters, kernel_size, input_shape = None, use_pooling=True):
        """
        Adds an encoding block to the model. Each encoder block consists of a Conv2D layer followed by an activation layer
        and then a pooling layer (optional)
        
        Args:
            model (Sequential): The model to which the encoder block is being added.
            filters (int): Number of filters in this Conv2D layer.
            kernel_size (tuple): Size of the kernel for this Conv2D layer.
            use_pooling (bool): Use pooling or not in this block
        """
        if input_shape != None:
            model.add(Conv2D(filters, kernel_size, padding='same', input_shape=input_shape))
        else:
            model.add(Conv2D(filters, kernel_size, padding='same'))
        model.add(self.get_activation_layer(self.activation, alpha=self.alpha))
        if use_pooling:
            if self.pooling_type.lower() == 'max':
                model.add(MaxPooling2D(self.pooling_size, padding='same'))
            elif self.pooling_type.lower() == 'average':
                model.add(AveragePooling2D(self.pooling_size, padding='same'))

    def add_decoder_block(self, model, filters, kernel_size):
        """
        Adds a decoding block to the model Each decoder block consists of an UpSampling2D layer followed by a Conv2D layer
        and an activation layer.
        
        Args:
            model (Sequential): The model to which the decoder block is being added.
            filters (int): Number of filters in this Conv2D layer.
            kernel_size (tuple): Size of the kernel for this Conv2D layer.
        """
        model.add(UpSampling2D(self.pooling_size))
        model.add(Conv2D(filters, kernel_size, padding='same'))
        model.add(self.get_activation_layer(self.activation, alpha=self.alpha))

    def build_model(self):
        """
        Builds the autoencoder model given parameters
        
        Returns:
            Sequential: The complete autoencoder model.
        """
        model = Sequential()
        # Add encoder blocks to the model
        self.add_encoder_block(model, self.encoder_filters[0], self.encoder_kernel_sizes[0], input_shape=tuple(self.input_shape))
        
        for filters, kernel_size in zip(self.encoder_filters[1:], self.encoder_kernel_sizes[1:]):
            self.add_encoder_block(model, filters, kernel_size)
        # Add ecoder blocks to the model
        for filters, kernel_size in zip(self.decoder_filters, self.decoder_kernel_sizes):
            self.add_decoder_block(model, filters, kernel_size)
        model.add(Conv2D(1, (3, 3), activation='linear', padding='same'))
        return model