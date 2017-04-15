from data.constants import image_size, num_labels
from data.text_load import vocabulary_size

LogisticRegressionConfiguration = {
    'name': 'logistic_regression',
    'image_size': image_size,
    'final_layer': {
        'input_depth': image_size * image_size,
        'output_depth': num_labels
    }
}

ConvolutionalNeuralNetworkConfiguration = {
    'name': 'convolution_3_layer',
    'convolutional_layers': [
        {
            'patch_size': 5,
            'input_depth': 1,
            'output_depth': 8,
            'biases_init_value': 0.0,
        },
        {
            'patch_size': 5,
            'input_depth': 8,
            'output_depth': 16,
            'biases_init_value': 1.0,
        },
        {
            'patch_size': 5,
            'input_depth': 16,
            'output_depth': 32,
            'biases_init_value': 1.0,
        },
    ],
    'fully_connected_layers': [
        {
            'input_depth': 512,
            'output_depth': 265,
        },
    ],
    'final_layer': {
        'input_depth': 265,
        'output_depth': num_labels,
    },
    'image_size': image_size,
}


NeuralNetworkConfiguration = {
    'name': 'neural_network_1_layer',
    'hidden_layers': [
        {
            'input': image_size * image_size,
            'output': 1024
        }
    ],
    'final_layer': {
        'input': 1024,
        'output': num_labels
    },
    'image_size': image_size,
}

SkipGramConfiguration = {
    'name': 'word_2_vec',
    'valid_size': 16,
    'valid_window': 100,
    'vocabulary_size': vocabulary_size,
    'embedding_size': 128,
}