class LayerParameter:

    def __init__(self):
        self.weights = None
        self.biases = None

    def eval(self):
        return {
            'biases': self.biases.eval(),
            'weights': self.weights.eval()
        }


class ModelParameters:

    def __init__(self):
        self.tf_train_dataset = None
        self.tf_train_labels = None
        self.convolutional_layers = []
        self.hidden_layers = []
        self.fully_connected_layers = []
        self.final_layer = LayerParameter()
