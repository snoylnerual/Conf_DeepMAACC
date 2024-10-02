import numpy as np
import keras
import random, math
from keras import Sequential, Model
from keras.utils import to_categorical

def is_softmax_classifier(model):
    output_layer = model.layers[-1]
    return hasattr(output_layer, 'activation') and output_layer.activation.__name__ == 'softmax'

def model_ok(model):
    return isinstance(model, Sequential) or isinstance(model, Model)

class ModelUtils:

    def __init__(self):
        pass

    def model_copy(self, model, mode=''):
        original_layers = [l for l in model.layers]
        #suffix = '_copy_' + mode
        new_model = keras.models.clone_model(model)
        for index, layer in enumerate(new_model.layers):
            original_layer = original_layers[index]
            original_weights = original_layer.get_weights()
            #layer.name = layer.name + suffix
            layer.set_weights(original_weights)
        #new_model.name = new_model.name + suffix
        return new_model

    def get_booleans_of_layers_should_be_mutated(self, num_of_layers, indices):
        if indices == None:
            booleans_for_layers = np.full(num_of_layers, True)
        else:
            booleans_for_layers = np.full(num_of_layers, False)
            for index in indices:
                booleans_for_layers[index] = True
        return booleans_for_layers


class ExaminationalUtils:

    def __init__(self):
        pass

    def valid_indices_of_mutated_layers_check(self, num_of_layers, indices):
        if indices is not None:
            for index in indices:
                assert index >= 0, 'Index should be positive'
                assert index < num_of_layers, 'Index should not be out of range, where index should be smaller than ' + str(num_of_layers)
                pass 

    def in_suitable_indices_check(self, suitable_indices, indices):
        if indices is not None:
            for index in indices:
                assert index in suitable_indices, 'Index ' + str(index) + ' is an invalid index for this mutation'
                pass 