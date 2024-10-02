class Mutant:
    def __init__(self, uid, mutant, mutation_type, layer, neuron):
        self._uid = uid
        self._mutant = mutant  # model changed by mutation operator
        self._mutation_type = mutation_type  # type of mutation operator
        self._layer = layer  # layer number in a model
        self._neuron = neuron  # neuron number in a layer
        self._tuple = None

    def get_uid(self):
        return self._uid

    def get_model(self):
        return self._mutant

    def get_mutation_type(self):
        return self._mutation_type

    def get_layer(self):
        return self._layer

    def get_neuron(self):
        return self._neuron

    def get_tuple(self):
        return self._tuple

    def set_tuple(self, tuple_):
        self._tuple = tuple_
