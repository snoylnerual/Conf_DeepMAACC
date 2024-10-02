from Mutant import Mutant
import numpy as np
import utils

class MutationOperator:
    def __init__(self):
        self._mutant_number = 0
        self._model_utils = utils.ModelUtils()
        self._check = utils.ExaminationalUtils()
        self._MGOp_utils = MutationOperatorsUtils()

    def Gaussian_Fuzzing_Mutator(self, model, clusters, mutation_level='neuron', prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None):
        #TODO: Haven't changed this to new clustering method
        valid_prob_distribution_types = ['normal', 'uniform']
        assert prob_distribution in valid_prob_distribution_types, 'The probability distribution type ' + prob_distribution + ' is not implemented in GF mutation operator'
        if prob_distribution == 'uniform' and ((lower_bound is None) or (upper_bound is None)):
            raise ValueError('In uniform distribution, users are required to specify the lower bound and upper bound of noises')
        if prob_distribution == 'exponential' and (lam is None):
            raise ValueError('In exponential distribution, users are required to specify the lambda value')

        valid_mutation_levels = ['neuron', 'cluster']
        assert mutation_level in valid_mutation_levels, 'The mutation level ' + mutation_level + ' is not implemented in GF mutation operator'

        GF_model = self._model_utils.model_copy(model, 'GF')
        layers = [l for l in GF_model.layers]

        num_of_layers = len(layers)
        self._check.valid_indices_of_mutated_layers_check(num_of_layers, [])
        layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers,[])
        layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]

        if mutation_level == 'cluster':
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                if not (len(weights) == 0):
                    layer_clusters = clusters[index]  # gives the clusters for this layer
                    for ci, cluster in enumerate(layer_clusters):
                        for i, v in enumerate(weights):  # weights for each layer are made up of weights and biases
                            cluster_weights = v[..., cluster]
                            cluster_shape = cluster_weights.shape
                            flat_cluster = cluster_weights.flatten()
                            GF_flat_cluster = self._MGOp_utils.GF_on_list(flat_cluster, prob_distribution, STD,
                                                                          lower_bound, upper_bound, lam)
                            GF_cluster_weights = GF_flat_cluster.reshape(cluster_shape)

                            if i==0:
                                weights[0][..., cluster] = GF_cluster_weights
                            elif i==1:
                                weights[1][cluster] = GF_cluster_weights
                    layer.set_weights(weights)

        else:
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                if not (len(weights) == 0) and layers_should_be_mutated[index]:
                    val = weights[0]
                    bias = weights[1]
                    val_shape = val.shape
                    for i in range(val_shape[1]):
                        neuron = val[..., i]
                        neuron_shape = neuron.shape
                        flat_neuron = neuron.flatten()
                        GF_flat_neuron = self._MGOp_utils.GF_on_list(flat_neuron, prob_distribution, STD,
                                                                     lower_bound, upper_bound, lam)
                        GF_neuron = GF_flat_neuron.reshape(neuron_shape)

                        GF_bias = self._MGOp_utils.GF_on_list([bias[i]], prob_distribution, STD,
                                                              lower_bound, upper_bound, lam)
                        weights[0][..., i] = GF_neuron
                        weights[1][i] = GF_bias
                    layer.set_weights(weights)

        return GF_model

    def Change_Weight_Mutator(self, model, model_name, clusters, mutation_level='neuron', mutation_percent=0.1):
        list_of_mutants = []

        if mutation_level == 'cluster':
            for cluster in clusters:
                CW_model = self._model_utils.model_copy(model, 'CW')
                cluster_indices = np.array(cluster.get_unit_indices())
                layer_index = cluster.get_layer_index()
                layer_name = type(CW_model.layers[layer_index]).__name__
                weights = CW_model.layers[layer_index].get_weights()
                if layer_name == 'Conv2D':
                    weights[0][:, :, :, cluster_indices] *= (mutation_percent+1)
                    weights[1][cluster_indices] *= (mutation_percent + 1)
                elif layer_name == 'Dense':
                    weights[0][:, cluster_indices] *= (mutation_percent+1)
                    weights[1][cluster_indices] *= (mutation_percent + 1)
                else:
                    pass
                CW_model.layers[layer_index].set_weights(weights)
                list_of_mutants.append(Mutant(self._mutant_number, CW_model, 'CW_Cluster', layer_index, cluster_indices))
                self._mutant_number += 1
                # del CW_model
                # clear_session()
                # gc.collect()
                print("CW Mutant number " + str(self._mutant_number))

        elif mutation_level == 'neuron':
            for layer_index, layer in enumerate(model.layers):
                weights_original = layer.get_weights().copy()
                if not (len(weights_original) == 0):  # weights with length of zero shouldn't be edited
                    layer_name = type(layer).__name__
                    CONV2D = layer_name == 'Conv2D'
                    DENSE = layer_name == 'Dense'
                    enum = 0
                    if CONV2D:
                        enum = weights_original[0].shape[3]
                    elif DENSE:
                        enum = weights_original[0].shape[1]
                    else:
                        print("Layer type: " + str(layer_name) + ' (not mutated)')
                        pass
                    for neuron_index in range(enum):
                        CW_model = self._model_utils.model_copy(model, 'CW')
                        weights = CW_model.layers[layer_index].get_weights()
                        if CONV2D:
                            weights[0][:, :, :, neuron_index] *= (mutation_percent + 1)
                            weights[1][neuron_index] *= (mutation_percent + 1)
                        elif DENSE:
                            weights[0][:, neuron_index] *= (mutation_percent + 1)
                            weights[1][neuron_index] *= (mutation_percent + 1)
                        # elif EMBEDDING:
                        #     weights[0][:, neuron_index] *= (mutation_percent + 1)
                        # elif layer_name == 'LSTM':
                        #     weights[0][:, cluster_indices] *= (mutation_percent + 1)
                        #     weights[2][cluster_indices] *= (mutation_percent + 1)
                        CW_model.layers[layer_index].set_weights(weights)
                        list_of_mutants.append(Mutant(self._mutant_number, CW_model, 'CW', layer_index, neuron_index))
                        self._mutant_number += 1
                        # del CW_model
                        # clear_session()
                        # gc.collect()
                        print("CW Mutant number " + str(self._mutant_number))

        return list_of_mutants

    def Neuron_Activation_Inversion_Mutation(self, model, model_name, clusters, mutation_level='neuron'):
        list_of_mutants = []

        if mutation_level == 'cluster':
            for cluster in clusters:
                NAI_model = self._model_utils.model_copy(model, 'NAI')
                cluster_indices = np.array(cluster.get_unit_indices())
                layer_index = cluster.get_layer_index()
                layer_name = type(NAI_model.layers[layer_index]).__name__
                weights = NAI_model.layers[layer_index].get_weights()
                if layer_name == 'Conv2D':
                    weights[0][:, :, :, cluster_indices] *= -1
                elif layer_name == 'Dense':
                    weights[0][:, cluster_indices] *= -1
                else:
                    pass
                NAI_model.layers[layer_index].set_weights(weights)
                list_of_mutants.append(Mutant(self._mutant_number, NAI_model, 'NAI_Cluster', layer_index, cluster_indices))
                self._mutant_number += 1
                print("NAI Mutant number " + str(self._mutant_number))

        elif mutation_level == 'neuron':
            for layer_index, layer in enumerate(model.layers):
                weights_original = layer.get_weights().copy()
                if not (len(weights_original) == 0):  # weights will have length of zero if they shouldn't be edited
                    layer_name = type(layer).__name__
                    CONV2D = layer_name == 'Conv2D'
                    DENSE = layer_name == 'Dense'
                    # EMBEDDING = layer_name == 'Embedding'
                    # LSTM = layer_name == 'LSTM'
                    enum = 0
                    if CONV2D:
                        enum = weights_original[0].shape[3]
                    elif DENSE:
                        enum = weights_original[0].shape[1]
                    else:
                        pass
                    for neuron_index in range(enum):
                        NAI_model = self._model_utils.model_copy(model, 'NAI')
                        weights = NAI_model.layers[layer_index].get_weights()
                        if CONV2D:
                            weights[0][:, :, :, neuron_index] *= -1
                        elif DENSE:
                            weights[0][:, neuron_index] *= -1
                        # elif EMBEDDING:
                        #     weights[0][:, neuron_index] *= -1
                        # elif layer_name == 'LSTM':
                        #     weights[0][:, neuron_index] *= -1
                        NAI_model.layers[layer_index].set_weights(weights)
                        list_of_mutants.append(Mutant(self._mutant_number, NAI_model, 'NAI', layer_index, neuron_index))
                        self._mutant_number += 1
                        print("NAI Mutant number " + str(self._mutant_number))

        return list_of_mutants


    def Neuron_Effect_Blocking_Mutation(self, model, model_name, clusters, mutation_level='neuron'):
        list_of_mutants = []

        if mutation_level == 'cluster':
            for cluster in clusters:
                NEB_model = self._model_utils.model_copy(model, 'NEB')
                cluster_indices = np.array(cluster.get_unit_indices())
                layer_index = cluster.get_layer_index()
                layer_name = type(NEB_model.layers[layer_index]).__name__
                weights = NEB_model.layers[layer_index].get_weights()
                vals_shape = weights[0].shape
                if layer_name == 'Conv2D':
                    #cluster_input_indexes = np.array([num for num in range(vals_shape[2])])
                    weights[0][:, :, :, cluster_indices] = 0
                elif layer_name == 'Dense':
                    weights[0][..., cluster_indices] *= 0
                # elif layer_name == 'Embedding':
                #     weights[0][..., cluster_indices] *= 0
                # elif layer_name == 'LSTM':
                #     weights[0][..., cluster_indices] *= 0
                else:
                    pass
                NEB_model.layers[layer_index].set_weights(weights)
                list_of_mutants.append(Mutant(self._mutant_number, NEB_model, 'NEB_Cluster', layer_index, cluster_indices))
                self._mutant_number += 1
                print("NEB Mutant number " + str(self._mutant_number))

        elif mutation_level == 'neuron':
            for layer_index, layer in enumerate(model.layers):
                weights_original = layer.get_weights().copy()
                if not (len(weights_original) == 0):  # weights will have length of zero if they shouldn't be edited
                    layer_name = type(layer).__name__
                    CONV2D = layer_name == 'Conv2D'
                    DENSE = layer_name == 'Dense'
                    # EMBEDDING = layer_name == 'Embedding'
                    # LSTM = layer_name == 'LSTM'
                    enum = 0
                    if CONV2D:
                        enum = weights_original[0].shape[3]
                    elif DENSE:
                        enum = weights_original[0].shape[1]
                    # elif EMBEDDING:
                    #     enum = weights[0].shape[1]
                    # elif LSTM:
                    #     enum = weights[0].shape[1]
                    else:
                        pass
                    for neuron_index in range(enum):
                        NEB_model = self._model_utils.model_copy(model, 'NEB')
                        weights = NEB_model.layers[layer_index].get_weights()
                        # temp_b = weights[1][neuron_index]
                        for val in weights:
                            val_shape = val.shape
                            if (len(val.shape) != 1):
                                if CONV2D:
                                    input_neuron_indices = [n for n in range(val_shape[2])]
                                    weights[0][:, :, input_neuron_indices, neuron_index] = 0
                                elif DENSE:
                                    input_neuron_indices = [n for n in range(val_shape[0])]
                                    weights[0][input_neuron_indices, neuron_index] = 0
                                # elif EMBEDDING:
                                #     input_neuron_indices = [n for n in range(val_shape[0])]
                                #     weights[0][input_neuron_indices, neuron_index] = 0
                                # elif layer_name == 'LSTM':
                                #     weights[0][input_neuron_indices, neuron_index] = 0
                                NEB_model.layers[layer_index].set_weights(weights)
                                list_of_mutants.append(Mutant(self._mutant_number, NEB_model, 'NEB', layer_index, neuron_index))
                                self._mutant_number += 1
                                print("NEB Mutant number " + str(self._mutant_number))

        return list_of_mutants


class MutationOperatorsUtils:
    def __init__(self):
        self.LD_mut_candidates = ['Dense']
        self.LAm_mut_candidates = ['Dense']

    def GF_on_list(self, lst, prob_distribution, STD, lower_bound, upper_bound, lam):
        copy_lst = lst.copy()

        if prob_distribution == 'normal':
            copy_lst += np.random.normal(scale=STD, size=len(copy_lst))
        elif prob_distribution == 'uniform':
            copy_lst += np.random.uniform(low=lower_bound, high=upper_bound, size=len(copy_lst))
        elif prob_distribution == 'exponential':
            assert lam != 0
            scale = 1 / lam
            copy_lst += np.random.exponential(scale=scale, size=len(copy_lst))
        else:
            pass

        return copy_lst

    def cluster_splitting(self, clusters, num_of_layers):
        # nums of layers, list of number of neurons
        # [#, [#,#,#]]
        new_cluster_array = []
        c = 0
        temp_arr = []
        for v in clusters:
            n = v.get_layer_index()
            if n == c:
                temp_arr.append(np.array(v.get_unit_indices()))
            else:
                if len(temp_arr) != 0:
                    new_cluster_array.append(temp_arr)
                temp_arr = [np.array(v.get_unit_indices())]
            c = n

        if len(temp_arr) != 0:
            new_cluster_array.append(temp_arr)
        return new_cluster_array


class MutationGenerator:

    def __init__(self, model_name, model, clusters, mutation_level):
        self._model_name = model_name
        self._model = model
        self._MO = MutationOperator()
        self._MGOpUtils = MutationOperatorsUtils()
        self._clusters = clusters  # self._MGOpUtils.cluster_splitting(clusters, len(model.layers))
        self._mutation_level = mutation_level
        self._mutation_percent = 0.1
        self._consumers = []

    def set_mutation_percent(self, mutation_percent):
        self._mutation_percent = mutation_percent


    def get_mutations(self, mutator_list):
        mutated_models = []
        print("Mutation model: " + self._model_name)

        if 'GF' in mutator_list:
            GF_model = self._MO.Gaussian_Fuzzing_Mutator(self._model, self._clusters, self._mutation_level,
                                                         prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None)
            GF_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            mutated_models.append(GF_model)
        if 'CW' in mutator_list:
            CW_models = self._MO.Change_Weight_Mutator(self._model, self._model_name, self._clusters, self._mutation_level, self._mutation_percent)
            for CW_mutant in CW_models:
                CW_model = CW_mutant.get_model()
                CW_model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                mutated_models.append(CW_mutant)
        if 'NEB' in mutator_list:
            NEB_models = self._MO.Neuron_Effect_Blocking_Mutation(self._model, self._model_name, self._clusters, self._mutation_level)
            for NEB_mutant in NEB_models:
                NEB_model = NEB_mutant.get_model()
                NEB_model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
                mutated_models.append(NEB_mutant)
        if 'NAI' in mutator_list:
            NAI_models = self._MO.Neuron_Activation_Inversion_Mutation(self._model, self._model_name, self._clusters, self._mutation_level)
            for NAI_mutant in NAI_models:
                NAI_model = NAI_mutant.get_model()
                NAI_model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
                mutated_models.append(NAI_mutant)
        return mutated_models
