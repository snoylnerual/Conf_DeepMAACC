import ctypes
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np



class OBO:
    def __init__(self, model_name, model, dataset, dataset_name, mutation_level):
        self._model_name = model_name
        self._model = model
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._mutation_level = mutation_level
        self._num_of_classes = self._model.layers[-1].units
        self._correct_test_points = None
        self._mutation_score = None
        self._clusters = None
        self._cluster_amount = None
        self._max_cluster_size = None
        self._min_cluster_size = None
        self._mean_cluster_size = None
        self._mutant_number = 0

    def get_cluster_amount(self):
        return self._cluster_amount

    def get_max_cluster_size(self):
        return self._max_cluster_size

    def get_min_cluster_size(self):
        return self._min_cluster_size

    def get_mean_cluster_size(self):
        return self._mean_cluster_size

    def mutate_one(self, model, layer_name, layer_index, neuron_index, mo_type, mutation_percent):
        weights = model.layers[layer_index].get_weights()

        if mo_type == 'CW':
            if layer_name == 'Conv2D':
                weights[0][:, :, :, neuron_index] *= (mutation_percent + 1)
                weights[1][neuron_index] *= (mutation_percent + 1)
            elif layer_name == 'Dense':
                weights[0][:, neuron_index] *= (mutation_percent + 1)
                weights[1][neuron_index] *= (mutation_percent + 1)
        elif mo_type == 'NAI':
            if layer_name == 'Conv2D':
                weights[0][:, :, :, neuron_index] *= -1
            elif layer_name == 'Dense':
                weights[0][:, neuron_index] *= -1
        elif mo_type == 'NEB':
            for val in weights:
                val_shape = val.shape
                if (len(val.shape) != 1):
                    if layer_name == 'Conv2D':
                        input_neuron_indices = [n for n in range(val_shape[2])]
                        weights[0][:, :, input_neuron_indices, neuron_index] = 0
                    elif layer_name == 'Dense':
                        input_neuron_indices = [n for n in range(val_shape[0])]
                        weights[0][input_neuron_indices, neuron_index] = 0

        model.layers[layer_index].set_weights(weights)
        self._mutant_number += 1
        print(mo_type + "Mutant number " + str(self._mutant_number))
        return model


    def mutate_cluster(self, model, layer_name, layer_index, cluster, mo_type, mutation_percent):
        weights = model.layers[layer_index].get_weights()
        cluster_indices = np.array(cluster.get_unit_indices())

        if mo_type == 'CW':
            if layer_name == 'Conv2D':
                weights[0][:, :, :, cluster_indices] *= (mutation_percent + 1)
                weights[1][cluster_indices] *= (mutation_percent + 1)
            elif layer_name == 'Dense':
                weights[0][:, cluster_indices] *= (mutation_percent + 1)
                weights[1][cluster_indices] *= (mutation_percent + 1)
        elif mo_type == 'NAI':
            if layer_name == 'Conv2D':
                weights[0][:, :, :, cluster_indices] *= -1
            elif layer_name == 'Dense':
                weights[0][:, cluster_indices] *= -1
        elif mo_type == 'NEB':
            if layer_name == 'Conv2D':
                weights[0][:, :, :, cluster_indices] = 0
            elif layer_name == 'Dense':
                weights[0][..., cluster_indices] *= 0

        model.layers[layer_index].set_weights(weights)
        self._mutant_number += 1
        print(mo_type + "Mutant number " + str(self._mutant_number))

    def get_one_graph_clusters(self, mutant_layer_dict, threshold):
        #list_of_clusters = []
        amounts = []
        end_n = 0
        start_n = 0
        nodes = []
        weights = []
        for layer_number, mutant_list in mutant_layer_dict.items():
            end_n += len(mutant_list)  # mutations is just a long list
            print('Adding mutants ' + str(start_n) + ' to ' + str(end_n))
            for i in range(start_n, end_n):
                a = mutant_list[i-start_n]
                for j in range(i + 1, end_n):
                    b = mutant_list[j-start_n]
                    nodes.append(i)
                    nodes.append(j)
                    weights.append(distance.euclidean(a.get_tuple(), b.get_tuple())) # removed []
            amounts.append(start_n)
            start_n = end_n
        amounts.append(end_n+1)
        weights = np.array(weights).reshape(-1, 1)
        list_of_clusters = self.do_clustering(nodes, (MinMaxScaler()).fit_transform(weights), threshold)

        cluster_size_list = []
        layer_cluster_list = []
        offset = amounts.pop(0)
        for cluster_indices in list_of_clusters:
            print(cluster_indices)
            temp = []
            if cluster_indices[0] >= amounts[0]:
                offset = amounts.pop(0)
                layer_cluster_list += [temp]
                temp = []
            for i in range(len(cluster_indices)):
                cluster_indices[i] -= offset
            temp += [cluster_indices]
            cluster_size_list.append(len(cluster_indices))

        self._cluster_amount = len(cluster_size_list)
        self._max_cluster_size = max(cluster_size_list)
        self._min_cluster_size = min(cluster_size_list)
        self._mean_cluster_size = sum(cluster_size_list) / len(cluster_size_list)
        print(layer_cluster_list)

        return layer_cluster_list

    def do_clustering(self, edges, weights, threshold):
        libquickstart = ctypes.CDLL('libquickstart.so')  # '../dms-codebase/dms/lib/clustering.dylib')#
        libquickstart.do_clustering.restype = ClusterArray
        libquickstart.do_clustering.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                ctypes.POINTER(ctypes.c_float),
                                                ctypes.c_int,
                                                ctypes.c_float]
        len_edges = len(edges)
        len_weights = len(weights)
        if len_weights != len_edges // 2:
            raise ValueError('Incompatible array lengths')
        rs = libquickstart.do_clustering((ctypes.c_int * len_edges)(*edges),
                                         (ctypes.c_float * len_weights)(*weights),
                                         len_edges,
                                         threshold)
        m = [node_array_to_list(rs.clusters[i]) for i in range(rs.length)]
        return m

def node_array_to_list(node_array):
    return [node_array.nodes[i] for i in range(node_array.length)]

class NodeArray(ctypes.Structure):
    _fields_ = [("nodes", ctypes.POINTER(ctypes.c_int)),
                ("length", ctypes.c_int)]

class ClusterArray(ctypes.Structure):
    _fields_ = [("clusters", ctypes.POINTER(NodeArray)),
                ("length", ctypes.c_int)]

class MiniMutant:
    def __init__(self, tup, ln, nn, kc, mo_type, ms_time):
        self._tuple = tup
        self._layer_num = ln
        self._neuron_num = nn
        self._killed_classes = kc
        self._mo_type = mo_type
        self._ms_time = ms_time

    def get_tuple(self):
        return self._tuple

    def get_layer_num(self):
        return self._layer_num

    def get_neuron_num(self):
        return self._neuron_num

    def get_killed_classes(self):
        return self._killed_classes

    def get_ms_time(self):
        return self._ms_time
