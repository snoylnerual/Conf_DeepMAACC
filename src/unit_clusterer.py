import numpy as np
from keras.layers import Dense, Conv2D
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.preprocessing import normalize
from utils import model_ok, is_softmax_classifier

import ctypes
from scipy.spatial import distance

def make_axes(w):
    return tuple(range(len(w.shape) - 1))

class UnitClustering:
    def __init__(self, model):
        if not model_ok(model):
            raise ValueError('An instance of Sequential or Model is expected')
        if not is_softmax_classifier(model):
            raise ValueError('A classifier with softmax output is expected')
        self._model = model
        self._threshold = 0.5
        self._max_cluster_size = None
        self._min_cluster_size = None
        self._mean_cluster_size = None

    def get_max_cluster_size(self):
        return self._max_cluster_size

    def get_min_cluster_size(self):
        return self._min_cluster_size

    def get_mean_cluster_size(self):
        return self._mean_cluster_size

    def set_threshold(self, threshold):
        self._threshold = threshold

    def get_clusters(self, cluster_sz):
        if cluster_sz <= 0:
            raise ValueError('Cluster size must be a positive number')
        clusters = []
        cluster_size_list = []
        for layer_index in range(len(self._model.layers) - 1):  # exclude last layer
            layer = self._model.layers[layer_index]

            if not isinstance(layer, Dense) and not isinstance(layer, Conv2D):
                continue

            w = layer.get_weights()
            a = w[0]
            b = w[1]
            b_expanded = np.expand_dims(b, axis=make_axes(a))
            b_tiled = np.tile(b_expanded, reps=(*a.shape[:-2], 1, 1))
            points = np.concatenate((a, b_tiled), axis=-2)
            points = points.transpose()
            points = points.reshape((points.shape[0], -1))
            points = (MinMaxScaler()).fit_transform(points)


            num_units = w[0].shape[-1]
            num_clusters = int(np.ceil(float(num_units) / float(cluster_sz)))
            clustering = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(points)
            cluster_index_map = dict()
            for unit_index in range(len(points)):
                cluster = clustering[unit_index]  # what is the cluster index of the jth neuron
                if cluster not in cluster_index_map:
                    cluster_index_map[cluster] = []
                cluster_index_map[cluster].append(unit_index)

            t_len = len(clusters)
            for (_, unit_indices) in cluster_index_map.items():
                clusters.append(MutableUnitCluster(self._model, layer_index, unit_indices))
            cluster_size_list += [len(clusters) - t_len]


        self._max_cluster_size = max(cluster_size_list)
        self._min_cluster_size = min(cluster_size_list)
        self._mean_cluster_size = sum(cluster_size_list) / len(cluster_size_list)
        return clusters


    def get_graph_clusters(self, mutations, threshold):

        mutant_layer_dict = {}
        for mut in mutations:  # mutant type in a list of mutants
            layer = mut.get_layer()
            neuron = mut.get_neuron()
            t = (layer, neuron, ) + tuple(mut.get_model().layers[layer].get_weights()[0][...,neuron].flatten(),) + tuple(mut.get_model().layers[layer].get_weights()[1][neuron].flatten(),)
            mut.set_tuple(t)
            if layer in mutant_layer_dict:
                mutant_layer_dict[layer] += [mut]
            else:
                mutant_layer_dict[layer] = [mut]
        list_of_clusters = []
        for layer_number, mutant_list in mutant_layer_dict.items():
            n = len(mutant_list)  # mutations is just a long list
            print('Clustering %d mutants...' % n)
            nodes = []
            weights = []
            for i in range(n):
                a = mutant_list[i]
                for j in range(i + 1, n):
                    b = mutant_list[j]
                    nodes.append(i)
                    nodes.append(j)
                    weights.append(distance.euclidean(a.get_tuple(), b.get_tuple())) # removed []

        weights = weights.reshape(-1, 1)
        list_of_clusters += [[self.do_clustering(nodes, (MinMaxScaler()).fit_transform(weights), threshold)]]

        cluster_size_list = []
        list_of_mutant_clusters = []
        for layer_cluster_list, mutant_list in zip(list_of_clusters, mutant_layer_dict.values()):
            temp = []
            for layer_cluster in layer_cluster_list:
                temp += [np.array(mutant_list)[np.array(layer_cluster)]]
                cluster_size_list += [len(layer_cluster)]
            list_of_mutant_clusters += temp

        self._max_cluster_size = max(cluster_size_list)
        self._min_cluster_size = min(cluster_size_list)
        self._mean_cluster_size = sum(cluster_size_list) / len(cluster_size_list)

        return list_of_mutant_clusters

    def get_graph_clusters_once(self, mutations, threshold):

        mutant_layer_dict = {}
        for mut in mutations:  # mutant type in a list of mutants
            layer = mut.get_layer()
            neuron = mut.get_neuron()
            t = (layer, neuron, ) + tuple(mut.get_model().layers[layer].get_weights()[0][...,neuron].flatten(),) + tuple(mut.get_model().layers[layer].get_weights()[1][neuron].flatten(),)
            mut.set_tuple(t)
            if layer in mutant_layer_dict:
                mutant_layer_dict[layer] += [mut]
            else:
                mutant_layer_dict[layer] = [mut]
        list_of_clusters = []
        amounts = []
        end_n = 0
        start_n = 0
        for layer_number, mutant_list in mutant_layer_dict.items():
            end_n += len(mutant_list)  # mutations is just a long list
            print('Clustering mutants' + str(start_n) + ' to ' + str(end_n))
            nodes = []
            weights = []
            for i in range(start_n, end_n):
                a = mutant_list[i]
                for j in range(i + 1, end_n):
                    b = mutant_list[j]
                    nodes.append(i)
                    nodes.append(j)
                    weights.append(distance.euclidean(a.get_tuple(), b.get_tuple())) # removed []
            list_of_clusters += [[self.do_clustering(nodes, (MinMaxScaler()).fit_transform(weights), threshold)]]
            start_n = end_n
            amounts.append(end_n)

        cluster_size_list = []
        list_of_mutant_clusters = []
        for layer_cluster_list, mutant_list in zip(list_of_clusters, mutant_layer_dict.values()):
            temp = []
            for layer_cluster in layer_cluster_list:
                temp += [np.array(mutant_list)[np.array(layer_cluster)]]
                cluster_size_list += [len(layer_cluster)]
            list_of_mutant_clusters += temp

        self._max_cluster_size = max(cluster_size_list)
        self._min_cluster_size = min(cluster_size_list)
        self._mean_cluster_size = sum(cluster_size_list) / len(cluster_size_list)

        return list_of_mutant_clusters



    def do_clustering(self, edges, weights, threshold):
        libquickstart = ctypes.CDLL('lib/libquickstart.so')  # '../dms-codebase/dms/lib/clustering.dylib')#
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


class MutableUnitCluster:
    def __init__(self, model, layer_index, unit_indices):
        self._model = model
        if len(unit_indices) == 0:
            raise ValueError('A cluster must contain at least one unit')
        self._unit_indices = unit_indices
        self._layer_index = layer_index
        self._layer = model.layers[layer_index]
        if not isinstance(self._layer, Dense) and not isinstance(self._layer, Conv2D):
            raise ValueError('Only Dense or Conv2D layers can be in a mutable cluster')
        self._original_weights_dict = dict()
        w = self._layer.get_weights()
        for i in unit_indices:
            if i in self._original_weights_dict:
                raise ValueError('Duplicate unit index')
            self._original_weights_dict[i] = (w[0][..., i] * 1.0, w[1][i])


    def add(self, fraction):
        if fraction < -1 or fraction > 1:
            raise ValueError('A fraction value in interval [-1, 1] is required')
        w = self._layer.get_weights()
        for i in self._unit_indices:
            #print(i)
            #print(w[0][..., i].shape)
            w[0][..., i] += w[0][..., i] * fraction  # input weights
            w[1][i] += w[1][i] * fraction  # biases
        self._layer.set_weights(w)
        return self._model


    def reset(self):
        w = self._layer.get_weights()
        for i in self._unit_indices:
            w[0][..., i] = self._original_weights_dict[i][0] * 1.0  # reset weights
            w[1][i] = self._original_weights_dict[i][1]  # reset biases
        self._layer.set_weights(w)
        return self._model


    def get_model(self):
        return self._model


    def get_layer_index(self):
        return self._layer_index


    def get_unit_indices(self):
        return self._unit_indices