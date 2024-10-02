from mutation_generator import MutationGenerator
from unit_clusterer import UnitClustering
from mutation_score import MutationScore
from keras.models import load_model
from network import Dataset
import numpy as np
import pandas as pd
import time

from one_by_one import OBO, MiniMutant
from keras.backend import clear_session
import utils
import gc


class DMAACC:
    def __init__(self):
        self._model_filename = None
        self._model = None
        self._mutation_level = 'neuron'
        self._mutator_list = ['CW', 'NAI', 'NEB']
        self._standard_deviation = 0.2
        self._cluster_size = 5
        self._one_unit_per_cluster = False
        self._threshold = 2
        self._mutation_percent = 0.1
        self._dataset = None
        self._PH_threshold = 0.5

    def load_model(self, model_filename):
        self._model_filename = model_filename
        self._model = load_model(model_filename)

    def get_model(self):
        return self._model

    def set_mutation_level(self, mutation_level):
        self._mutation_level = mutation_level

    def get_mutation_level(self, mutation_level):
        self._mutation_level = mutation_level

    def set_mutation_percent(self, mutation_percent):
        self._mutation_percent = mutation_percent

    def set_mutator_list(self, mutator_list):
        self._mutator_list = mutator_list

    def set_standard_deviation(self, std):
        self._standard_deviation = std

    def set_cluster_size(self, cluster_size):
        self._cluster_size = cluster_size

    def set_one_unit_per_cluster(self, u):
        self._one_unit_per_cluster = u

    def set_cluster_selection_threshold(self, threshold):
        self._threshold = threshold

    def set_dataset(self, dataset):
        self._dataset = dataset

    def set_ParHAC_threshold(self, threshold):
        self._PH_threshold = threshold

    def __get_outputs_count(self):
        return self._model.layers[-1].units

    def run_vanilla(self):
        df_vanilla = pd.DataFrame(
            columns=['Model_Type', 'Dataset', 'Mutation_Level', 'Mutate_time',
                     'Number_of_Mutants', 'Mutation_Score', 'MS_time', 'Total_time'])
        start = time.time()
        m_start = time.time()
        mg = MutationGenerator(self._model_filename, self._model, [], self._mutation_level)
        mg.set_mutation_percent(self._mutation_percent)
        mutations = mg.get_mutations(self._mutator_list)  # this is M'
        m_end = time.time()
        self._dataset = Dataset(self._model_filename.split('-')[1].split('.')[0])  # this is T
        ms_start = time.time()
        ms = MutationScore(self._model_filename.split('.')[0], self._model, mutations, self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)
        ms.run()
        ms_end = time.time()
        df_vanilla.loc[len(df_vanilla.index)] = [self._model_filename, self._dataset.get_dataset_name(),
                             self._mutation_level, m_end - m_start, len(mutations),
                             ms.get_mutation_score(), ms_end - ms_start, ms_end - start]

        del mg, mutations, ms
        clear_session()
        gc.collect()

        return df_vanilla

    def run_approach_1(self):
        df_clusters = pd.DataFrame(
            columns=['Model_Type', 'Dataset', 'Mutation_Level', 'Mutate_time',
                     'Number_of_Mutants', 'Number_of_Clusters', 'Clusters_per_layer',
                     'Max_Cluster_Sz', 'Min_Cluster_Sz', 'Mean_Cluster_Sz',
                     'Cluster_time', 'Mutation_Score', 'MS_time', 'Total_time']) # Clusters_per_layer -> Neurons_per_Cluster_param
        clusters = []
        start = time.time()
        c_start = 0
        c_end = 0
        unit_clustering = UnitClustering(self._model)
        if self._mutation_level == 'cluster':
            c_start = time.time()
            clusters = unit_clustering.get_clusters(self._cluster_size)
            c_end = time.time()
            for index, mu in enumerate(clusters):
               print('layer '+str(mu.get_layer_index())+' cluster '+str(index)+' '+str(mu.get_unit_indices()))
        m_start = time.time()
        mg = MutationGenerator(self._model_filename, self._model, clusters, self._mutation_level)
        mg.set_mutation_percent(self._mutation_percent)
        mutations = mg.get_mutations(self._mutator_list)  # this is M'
        # self.model is the original model
        m_end = time.time()
        self._dataset = Dataset(self._model_filename.split('-')[1].split('.')[0])  # this is T
        ms_start = time.time()
        ms = MutationScore(self._model_filename.split('.')[0], self._model, mutations, self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)
        ms.run()
        ms_end = time.time()
        print('Mutation Score: ' + str(ms.get_mutation_score()))
        print('Number of mutations: ' + str(len(mutations)))
        print('Average size of clusters per layer: ' + str(self._cluster_size))
        df_clusters.loc[len(df_clusters.index)] = [self._model_filename, self._dataset.get_dataset_name(),
                           self._mutation_level, m_end - m_start, len(mutations), len(clusters),
                           self._cluster_size, unit_clustering.get_max_cluster_size(),
                           unit_clustering.get_min_cluster_size(), unit_clustering.get_mean_cluster_size(),
                           c_end - c_start, ms.get_mutation_score(), ms_end - ms_start, (c_end-c_start)+(ms_end-ms_start)+(m_end-start)]

        del unit_clustering, clusters, mg, mutations, ms
        clear_session()
        gc.collect()

        return df_clusters

    def run_approach_2(self):
        start = time.time()
        mg = MutationGenerator(self._model_filename, self._model, [], 'neuron')
        mg.set_mutation_percent(self._mutation_percent)
        mutations = mg.get_mutations(self._mutator_list)
        m_end = time.time()
        self._dataset = Dataset(self._model_filename.split('-')[1].split('.')[0])

        ms = MutationScore(self._model_filename.split('.')[0], self._model, mutations, self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)

        if 'cluster' in self._mutation_level:
            df_cluster = pd.DataFrame(columns=['Model_Type', 'Dataset', 'Mutable_Layers', 'Mutation_Level',
                                               'Mutate_time', 'Number_of_Mutants', 'ParHAC_Threshold',
                                               'Number_of_Clusters', 'Max_Cluster_Sz', 'Min_Cluster_Sz',
                                               'Mean_Cluster_Sz', 'Cluster_time',
                                               'Mutation_Score', 'MS_time', 'Total_time'])

            unit_clustering = UnitClustering(self._model)
            c_start = time.time()
            graph_clusters = unit_clustering.get_graph_clusters(mutations, self._PH_threshold)
            c_end = time.time()
            ms_start = time.time()
            ms.set_clusters(graph_clusters)
            ms.cluster_run()
            ms_end = time.time()
            df_cluster.loc[len(df_cluster.index)] = [self._model_filename, self._dataset.get_dataset_name(), len(graph_clusters),
                               'cluster', m_end-start, len(mutations), self._PH_threshold, ms.get_cluster_amount(),
                               unit_clustering.get_max_cluster_size(), unit_clustering.get_min_cluster_size(),
                               unit_clustering.get_mean_cluster_size(), c_end-c_start, ms.get_mutation_score(),
                               ms_end-ms_start, (c_end-c_start)+(ms_end-ms_start)+(m_end-start)]

        del unit_clustering, graph_clusters, mg, mutations, ms
        clear_session()
        gc.collect()

        return df_cluster

    def run_one_by_one_v(self):
        obo = OBO(self._model_filename, self._model, self._dataset, self._dataset.get_dataset_name(), 'neuron')
        nb_classes = self._dataset.get_nb_classes()
        ms = MutationScore(self._model_filename.split('.')[0], self._model, [], self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)

        mutant_layer_dict = {}
        m_time = 0
        ms_time = 0

        original_x, original_y = ms.get_correct_test_points()

        for layer_index, layer in enumerate(self._model.layers):
            weights = layer.get_weights().copy()

            if not (len(weights) == 0):  # weights with length of zero shouldn't be edited
                layer_name = type(layer).__name__
                CONV2D = layer_name == 'Conv2D'
                DENSE = layer_name == 'Dense'
                enum = 0
                if CONV2D:
                    enum = weights[0].shape[3]
                elif DENSE:
                    enum = weights[0].shape[1]
                else:
                    print("Layer type: " + str(layer_name) + ' (not mutated)')
                    pass
                for neuron_index in range(enum):
                    for mo_type in ['CW', 'NAI', 'NEB']:
                        m_start = time.time()
                        self._model = obo.mutate_one(self._model, layer_name, layer_index, neuron_index, mo_type,
                                                     self._mutation_percent)
                        m_end = time.time()
                        m_time += m_end - m_start
                        t = tuple((layer_index, neuron_index))
                        ms.set_mutations([self])
                        ms_start = time.time()
                        ms.run_obo(original_x, original_y)
                        killed_classes = ms.get_killed_classes()
                        ms_end = time.time()
                        ms_time += ms_end - ms_start
                        if layer_index in mutant_layer_dict:
                            mutant_layer_dict[layer_index] += [
                                MiniMutant(t, layer_index, neuron_index, killed_classes, mo_type)]
                        else:
                            mutant_layer_dict[layer_index] = [
                                MiniMutant(t, layer_index, neuron_index, killed_classes, mo_type)]
                        # resets
                        layer.set_weights(weights)

        ms_mutants_kc = 0
        amt = 0
        mutation_len = 0
        for key, value in mutant_layer_dict.items():
            amt += len(value)
            for minimutant in value:
                ms_mutants_kc += minimutant.get_killed_classes()

        mutation_score_n = ms_mutants_kc / (amt * self._dataset.get_nb_classes())
        print('Mutation Score' + str(mutation_score_n))


        df_vanilla = pd.DataFrame(
            columns=['Model_Type', 'Dataset', 'Mutation_Level', 'Mutate_time',
                     'Number_of_Mutants', 'Mutation_Score', 'MS_time', 'Total_time'])
        df_vanilla.loc[len(df_vanilla.index)] = [self._model_filename, self._dataset.get_dataset_name(),
                                                 self._mutation_level, m_time, amt,
                                                 mutation_score_n, ms_time, m_time + ms_time]

        del obo, ms, original_x, original_y, weights, layer, mutant_layer_dict
        clear_session()
        gc.collect()

        return df_vanilla

    def run_one_by_one_a1(self):
        obo = OBO(self._model_filename, self._model, self._dataset, self._dataset.get_dataset_name(), 'cluster')
        ms = MutationScore(self._model_filename.split('.')[0], self._model, [], self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)

        mutant_layer_dict = {}
        mutant_num = 0
        mutation_score = 0
        killed_classes = 0
        m_time = 0
        c_time = 0
        ms_time = 0

        original_x, original_y = ms.get_correct_test_points()

        unit_clustering = UnitClustering(self._model)
        c_start = time.time()
        clusters = unit_clustering.get_clusters(self._cluster_size)
        c_end = time.time()
        c_time = c_end - c_start

        for cluster in clusters:
            li = cluster.get_layer_index()
            weights = self._model.layers[li].get_weights().copy()
            for mo_type in ['CW', 'NAI', 'NEB']:
                m_start = time.time()
                obo.mutate_cluster(self._model, type(self._model.layers[li]).__name__, li, cluster, mo_type, self._mutation_percent)
                m_end = time.time()
                m_time += m_end - m_start
                t = tuple((li, cluster))
                ms.set_mutations([self])
                ms_start = time.time()
                ms.run_obo(original_x, original_y)
                killed_classes = ms.get_killed_classes()
                ms_end = time.time()
                ms_time += ms_end - ms_start
                if li in mutant_layer_dict:
                    mutant_layer_dict[li] += [
                        MiniMutant(t, li, cluster, killed_classes, mo_type)]
                else:
                    mutant_layer_dict[li] = [
                        MiniMutant(t, li, cluster, killed_classes, mo_type)]
                # resets
                self._model.layers[li].set_weights(weights)

        ms_mutants_kc = 0
        amt = 0
        mutation_len = 0
        for key, value in mutant_layer_dict.items():
            amt += len(value)
            for minimutant in value:
                ms_mutants_kc += minimutant.get_killed_classes()

        mutation_score_n = ms_mutants_kc / (amt * self._dataset.get_nb_classes())
        print('Mutation Score' + str(mutation_score_n))

        df_clusters = pd.DataFrame(
            columns=['Model_Type', 'Dataset', 'Mutation_Level', 'Mutate_time',
                     'Number_of_Mutants', 'Number_of_Clusters', 'Neurons_per_Cluster_param',
                     'Max_Cluster_Sz', 'Min_Cluster_Sz', 'Mean_Cluster_Sz',
                     'Cluster_time', 'Mutation_Score', 'MS_time', 'Total_time'])
        df_clusters.loc[len(df_clusters.index)] = [self._model_filename, self._dataset.get_dataset_name(),
                                                   self._mutation_level, m_time, amt, len(clusters),
                                                   self._cluster_size, unit_clustering.get_max_cluster_size(),
                                                   unit_clustering.get_min_cluster_size(),
                                                   unit_clustering.get_mean_cluster_size(),
                                                   c_time, mutation_score_n, ms_time,
                                                   m_time + c_time + ms_time]

        del obo, ms, original_x, original_y, unit_clustering, clusters, weights, cluster, mutant_layer_dict
        clear_session()
        gc.collect()

        return df_clusters


    def run_one_by_one_a2(self):
        obo = OBO(self._model_filename, self._model, self._dataset, self._dataset.get_dataset_name(), 'neuron')
        nb_classes = self._dataset.get_nb_classes()
        ms = MutationScore(self._model_filename.split('.')[0], self._model, [], self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)

        original_x, original_y = ms.get_correct_test_points()

        mutant_layer_dict = {}
        mutant_num = 0
        mutation_score = 0
        killed_classes = 0
        m_time = 0
        c_time = 0
        ms_time = 0
        start = time.time()
        #TODO: if the process cant handle more than one model, then I will have to change this...
        for layer_index, layer in enumerate(self._model.layers):
            weights = layer.get_weights().copy()

            if not (len(weights) == 0):  # weights with length of zero shouldn't be edited
                layer_name = type(layer).__name__
                CONV2D = layer_name == 'Conv2D'
                DENSE = layer_name == 'Dense'
                # EMBEDDING = layer_name == 'Embedding'
                # LSTM = layer_name == 'LSTM'
                enum = 0
                if CONV2D:
                    enum = weights[0].shape[3]
                elif DENSE:
                    enum = weights[0].shape[1]
                # elif EMBEDDING:
                #     enum = weights[0].shape[1]
                # elif LSTM:
                #     enum = weights[0].shape[1]
                else:
                    print("Layer type: " + str(layer_name) + ' (not mutated)')
                    pass
                for neuron_index in range(enum):
                    for mo_type in ['CW', 'NAI', 'NEB']:
                        m_start = time.time()
                        obo.mutate_one(self._model, layer_name, layer_index, neuron_index, mo_type, self._mutation_percent)
                        m_end = time.time()
                        m_time += m_end - m_start
                        t = (layer_index, neuron_index,) + tuple(
                            self._model.layers[layer_index].get_weights()[0][..., neuron_index].flatten(), ) + tuple(
                            self._model.layers[layer_index].get_weights()[1][neuron_index].flatten(), )
                        ms.set_mutations([self])
                        ms_start = time.time()
                        ms.run_obo(original_x, original_y)
                        killed_classes = ms.get_killed_classes()
                        ms_end = time.time()
                        if layer_index in mutant_layer_dict:
                            mutant_layer_dict[layer_index] += [MiniMutant(t, layer_index, neuron_index, killed_classes, mo_type, ms_end - ms_start)]
                        else:
                            mutant_layer_dict[layer_index] = [MiniMutant(t, layer_index, neuron_index, killed_classes, mo_type, ms_end - ms_start)]
                        # resets
                        layer.set_weights(weights)

        c_start = time.time()
        g_clusters = obo.get_one_graph_clusters(mutant_layer_dict, self._PH_threshold)
        c_end = time.time()
        c_time += c_end - c_start

        ms_mutants_kc = []
        ms_times = []
        mutant_cluster_lengths = []
        for layer_cluster_list, mutant_list in zip(g_clusters, mutant_layer_dict.values()):
            for cluster in layer_cluster_list:
                m = np.random.choice(np.array(cluster).flatten())
                m = mutant_list[m]
                ms_mutants_kc += [m.get_killed_classes()*len(cluster)]
                mutant_cluster_lengths += [len(cluster)]
                ms_times += [m.get_ms_time()]


        mutation_score_n = sum(ms_mutants_kc) / (sum(mutant_cluster_lengths) * self._dataset.get_nb_classes())
        print('Mutation Score' + str(mutation_score_n))

        mutation_len = 0
        for l in mutant_layer_dict.values():
            mutation_len += len(l)

        df_cluster = pd.DataFrame(columns=['Model_Type', 'Dataset', 'Mutable_Layers', 'Mutation_Level',
                                           'Mutate_time', 'Number_of_Mutants', 'ParHAC_Threshold',
                                           'Number_of_Clusters', 'Max_Cluster_Sz', 'Min_Cluster_Sz',
                                           'Mean_Cluster_Sz', 'Cluster_time',
                                           'Mutation_Score', 'MS_time', 'Total_time'])
        df_cluster.loc[len(df_cluster.index)] = [self._model_filename, self._dataset.get_dataset_name(),
                                                 len(g_clusters),
                                                 'cluster', m_time, mutation_len, self._PH_threshold,
                                                 obo.get_cluster_amount(),
                                                 obo.get_max_cluster_size(),
                                                 obo.get_min_cluster_size(),
                                                 obo.get_mean_cluster_size(), c_time,
                                                 mutation_score_n,
                                                 ms_time, m_time+c_time+ms_time]

        del obo, ms, original_x, original_y, g_clusters, layer, weights, cluster, mutant_layer_dict, mutant_list, m
        clear_session()
        gc.collect()

        return df_cluster




