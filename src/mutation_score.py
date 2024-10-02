import numpy as np
import random

class MutationScore:
    def __init__(self, model_name, model, mutations, dataset, dataset_name, mutation_level):
        self._model_name = model_name
        self._model = model
        self._mutations = mutations
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._mutation_level = mutation_level
        self._num_of_classes = self._model.layers[-1].units
        self._correct_test_points = None
        self._mutation_score = None
        self._killed_classes = None
        self._clusters = None
        self._cluster_amount = None

    def get_mutation_score(self):
        return self._mutation_score

    def get_killed_classes(self):
        return self._killed_classes

    def get_cluster_amount(self):
        return self._cluster_amount

    def set_mutations(self, mutations):
        self._mutations = mutations

    def set_clusters(self, clusters):
        self._clusters = clusters

    def get_correct_test_points(self):  # , dataset):
        x = self._dataset.get_x_test()
        y = self._dataset.get_y_test()

        x_new = []
        y_new = []

        predictions = self._model.predict(x)

        for i, row in enumerate(predictions):
            if np.argmax(y[i]) == np.argmax(row):
                x_new += [x[i]]
                y_new += [y[i]]

        return x_new, y_new

    def killed_classes(self):
        classes_list = [1] * self._num_of_classes
        sum = 0
        for m_prime in self._mutations:
            predictions = m_prime.get_model().predict(np.array(self._correct_test_points[0]))
            for i, predicted_label in enumerate(predictions):
                if np.argmax(self._correct_test_points[1][i]) != np.argmax(predicted_label):
                    classes_list[np.argmax(predicted_label)] *= 0
                    # print("Class killed - " + str(np.argmax(predicted_label)))
            sum += self._num_of_classes - np.sum(classes_list)
            classes_list = [1] * self._num_of_classes
        return sum

    def killed_cluster_classes(self, clusters):
        classes_list = [1] * self._num_of_classes
        kc_list = []
        for layer_clusters in clusters:  # (a list for each layer <-(a list for each cluster))
            for clust in layer_clusters:
                sum = 0
                m_prime = random.choice(np.array(clust))
                predictions = m_prime.get_model().predict(np.array(self._correct_test_points[0]))
                for i, predicted_label in enumerate(predictions):
                    if np.argmax(self._correct_test_points[1][i]) != np.argmax(predicted_label):
                        classes_list[np.argmax(predicted_label)] *= 0
                        #print("Class killed - " + str(np.argmax(predicted_label)))
                sum += self._num_of_classes - np.sum(classes_list)
                kc_list += [[sum, len(clust)]]
                classes_list = [1] * self._num_of_classes
        return kc_list

    def run(self):
        x, y = self.get_correct_test_points()
        self._correct_test_points = [x, y]
        self._killed_classes = self.killed_classes()
        print("Killed classes: " + str(self._killed_classes))
        self._mutation_score = self._killed_classes / (len(self._mutations) * self._num_of_classes)

    def cluster_run(self):
        x, y = self.get_correct_test_points()
        self._correct_test_points = [x, y]

        kc_list = self.killed_cluster_classes(self._clusters)
        # for each cluster => [number of classes killed, number of mutants in cluster]

        mutation_score_total = 0
        mutation_score_cluster_reps = 0
        total_length = 0
        cluster_amount = 0

        for kc, length in kc_list:
            mutation_score_total += (kc * length)
            total_length += length
            mutation_score_cluster_reps += kc
            cluster_amount += 1

        self._mutation_score = mutation_score_total / (total_length * self._num_of_classes)
        self._cluster_amount = cluster_amount
        print("Mutation score only for cluster reps: " + str(mutation_score_cluster_reps/(cluster_amount* self._num_of_classes)))
        print('Number of clusters: ' + str(cluster_amount))
        print("Mutation score: " + str(self._mutation_score))

    def run_obo(self, x, y):
        # x, y = self.get_correct_test_points()
        self._correct_test_points = [x, y]
        self._killed_classes = self.killed_classes()
        print("Killed classes: " + str(self._killed_classes))
        self._mutation_score = self._killed_classes / (len(self._mutations) * self._num_of_classes)


