import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from network import Dataset
from DMAACC import DMAACC
import os.path


if __name__ == "__main__":
    run_type = 'vanilla'
    # run_type = 'approach1'
    # run_type = 'approach2'

    # arch_type = 'all'
    arch_type = 'one_by_one'

    if run_type == 'once':
        parser = ArgumentParser()
        parser.add_argument('-m',
                            '--model',
                            dest='model_filename',
                            help='Model file in .keras format',
                            required=True)
        parser.add_argument('-ml',
                            '--mutation_level',
                            dest='mutation_level',
                            help='Potential values are \'neuron\' or \'cluster\'',
                            required=True)
        parser.add_argument('-mp',
                            '--mutation-percent',
                            dest='mutation_percent',
                            default='0.1',
                            help='The fraction to add/subtract when mutating the weights (default: 0.1)',
                            required=False)
        parser.add_argument('-c',
                            '--cluster-size',
                            dest='cluster_size',
                            default='5',
                            help='Number of neurons per cluster. Each layer\'s neurons are partioned by this number. Ignored if one-unit-per-cluster activated.',
                            required=False)
        parser.add_argument('-u',
                            '--one-unit-per-cluster',
                            dest='one_unit_per_cluster',
                            default='False',
                            help='One unit per cluster (default: False)',
                            required=False)
        args = parser.parse_args()

        dmaacc_run = DMAACC()

        dmaacc_run.load_model(args.model_filename)
        dmaacc_run.set_mutation_level(args.mutation_level)
        dmaacc_run.set_mutation_percent(float(args.mutation_percent))
        dmaacc_run.set_cluster_size(int(args.cluster_size))
        dmaacc_run.set_one_unit_per_cluster(args.one_unit_per_cluster.lower() == 'true')
        df1 = dmaacc_run.run_approach_1()
        df2, df3 = dmaacc_run.run_approach_2()

    # ========================================================================================================
    elif run_type == 'vanilla':
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_percent(0.1)
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        model_list = [['fcnn-mnist.keras', 'lenet5-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras']]
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist']

        for ds, model_l in zip(dataset_list, model_list):
            d = Dataset(ds)
            dmaacc_run.set_dataset(d)
            for model_n in model_l:
                dmaacc_run.load_model('../inputs/' + ds + '/' + model_n)
                model_current = dmaacc_run.get_model()
                for i in range(30):
                    dmaacc_run.set_mutation_level('neuron')
                    if arch_type == 'all':
                        df_clusters = dmaacc_run.run_vanilla()
                        csvfile = 'vanilla_experiments.csv'
                        if os.path.isfile(csvfile):
                            df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                        else:
                            df_clusters.to_csv(csvfile, mode='w', header=True, index=False)
                    elif arch_type == 'one_by_one':
                        df_clusters = dmaacc_run.run_one_by_one_v()
                        csvfile = 'vanilla_experiments_obo.csv'
                        if os.path.isfile(csvfile):
                            df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                        else:
                            df_clusters.to_csv(csvfile, mode='w', header=True, index=False)

    # ========================================================================================================
    elif run_type == 'approach1':
        dmaacc_run = DMAACC()
        # dmaacc_run.set_one_unit_per_cluster(args.one_unit_per_cluster.lower() == 'true')
        dmaacc_run.set_mutation_percent(0.1)
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        n_list = range(1, 11)

        #model_type_list = [method for method in dir(Network) if ('_scratch' in method or '_keras' in method)]
        model_list = [['fcnn-mnist.keras', 'lenet5-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras']]
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist']

        for ds, model_l in zip(dataset_list, model_list):
            d = Dataset(ds)
            dmaacc_run.set_dataset(d)
            for model_n in model_l:
                dmaacc_run.load_model('../inputs/' + ds + '/' + model_n)
                model_current = dmaacc_run.get_model()
                for num in n_list:
                    dmaacc_run.set_mutation_level('cluster')
                    for i in range(6):
                        dmaacc_run.set_cluster_size(num)
                        if arch_type == 'all':
                            df_clusters = dmaacc_run.run_approach_1()
                            csvfile = 'experiments_approach1.csv'
                            if os.path.isfile(csvfile):
                                df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                            else:
                                df_clusters.to_csv(csvfile, mode='w', header=True, index=False)
                        elif arch_type == 'one_by_one':
                            df_clusters = dmaacc_run.run_one_by_one_a1()
                            csvfile = 'experiments_approach1_obo.csv'
                            if os.path.isfile(csvfile):
                                df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                            else:
                                df_clusters.to_csv(csvfile, mode='w', header=True, index=False)

    # ========================================================================================================
    elif run_type == 'approach2':
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_level("cluster")
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        PH_thresholds = [n / 10 for n in range(1, 10)] + [0.99]
        model_list = [['fcnn-mnist.keras', 'lenet5-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras']]
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist']

        for ds, model_l in zip(dataset_list, model_list):
            d = Dataset(ds)
            dmaacc_run.set_dataset(d)
            for model_n in model_l:
                dmaacc_run.load_model('../inputs/' + ds + '/' + model_n)
                model_current = dmaacc_run.get_model()
                for threshold in PH_thresholds:
                    for i in range(6):
                        dmaacc_run.set_ParHAC_threshold(threshold)
                        dmaacc_run.set_mutation_level(['cluster'])
                        if arch_type == 'all':
                            df_clusters = dmaacc_run.run_approach_2()
                            csvfile = 'experiments_approach2.csv'
                            if os.path.isfile(csvfile):
                                df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                            else:
                                df_clusters.to_csv(csvfile, mode='w', header=True, index=False)
                        elif arch_type == 'one_by_one':
                            df_clusters = dmaacc_run.run_one_by_one_a2()
                            csvfile = 'experiments_approach2_obo.csv'
                            if os.path.isfile(csvfile):
                                df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                            else:
                                df_clusters.to_csv(csvfile, mode='w', header=True, index=False)