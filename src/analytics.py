# RQ1: How much speedup you gain when using neuron clustering or mutant clustering vs vanilla mutation testing?
# RQ2: How much mutation score is lost when using neuron clustering or mutant clustering vs vanilla mutation testing
# RQ3: What is the impact of non-determinacy in training process on mutation score of your tool
# 		when using neuron clustering or mutant clustering vs in vanilla mode
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import mannwhitneyu
from keras.models import load_model
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

rcParams['font.weight'] = 'bold'
plt.rc('font',family='Times New Roman')

vanilla = pd.read_csv('../outputs/data/final_vanilla_experiments.csv')
vl = len(vanilla.index)
approach1 = pd.read_csv('../outputs/data/final_approach1_experiments.csv')
a1l = len(approach1.index)
a1_params = pd.unique(approach1['Neurons_per_Cluster_param'])
a1_params.sort()
approach2 = pd.read_csv('../outputs/data/final_approach2_experiments.csv')
a2l = len(approach2.index)
#a2_params = pd.unique(approach2['ParHAC_Threshold'])
a2_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
a2_models = pd.unique(approach2['Model_Type'])

avg_vanilla = {}
v_models = pd.unique(vanilla['Model_Type'])
for model in v_models:
	avg_vanilla[model] = [vanilla[vanilla['Model_Type'] == model]['Mutate_time'].mean(),
						  vanilla[vanilla['Model_Type'] == model]['Mutation_Score'].mean(),
						  vanilla[vanilla['Model_Type'] == model]['MS_time'].mean()]

avg_a1 = {}
a1_models = pd.unique(approach1['Model_Type'])
for model in a1_models:
	temp = []
	for n in a1_params:
		temp += [tuple((n,
				   [approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Mutate_time'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Cluster_time'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Mutation_Score'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['MS_time'].mean()]))]
	avg_a1[model] = temp

avg_a2 = {}
for model in a2_models:
	temp = []
	for n in a2_params:
		temp += [tuple((n,
				   [approach2[approach2['ParHAC_Threshold'] == n][approach2['Model_Type'] == model]['Mutate_time'].mean(),
				   approach2[approach2['ParHAC_Threshold'] == n][approach2['Model_Type'] == model]['Cluster_time'].mean(),
				   approach2[approach2['ParHAC_Threshold'] == n][approach2['Model_Type'] == model]['Mutation_Score'].mean(),
				   approach2[approach2['ParHAC_Threshold'] == n][approach2['Model_Type'] == model]['MS_time'].mean()]))]
	avg_a2[model] = temp


config = {
	'Print_Averages': True,
	'Approach1_Speedup_FCNN': True,
	'Approach1_Speedup_LeNet-5': True,
	'Approach2_Speedup_FCNN': True,
	'Approach2_Speedup_LeNet-5': True,
	'Approach1_MSE_FCNN': True,
	'Approach1_MSE_LeNet-5': True,
	'Approach2_MSE_FCNN': True,
	'Approach2_MSE_LeNet-5': True,
	'Box_and_Whisker_Speedup': True,
	'Box_and_Whisker_MSE': True,
	'Approach2_NumofClusters': True,
	'Approach1&2_NumofClusters': True,
	'Approach1&2_NumofTestedMutants': True,
	'Model_Info': True
}

# ===================================================================================================================================
# ===================================================================================================================================
# Print_Averages

if config['Print_Averages']:
	label, X, Y = [], [], []
	for key, value in avg_a1.items():
		if 'fcnn' in key or 'lenet5' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average speedup of Approach1 across all fcnn&lenet5 models and all parameters is: " + str(Y.mean()))

	label, X, Y = [], [], []
	for key, value in avg_a2.items():
		if 'fcnn' in key or 'lenet5' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average speedup of Approach2 across all fcnn&lenet5 models and all parameters is: " + str(Y.mean()))

	label, X, Y = [], [], []
	for key, value in avg_a1.items():
		if 'fcnn' in key or 'lenet5' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average MSE of Approach1 across all fcnn&lenet5 models and all parameters is: " + str(Y.mean()))

	label, X, Y = [], [], []
	for key, value in avg_a2.items():
		if 'fcnn' in key or 'lenet5' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average MSE of Approach2 across all fcnn&lenet5 models and all parameters is: " + str(Y.mean()))

# ===================================================================================================================================
# ===================================================================================================================================
# Approach1_Speedup

if config['Approach1_Speedup_FCNN']:
	label, X, Y = [], [], []
	for key, value in avg_a1.items():
		if 'fcnn' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average speedup of Approach1 across all fcnn models and all parameters is: " + str(Y.mean()))

	valid_indices = ~np.isnan(Y)
	X_clean = X[valid_indices]
	Y_clean = Y[valid_indices]
	labels_clean = labels[valid_indices]
	unique_labels = np.unique(labels_clean)

	plt.figure(figsize=(10, 6))
	for label in unique_labels:
		indices = labels_clean == label
		plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper())

	plt.xticks(np.array(a1_params))
	plt.xlabel('Neurons per Cluster', fontsize=24, weight='bold')
	plt.ylabel('Speedup', fontsize=24, weight='bold')
	# plt.title('Approach 1: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/NeuronC_Speedup_FCNN.pdf')

# ------------------------------------------------------------------------------------------------------------

if config['Approach1_Speedup_LeNet-5']:
	label, X, Y = [], [], []
	for key, value in avg_a1.items():
		if 'lenet' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average speedup of Approach1 across all lenet5 models and all parameters is: " + str(Y.mean()))

	valid_indices = ~np.isnan(Y)
	X_clean = X[valid_indices]
	Y_clean = Y[valid_indices]
	labels_clean = labels[valid_indices]
	unique_labels = np.unique(labels_clean)

	plt.figure(figsize=(10, 6))
	for label in unique_labels:
		indices = labels_clean == label
		plt.plot(X_clean[indices], Y_clean[indices],
				 label=label.split('/')[-1].split('.')[0].upper().replace('LENET', 'LeNet-'))

	plt.xticks(np.array(a1_params))
	plt.xlabel('Neurons per Cluster', fontsize=24, weight='bold')
	plt.ylabel('Speedup', fontsize=24, weight='bold')
	# plt.title('Approach 1: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/NeuronC_Speedup_LeNet-5.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Approach2_Speedup

if config['Approach2_Speedup_FCNN']:
	label, X, Y = [], [], []
	for key, value in avg_a2.items():
		if 'fcnn' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)

	print("Average speedup of Approach2 across all fcnn models and all parameters is: " + str(Y.mean()))

	valid_indices = ~np.isnan(Y)
	X_clean = X[valid_indices]
	Y_clean = Y[valid_indices]
	labels_clean = labels[valid_indices]
	unique_labels = np.unique(labels_clean)

	plt.figure(figsize=(10, 6))
	for label in unique_labels:
		indices = labels_clean == label
		plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper())

	plt.xticks(np.array(a2_params))
	plt.xlabel('PARHAC Threshold', fontsize=24, weight='bold')
	plt.ylabel('Speedup', fontsize=24, weight='bold')
	# plt.title('Approach 2: How much speedup you gain when using mutant clustering vs vanilla mutation testing?')
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/MutantC_Speedup_FCNN.pdf')

	# ------------------------------------------------------------------------------------------------------------

if config['Approach2_Speedup_LeNet-5']:
	label, X, Y = [], [], []
	for key, value in avg_a2.items():
		if 'lenet' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average speedup of Approach2 across all lenet5 models and all parameters is: " + str(Y.mean()))

	valid_indices = ~np.isnan(Y)
	X_clean = X[valid_indices]
	Y_clean = Y[valid_indices]
	labels_clean = labels[valid_indices]
	unique_labels = np.unique(labels_clean)

	plt.figure(figsize=(10, 6))
	for label in unique_labels:
		indices = labels_clean == label
		plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper().replace('LENET', 'LeNet-'))

	plt.xticks(np.array(a2_params))
	plt.xlabel('PARHAC Threshold', fontsize=24, weight='bold')
	plt.ylabel('Speedup', fontsize=24, weight='bold')
	# plt.title('Approach 2: How much speedup you gain when using mutant clustering vs vanilla mutation testing?')
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/MutantC_Speedup_LeNet-5.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Approach1_Mutation_Score_Error

if config['Approach1_MSE_FCNN']:
	label, X, Y = [], [], []
	for key, value in avg_a1.items():
		if 'fcnn' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average MSE of Approach1 across all fcnn models and all parameters is: " + str(Y.mean()))

	valid_indices = ~np.isnan(Y)
	X_clean = X[valid_indices]
	Y_clean = Y[valid_indices]
	labels_clean = labels[valid_indices]
	unique_labels = np.unique(labels_clean)

	plt.figure(figsize=(10, 6))
	for label in unique_labels:
		indices = labels_clean == label
		plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper())

	plt.xticks(np.array(a1_params))
	plt.xlabel('Neurons per Cluster', fontsize=24, weight='bold')
	plt.ylabel('Mutation Score Error', fontsize=24, weight='bold')
	# plt.title('Approach 1: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/NeuronC_MSE_FCNN.pdf')

	# ------------------------------------------------------------------------------------------------------------

if config['Approach1_MSE_LeNet-5']:
	label, X, Y = [], [], []
	for key, value in avg_a1.items():
		if 'lenet' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average MSE of Approach1 across all lenet5 models and all parameters is: " + str(Y.mean()))

	valid_indices = ~np.isnan(Y)
	X_clean = X[valid_indices]
	Y_clean = Y[valid_indices]
	labels_clean = labels[valid_indices]
	unique_labels = np.unique(labels_clean)

	plt.figure(figsize=(10, 6))
	for label in unique_labels:
		indices = labels_clean == label
		plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper().replace('LENET', 'LeNet-'))

	plt.xticks(np.array(a1_params))
	plt.xlabel('Neurons per Cluster', fontsize=24, weight='bold')
	plt.ylabel('Mutation Score Error', fontsize=24, weight='bold')
	# plt.title('Approach 1: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/NeuronC_MSE_LeNet-5.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Approach2_Mutation_Score_Error

if config['Approach2_MSE_FCNN']:
	label, X, Y = [], [], []
	for key, value in avg_a2.items():
		if 'fcnn' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average MSE of Approach2 across all fcnn models and all parameters is: " + str(Y.mean()))

	valid_indices = ~np.isnan(Y)
	X_clean = X[valid_indices]
	Y_clean = Y[valid_indices]
	labels_clean = labels[valid_indices]
	unique_labels = np.unique(labels_clean)

	plt.figure(figsize=(10, 6))
	for label in unique_labels:
		indices = labels_clean == label
		plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper())

	plt.xticks(np.array(a2_params))
	plt.xlabel('PARHAC Threshold', fontsize=24, weight='bold')
	plt.ylabel('Mutation Score Error', fontsize=24, weight='bold')
	# plt.title('Approach 2: How much mutation score is lost when using mutant clustering vs vanilla mutation testing?')
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/MutantC_MSE_FCNN.pdf')

	# ------------------------------------------------------------------------------------------------------------

if config['Approach2_MSE_LeNet-5']:
	label, X, Y = [], [], []
	for key, value in avg_a2.items():
		if 'lenet5' in key:
			for thing in value:
				label += [key]
				X += [thing[0]]
				Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]
	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)
	print("Average MSE of Approach2 across all lenet5 models and all parameters is: " + str(Y.mean()))

	valid_indices = ~np.isnan(Y)
	X_clean = X[valid_indices]
	Y_clean = Y[valid_indices]
	labels_clean = labels[valid_indices]
	unique_labels = np.unique(labels_clean)

	plt.figure(figsize=(10, 6))
	for label in unique_labels:
		indices = labels_clean == label
		plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper().replace('LENET', 'LeNet-'))

	plt.xticks(np.array(a2_params))
	plt.xlabel('PARHAC Threshold', fontsize=24, weight='bold')
	plt.ylabel('Mutation Score Error', fontsize=24, weight='bold')
	# plt.title('Approach 2: How much mutation score is lost when using mutant clustering vs vanilla mutation testing?')
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/MutantC_MSE_LeNet-5.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Approach1&2_Box_and_Whisker_Speedup

if config['Box_and_Whisker_Speedup']:

	a2_params_man = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
	#a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
	a1_params_man = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	param_names = ['p1', 'p1', 'p2', 'p2', 'p3', 'p3', 'p4', 'p4', 'p5', 'p5',
				   'p6', 'p6', 'p7', 'p7', 'p8', 'p8', 'p9', 'p9', 'p10', 'p10']
	models_bp = [value for value in a2_models if value in a1_models]

	for model in models_bp:
		if 'fcnn' in model or 'lenet5' in model:
			Y = []
			for param_a1, param_a2 in zip(a1_params_man, a2_params_man):
				a1_vals = approach1[approach1['Neurons_per_Cluster_param'] == param_a1][approach1['Model_Type'] == model]['MS_time']
				a2_vals = approach2[approach2['ParHAC_Threshold'] == param_a2][approach2['Model_Type'] == model]['MS_time']
				temp = []
				for index, thing in a1_vals.items():
					temp += [float(((avg_vanilla[model][2]) - thing) / (avg_vanilla[model][2]))]
				Y += [temp]
				temp = []
				for index, thing in a2_vals.items():
					temp += [float(((avg_vanilla[model][2]) - thing) / (avg_vanilla[model][2]))]
				Y += [temp]

			fig = plt.figure(figsize=(10, 6))
			ax = fig.add_subplot(111)
			pos=[0.6,1.4,2.6,3.4,4.6,5.4,6.6,7.4,8.6,9.4,10.6,11.4,12.6,13.4,14.6,15.4,16.6,17.4,18.6,19.4]
			bp = ax.boxplot(Y,positions=pos, boxprops={'linewidth': 2}, patch_artist=True)
			ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
			for i in range(1, 20, 2):
				if len(Y[i])!=0 and len(Y[i-1])!=0:
					U1, p = mannwhitneyu(Y[i-1], Y[i])
					ax.text(i-0.5, max(max(Y[i-1]), max(Y[i]))*1.02, "{:.3f}".format(p), fontsize=16)
			plt.xlabel('Parameters', fontsize=24)
			plt.ylabel('Speedup', fontsize=24)
			# plt.title(model.split("/")[-1]+' Box-and-Whisker Plot for Mutation Testing Time Speedup')
			plt.grid(True)
			plt.tight_layout()

			colors = ['blue', 'red'] * 10
			for patch, color in zip(bp['boxes'], colors):
				patch.set_facecolor(color)
			colors = ['lightblue', 'orange'] * 10
			for median, color in zip(bp['medians'], colors):
				median.set_color(color)

			plt.savefig('../outputs/BoxPlot_Speedup_'+model.split("/")[-1].split(".")[0].upper().replace('LENET', 'LeNet-')+'.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Approach1&2_Box_and_Whisker_MSE

if config['Box_and_Whisker_MSE']:
	a2_params_man = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
	#a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
	a1_params_man = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	param_names = ['p1', 'p1', 'p2', 'p2', 'p3', 'p3', 'p4', 'p4', 'p5', 'p5',
				   'p6', 'p6', 'p7', 'p7', 'p8', 'p8', 'p9', 'p9', 'p10', 'p10']
	models_bp = [value for value in a2_models if value in a1_models]

	for model in models_bp:
		if 'fcnn' in model or 'lenet5' in model:
			Y = []
			for param_a1, param_a2 in zip(a1_params_man, a2_params_man):
				a1_vals = approach1[approach1['Neurons_per_Cluster_param'] == param_a1][approach1['Model_Type'] == model]['Mutation_Score']
				a2_vals = approach2[approach2['ParHAC_Threshold'] == param_a2][approach2['Model_Type'] == model]['Mutation_Score']
				temp = []
				for index, thing in a1_vals.items():
					temp += [float(((avg_vanilla[model][1]) - thing) / (avg_vanilla[model][1]))]
				Y += [temp]
				temp = []
				for index, thing in a2_vals.items():
					temp += [float(((avg_vanilla[model][1]) - thing) / (avg_vanilla[model][1]))]
				Y += [temp]

			fig = plt.figure(figsize=(10, 6))
			ax = fig.add_subplot(111)
			pos=[0.6,1.4,2.6,3.4,4.6,5.4,6.6,7.4,8.6,9.4,10.6,11.4,12.6,13.4,14.6,15.4,16.6,17.4,18.6,19.4]
			bp = ax.boxplot(Y,positions=pos, boxprops={'linewidth': 2}, patch_artist=True)
			ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
			for i in range(1, 20, 2):
				if len(Y[i])!=0 and len(Y[i-1])!=0:
					U1, p = mannwhitneyu(Y[i-1], Y[i])
					ax.text(i-0.5, max(max(Y[i-1]), max(Y[i]))*1.02, "{:.3f}".format(p), fontsize=16)

			plt.xlabel('Parameters', fontsize=24)
			plt.ylabel('Mutation Score Error', fontsize=24)
			# plt.title(model.split("/")[-1]+' Box-and-Whisker Plot for Mutation Score Error')
			plt.grid(True)
			plt.tight_layout()

			colors = ['blue', 'red'] * 10
			for patch, color in zip(bp['boxes'], colors):
				patch.set_facecolor(color)
			colors = ['lightblue', 'orange'] * 10
			for median, color in zip(bp['medians'], colors):
				median.set_color(color)

			plt.savefig('../outputs/BoxPlot_MSE_'+model.split("/")[-1].split(".")[0].upper().replace('LENET', 'LeNet-')+'.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Approach2_Num_of_Clusters

if config['Approach2_NumofClusters']:
	label, X, Y = [], [], []
	a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
	a2_models = pd.unique(approach2['Model_Type'])

	for model in a2_models:
		if 'fcnn' in model or 'lenet5' in model:
			temp = []
			tempy = []
			for param in a2_params_man:
				temp += [param]
				tempy += [(approach2[approach2['ParHAC_Threshold'] == param][approach2['Model_Type'] == model]['Number_of_Clusters']).mean()]
			Y += [tempy]
			X += [temp]
			label += [model.split('/')[-1].split('.')[0]]

	X = np.array(X)
	Y = np.array(Y)
	labels = np.array(label)

	plt.figure(figsize=(10, 6))
	for x, y, l in zip(X, Y, labels):
		plt.plot(x, y, label=l)

	plt.xticks(np.array(a2_params))
	plt.xlabel('PARHAC Threshold')
	plt.ylabel('Number of Clusters')
	plt.legend(title='Models')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/MutantC_NumofClusters.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Approach1&2_Num_of_Clusters

if config['Approach1&2_NumofClusters']:
	label_a1 = []
	label_a2 = []
	A1_Y = []
	A2_Y = []
	X = []
	#a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
	a2_params_man = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
	a1_params_man = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	a2_models = pd.unique(approach2['Model_Type'])
	a1_models = pd.unique(approach1['Model_Type'])
	param_names = ['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
	models_both = [value for value in a2_models if value in a1_models]
	models = [[m for m in models_both if 'fcnn' in m], [m for m in models_both if 'lenet5' in m]]

	for model in models:
		temp_a1_y = []
		temp_a2_y = []
		temp_x = []
		for param_a1, param_a2, param in zip(a1_params_man, a2_params_man, param_names):
			a1_val = approach1[approach1['Neurons_per_Cluster_param'] == param_a1][approach1['Model_Type'].isin(model)]['Number_of_Clusters'].mean()
			a2_val = approach2[approach2['ParHAC_Threshold'] == param_a2][approach2['Model_Type'].isin(model)]['Number_of_Clusters'].mean()
			temp_a1_y += [a1_val]
			temp_a2_y += [a2_val]
			temp_x += [param]
		A1_Y += [temp_a1_y]
		A2_Y += [temp_a2_y]
		X += [temp_x]
		label_a1 += ['Neuron Clustering FCNN' if 'fcnn' in model[0] else 'Neuron Clustering LeNet-5']
		label_a2 += ['Mutant Clustering FCNN' if 'fcnn' in model[0] else 'Mutant Clustering LeNet-5']

	A1_Y = np.array(A1_Y)
	A2_Y = np.array(A2_Y)
	X = np.array(X)
	plt.figure(figsize=(10, 6))

	for x, y, l in zip(X, A1_Y, label_a1):
		plt.plot(x, y, label=l)
	for x, y, l in zip(X, A2_Y, label_a2):
		plt.plot(x, y, label=l)

	plt.xlabel('Parameter Value', fontsize=24)
	plt.ylabel('Number of Clusters', fontsize=24)
	plt.legend(title='Models', fontsize=22)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/Neuron_and_MutantC_NumofClusters.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Approach1&2_Num_of_Tested_Mutants

if config['Approach1&2_NumofTestedMutants']:
	label_a1 = []
	label_a2 = []
	A1_Y = []
	A2_Y = []
	X = []
	#a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
	a2_params_man = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
	a1_params_man = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	a2_models = pd.unique(approach2['Model_Type'])
	a1_models = pd.unique(approach1['Model_Type'])
	param_names = ['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
	models_both = [value for value in a2_models if value in a1_models]
	models = [[m for m in models_both if 'fcnn' in m], [m for m in models_both if 'lenet5' in m]]

	for model in models:
		temp_a1_y = []
		temp_a2_y = []
		temp_x = []
		for param_a1, param_a2, param in zip(a1_params_man, a2_params_man, param_names):
			a1_val = approach1[approach1['Neurons_per_Cluster_param'] == param_a1][approach1['Model_Type'].isin(model)]['Number_of_Mutants'].mean()
			a2_val = approach2[approach2['ParHAC_Threshold'] == param_a2][approach2['Model_Type'].isin(model)]['Number_of_Clusters'].mean()
			temp_a1_y += [a1_val]
			temp_a2_y += [a2_val]
			temp_x += [param]
		A1_Y += [temp_a1_y]
		A2_Y += [temp_a2_y]
		X += [temp_x]
		label_a1 += ['FCNN Neuron Clustering' if 'fcnn' in model[0] else 'LeNet-5 Neuron Clustering']
		label_a2 += ['FCNN Mutant Clustering' if 'fcnn' in model[0] else 'LeNet-5 Mutant Clustering']

	A1_Y = np.array(A1_Y)
	A2_Y = np.array(A2_Y)
	X = np.array(X)
	labels = np.array(label)
	plt.figure(figsize=(10, 6))

	for x, y, l in zip(X, A1_Y, label_a1):
		plt.plot(x, y, label=l)
	for x, y, l in zip(X, A2_Y, label_a2):
		plt.plot(x, y, label=l)
	# plt.plot(['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10'], [vanilla[vanilla['Model_Type'].isin(models[0])]['Number_of_Mutants'].mean()]*10, label='FCNN Vanilla Mutation Testing')
	# plt.plot(['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10'], [vanilla[vanilla['Model_Type'].isin(models[1])]['Number_of_Mutants'].mean()]*10, label='LeNet-5 Vanilla Mutation Testing')

	plt.xlabel('Parameter Value', fontsize=24)
	plt.ylabel('Number of Mutants Tested', fontsize=24)
	plt.legend(title='Models', fontsize=19, loc='center right', bbox_to_anchor=(1, 0.35))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('../outputs/Neuron_and_MutantC_NumofTestedMutants.pdf')

# ===================================================================================================================================
# ===================================================================================================================================
# Model_Info

if config['Model_Info']:

	models = ['fcnn-mnist.keras', 'lenet5-mnist.keras', 'fcnn-fmnist.keras', 'lenet5-fmnist.keras',
			  'fcnn-kmnist.keras', 'lenet5-kmnist.keras', 'fcnn-emnist.keras', 'lenet5-emnist.keras']

	for m in models:
		s_model = load_model('../inputs/' + m.split('-')[1].split('.')[0] + '/' + m)
		print(m)
		print(s_model.summary())
		x_train = np.load('../inputs/' + m.split('-')[1].split('.')[0]  + '/data/' + m.split('-')[1].split('.')[0]  + '_train_inputs.npy')
		print(len(x_train))
		y_train = np.load('../inputs/' + m.split('-')[1].split('.')[0]  + '/data/' + m.split('-')[1].split('.')[0]  + '_train_outputs.npy')
		print(len(y_train))
		x_test = np.load('../inputs/' + m.split('-')[1].split('.')[0]  + '/data/' + m.split('-')[1].split('.')[0]  + '_test_inputs.npy')
		print(len(x_test))
		y_test = np.load('../inputs/' + m.split('-')[1].split('.')[0]  + '/data/' + m.split('-')[1].split('.')[0]  + '_test_outputs.npy')
		print(len(y_test))
		hist = s_model.evaluate(x_test, y_test)
		print(hist)
		print()