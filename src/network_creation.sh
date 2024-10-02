#!/bin/bash

python network.py -m fcnn -d mnist
python network.py -m fcnn -d fmnist
python network.py -m fcnn -d kmnist
python network.py -m fcnn -d emnist
echo "Finished fcnn models"

python network.py -m lenet5 -d mnist
python network.py -m lenet5 -d fmnist
python network.py -m lenet5 -d kmnist
python network.py -m lenet5 -d emnist
echo "Finished lenet5 models"

echo "Finished creating models"