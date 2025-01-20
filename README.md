# DeepMAACC: On Accelerating Deep Neural Network Mutation Analysis by Neuron and Mutant Clustering - Replication Package

This repository contains all the data and code needed to replicate our experiments.

In the inputs directory, the models are held within the corresponding dataset names. Not all the data is currently present, but DeepMAACC will download and format the datasets when it is run.

In the outputs directory, the run data and relevant figures from that data are present.

The src directory contains the source code of DeepMAACC.

In order to rerun the models, there is a _network_creation.sh_ bash file that you may run by running the scripts

    $ cd src
    $ ./network_creation.sh

To run DeepMAACC, run the script

    $ python DM_driver.py

DeepMAACC is written using Python 3.9 and its dependencies can be found in the file requirements.txt under the src folder. 

Within src folder, there is a compiled version of modified Graph Mining library. This version is built under a 64-bit Ubuntu Linux. To build the library for a different operating system, download the original repository from https://github.com/google/graph-mining and patch it using the file graph-mining.patch. After patching, our changes shall be applied and you can build the project using the command bazel build //examples:quickstart (we observed that macOS users for the newer versions of the OS need to pass an extra flag to the build system: --macos_minimum_os=14)


DeepMAACC is an implementation of a framework presented in ICST 2025.
If you use DeepMAACC in your research, please use the following BibTeX entry.
```text
@inproceedings{lyons2025deepaaacc,
  title={On Accelerating Deep Neural Network Mutation Analysis by Neuron and Mutant Clustering},
  author={Lyons, Lauren and Ghanbari, Ali},
  booktitle={Proceedings of the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)},
  note={12 pages to appear},
  year={2025}
}
