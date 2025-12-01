# NoiseAttack
This repository contains code for generating stochastic input noises (single and compound) and for running in-domain attacks.
Synthetic data can be downloaded from [Zenodo](10.5281/zenodo.17779337)



# Description
This repository provides code for generating a wide range of stochastic input noises, including both single-source and compound variants—used in the study: “Should All Noises Be Treated Equally? Impact of Input Noise Variability on Neural Network Robustness.”

It also includes scripts for implementing in-domain noise-based attacks described in: “Noise Attacks: Enhancing the Robustness of Neural Networks through In-domain Attacks.”

# Applications
1- Evaluating ID/OOD generalization under diverse noise distributions.

2- Simulating noise-induced failure cases in vision models.

3- Benchmarking robust training strategies under diverse input perturbation regimes.

4- Analyzing architectural sensitivity to noise characteristics.

5- Generate and apply structured, unstructured, single and compound noise patterns to images.

6- Reproduce in-domain adversarial attacks (soon).

# Citation
If you use this code, please cite the relevant paper(s).

## License

Use of this software implies accepting all the terms and conditions described in
the
[license](LICENSE.txt)
document available in this repository.  We remind users that the use of this
software is permitted for non-commercial applications, and proper credit must be
given to the authors whenever this software is used.

## Requirements

### Hardware

The codes provided are optimized for running on a CUDA capable NVIDIA GPU.
While not strictly required, the user is advised that the neural network training
process can take several hours when running on the GPU and may become prohibitively
long if running on a single CPU. 

### Software

The use of a Linux based operating system is strongly recommended. 
All codes were tested on a Ubuntu 18.04 system and Windows 10 system.

A working distribution of python 3.8 or higher is required.
The use of the [anaconda python distribution](https://www.anaconda.com/) is recommended
to ease the installation of the required python packages.

Examples of use of this software are provided as Jupyter notebooks and as such 
it requires the [Jupyter notebook](https://jupyter.org/) package. Note that this package
is included by default in the anaconda distribution.


## Initial set up

The usage of an Ubuntu 18.04 system or similar with a CUDA capable GPU and the anaconda python
distribution is assumed for the rest of this document. 

### System setup

The use of a separate python virtual environment is recommended for running the provided
programs. The file "NoiseAttack.yml" is provided to quickly setup this environment in Linux
systems. To create an environment using the provided file and activate it do:

```bash
$ cd NoiseAttack
$ conda env create -f NoiseAttack.yml
$ conda activate NoiseAttack
```

To use a Jupyter notebook inside the created virtual environment, type the following code:

```bash
pip install ipykernel ipython kernel install --user --name=NoiseAttack
```
## Usage

Usage instructions are provided in the jupyter notebook files of the repository. Training scripts are provided in src/ directory

Please ensure the kernel is the correct one once the notebook starts running.
 
