# Working with [Tensorflow 2](https://www.tensorflow.org/install?hl=es-419) | Anaconda | Jupyter Notebook | NVIDIA GeForce RTX2070
## Problem Description:
- After creating the anaconda environment I install tensorflow.
- Then, I try to install ipykernel and jupyter. In this step, these libraries are not installed due to conflicts between packages distributions.
- So, make the installation following below steps.


## Steps:
1. **System Preparation - NVIDIA Driver Update and checking your PATH variable:**
  - [x] Check your NVIDIA Driver
  - [x] Check your PATH environment variable. See this [guide](https://www.tensorflow.org/install/gpu#pip_package).


2. **Python Environment Setup with Anaconda Python.**
  - [x] Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
  - [x] Check and Update your Anaconda Python Install.
    - [x] Open Anaconda Prompt *(or Powershell Prompt)* and write `python`. Below should appear in the terminal.
    ```
    Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] :: Anaconda, Inc. on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>>
    ```
    - [x] Update your base Anaconda packages. Run the following commands
    ```
    conda update conda

    conda update anaconda

    conda update python

    conda update --all
    ```
    
3. **Create a Python "virtual environment" for TensorFlow using conda.**
  - [x] Write from Anaconda prompt 
```
conda create --name tf-gpu
```
  - [x] Check the already create anaconda environment. You sould see the name `tf-gpu` in the list.
```
conda info --envs
```
  - [x] Activate the environment.
```
conda activate tf-gpu
```
