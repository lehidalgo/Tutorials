# Working locally with [Tensorflow 2](https://www.tensorflow.org/install?hl=es-419) | Anaconda | Jupyter Notebook | NVIDIA GeForce RTX2070
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
  - [x] Write from Anaconda prompt. We create an Anaconda env with name `tf2gpu` with `Python3.7` and install the packages `ipython` | `ipykernel` | `jupyter`
    - NOTE: We are installing `ipython` | `ipykernel` | `jupyter` before install `tensorflow`. This will make `conda` to install the compatible version of `tensorflow`.
```
conda create -n tf2gpu python=3.7 ipython ipykernel jupyter
```
  - [x] Check the already create anaconda environment. You sould see the name `tf2gpu` in the list.
```
conda info --envs
```
  - [x] Activate the environment.
```
conda activate tf2gpu
```
  - [x] Install tensorflow.
```
conda install tensorflow-gpu
```
  - [x] Install jupyter kernel.
```
python -m ipykernel install --user --name=tf2gpu-kernel
```

4. Simple check to see that TensorFlow is working with your GPU
  - [x] Open `python` in the Anaconda prompt and write the code below. This will show the model of your GPU and the version of `tensorflow`
```
import tensorflow as tf
print( tf.constant('Hello from TensorFlow ' + tf.__version__) )
```
# An Example Convolution Neural Network training using Keras with TensorFlow
1.  Activate `tf2gpu` environment from Anaconda prompt.
2.  Launch Jupyter Notebook
```
jupyter notebook
```
3.  Write the following code in the code notebook cells
```
# import packages 
import tensorflow as tf
import time


# Load and process the MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and rescale data for the CNN
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images, test_images = train_images/255, test_images/255

# Create the LeNet-5 convolution neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set log data to feed to TensorBoard for visual analysis
tensor_board = tf.keras.callbacks.TensorBoard('./logs/LeNet-MNIST-1')

# Train the model
start_time=time.time()
model.fit(train_images, train_labels, batch_size=128, epochs=15, verbose=1,
         validation_data=(test_images, test_labels), callbacks=[tensor_board])
print('Training took {} seconds'.format(time.time()-start_time))
```
4.  Look at the job run with TensorBoard
  - `conda activate tf-gpu`
  - `tensorboard --logdir=./logs --port 6006`  
