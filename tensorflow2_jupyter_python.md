# Working locally with [Tensorflow 2](https://www.tensorflow.org/install?hl=es-419) | Python | Jupyter Notebook | NVIDIA GeForce RTX2070
## Problem Description:
- After creating the python environment I install tensorflow.
- I try to work with a Jupyter Notebook but the kernels dies. This happens because tensorflow do not find the cudnn*.dll. Maybe I missed an step of this [guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
- So, make the installation following below steps.


## Steps:
1. **System Preparation - NVIDIA Driver Update and checking your PATH variable:**
  - [x] Check your NVIDIA Driver
  - [x] Check your PATH environment variable. See this [guide](https://www.tensorflow.org/install/gpu#pip_package).
  - [x] BE SURE you follow this [guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows).


2. **Python Environment Setup.**
  - [x] Install [Python](https://www.python.org/downloads/)
  - [x] Check and Update your Python Install.
    - [x] Open the prompt *(or Powershell Prompt)* and write `python`. Below should appear in the terminal.
    ```
    Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>>
    ```
    
3. **Create a Python "virtual environment" for TensorFlow using pip.**
  - [x] Follow this [tutorial](https://python.land/virtual-environments/virtualenv) about "create, activate and delete python virtual environment"

4. Simple check to see that TensorFlow is working with your GPU
  - [x] Open `python` in the command prompt and write the code below. This will show the model of your GPU and the version of `tensorflow`
```
import tensorflow as tf
print( tf.constant('Hello from TensorFlow ' + tf.__version__) )
```
# An Example Convolution Neural Network training using Keras with TensorFlow
1.  Activate python environment from command prompt.
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
4.  Look at the job run with TensorBoard using command prompt and browser
  - Activate python environment
  - `tensorboard --logdir=./logs --port 6006`  
