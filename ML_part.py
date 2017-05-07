"""Machine learning part
Stock price change ratio regression using machine learning models
"""


import pdb
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN.
       CNN model to simulate sell, buy and hold Q-function
       Three models with the same structure
    """
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Each TP_matrix can be regarded as 18x18 image with 1 color channel
    input_layer = tf.reshape(features, [-1, 18, 18, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape:n [batch_size, 18, 18, 1]
    # Output Tensor Shape: [batch_size, 18, 18, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 18, 18, 32]
    # Output Tensor Shape: [batch_size, 9, 9, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 9, 9, 32]
    # Output Tensor Shape: [batch_size, 9, 9, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 9, 9, 64]
    # Output Tensor Shape: [batch_size, 5, 5, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        padding='same')

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 5, 5, 64]
    # Output Tensor Shape: [batch_size, 5 * 5 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 5 * 5 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == learn.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1]
    logits = tf.layers.dense(
        inputs=dropout,
        units=1,
        name="logits")

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    # Mean Square Error
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(
            labels=tf.reshape(labels, [-1, 1]),
            predictions=logits)

    # Configure the Training Op (for TRAIN mode)
    # Adam Optimizer
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="Adam")

    # Generate Predictions
    predictions = {
        "results": logits
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    # # Load training and eval data
    # mnist = learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    cnn_estimator = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="model/convnet_model")

    # Set up logging for predictions
    # Log the values in the "logits" tensor with label "change_ratio"
    tensors_to_log = {"results": "logits"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    cnn_estimator.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=20000,
        monitors=[logging_hook])

    # Evaluate the model and print results
    eval_results = cnn_estimator.evaluate(
        x=eval_data,
        y=eval_labels)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
