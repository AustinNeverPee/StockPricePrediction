"""Machine learning part
Stock price change ratio regression using machine learning models
"""


import pdb
import numpy as np
import tensorflow as tf
import random
import pickle

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


class DataSet(object):
    def __init__(self):
        self.tp_features = []
        self.labels = 1


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
        units=1)

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
            learning_rate=0.0001,
            optimizer="Adam")

    # Generate Predictions
    predictions = {
        "results": logits
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    # Load training and eval data
    pkl_file = open("data/data_set_" + stock_name + ".pkl", "rb")
    data = pickle.load(pkl_file)
    pkl_file.close()

    l = int(len(data) * 0.7)
    train_data = np.zeros((l, 18, 18), dtype=np.float32)
    train_labels = np.zeros(l, dtype=np.float32)
    eval_data = np.zeros((len(data) - l, 18, 18), dtype=np.float32)
    eval_labels = np.zeros((len(data) - l), dtype=np.float32)

    # random.shuffle(data)
    for i in range(l):
        train_data[i] = data[i].tp_features
        train_labels[i] = data[i].labels
    for i in range(len(data) - l):
        eval_data[i] = data[i + l].tp_features
        eval_labels[i] = data[i + l].labels

    # Create the Estimator
    cnn_estimator = learn.Estimator(
        model_fn=cnn_model_fn,
        model_dir="model/" + stock_name + "/convnet_model")

    # # Set up logging for predictions
    # # Log the values in the "logits" tensor with label "change_ratio"
    # tensors_to_log = {
    #     "predictions": "dense_2/BiasAdd:0",
    #     "labels": "output:0"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=1000)

    # Train the model
    cnn_estimator.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=4000)

    # Evaluate the model and print results
    eval_results = cnn_estimator.evaluate(
        x=eval_data,
        y=eval_labels)
    print(eval_results)

    # Output data prediction
    train_predictions = cnn_estimator.predict(
        x=train_data,
        as_iterable=False
    )
    eval_predictions = cnn_estimator.predict(
        x=eval_data,
        as_iterable=False
    )

    # Store prediction and labels into pickle format
    # in convenience of further inspection
    output = open('ML_result/' + stock_name + '/train_labels.pkl', 'wb')
    pickle.dump(train_labels, output)
    output.close()
    output = open('ML_result/' + stock_name + '/eval_labels.pkl', 'wb')
    pickle.dump(eval_labels, output)
    output.close()
    output = open('ML_result/' + stock_name + '/train_predictions.pkl', 'wb')
    pickle.dump(train_predictions, output)
    output.close()
    output = open('ML_result/' + stock_name + '/eval_predictions.pkl', 'wb')
    pickle.dump(eval_predictions, output)
    output.close()


if __name__ == "__main__":
    # Sotck name
    stock_name = "BA"

    tf.app.run()
