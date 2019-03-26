#   importing required library
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

#   Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#   Preprocess the dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
label_encoder_1 = LabelEncoder()
x[:, 1] = label_encoder_1.fit_transform(x[:, 1])
label_encoder_2 = LabelEncoder()
x[:, 2] = label_encoder_2.fit_transform(x[:, 2])
one_hot_encoder = OneHotEncoder(categorical_features = [1])
x = one_hot_encoder.fit_transform(x).toarray()
x = x[:, 1:]
# ct = ColumnTransformer([("country", OneHotEncoder(), [1])])
# x = ct.fit_transform(x)

#   Split data for train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

#   Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#   Create the neural network
import tensorflow as tf

#   Defining the placeholders
placeholder_x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
placeholder_y = tf.placeholder(tf.float32, [None, 1])

#   Function to add a layer of neurons
def add_layer(inp, in_size, out_size, activation_function = None):
    weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wxb = tf.matmul(inp, weight) + bias
    if activation_function is None:
        out = wxb
    else:
        out = activation_function(wxb)
    return out

#   Add layers
neuron_1 = 16
neuron_2 = 32
neuron_3 = 8
features = x_train.shape[1]

layer_1 = add_layer(placeholder_x, features, neuron_1, activation_function = tf.nn.relu)
layer_2 = add_layer(layer_1, neuron_1, neuron_2, activation_function = tf.nn.relu)
layer_3 = add_layer(layer_2, neuron_2, neuron_3, activation_function = tf.nn.relu)
layer_out = add_layer(layer_3, neuron_3, 1, activation_function = tf.nn.sigmoid)

#   Calculating error
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_out, labels = placeholder_y))
optimizer = tf.train.AdamOptimizer(0.1).minimize(error)

#   Define batch size and epoch
epoch = 10
batch_size = 10
def next_batch(size, x, y):
    index = np.arange(0, len(x))
    np.random.shuffle(index)
    idx = index[:size]
    x_shuffle = [x[i] for i in idx]
    y_shuffle = [y[i] for i in idx]
    return x_shuffle[2].reshape(1, 11), y_shuffle[2].reshape(1, 1)
#a, b = next_batch(batch_size, x_train, y_train)

#   Running the session
def run_network():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        index = np.arange(len(x_train))
        np.random.shuffle(index)
        for epoc in range(epoch):
            for i in range(batch_size):
                x_data, y_data = x_train[batch_size*i:batch_size*(i+1), :], y_train[batch_size*i:batch_size*(i+1)].reshape(batch_size, 1)
                loss, _ = sess.run([error, optimizer], feed_dict = {placeholder_x: x_data, placeholder_y: y_data})
    #        print('Epoc:', epoc, 'at loss:', loss)
    #    y_pred = sess.run(optimizer, feed_dict = {placeholder_x: x_test})
    #    correct = tf.equal(tf.argmax(layer_out, 1), tf.argmax(placeholder_y, 1))
        predicted = tf.nn.sigmoid(layer_out)
        correct = tf.equal(tf.round(predicted), placeholder_y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:', accuracy.eval(feed_dict = {placeholder_x: x_train, placeholder_y: y_train.reshape(len(y_train), 1)}))
        
run_network()