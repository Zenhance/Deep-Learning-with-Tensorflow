import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_v2_behavior()


def read_dataset():
    data = pd.read_csv("G:/Python/Deep Learing Course/Deep Learning with Tensorflow/data/sonar.all-data.csv" )

    x = data[data.columns[0:60]].values
    y = data[data.columns[60]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = one_hot_encoder(y)
    return (x,y)

def one_hot_encoder(arr):
    element = len(arr)
    u_element = len(np.unique(arr))
    one_hot_encode = np.zeros((element, u_element))
    one_hot_encode[np.arange(element), arr] = 1
    return one_hot_encode


#read the dataset
x, y = read_dataset()

#shuffle the dataset
x, y = shuffle(x, y, random_state=1)

#split the dataset into training and testing part
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=46)

"""print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)"""

#Defining some important constants to train the model and working with tensors.
learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = x.shape[1]
n_class = y.shape[1]
model_path = "G:/Python/Deep Learing Course/models/model"


#Defining the hidden layers and the number of neurons in each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

input_x = tf.placeholder(tf.float32, [None, n_dim])
w = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
output_y = tf.placeholder(tf.float32, [None, n_class])


weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

def multilayer_perception(input_values, weights, biases):
      layer_1 = tf.add(tf.matmul(input_values, weights['h1']), biases['b1'])
      layer_1 = tf.nn.sigmoid(layer_1)

      layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
      layer_2 = tf.nn.sigmoid(layer_2)

      layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
      layer_3 = tf.nn.sigmoid(layer_3)

      layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
      layer_4 = tf.nn.relu(layer_4)

      out_layer = tf.matmul(layer_4, weights['out'])+biases['out']
      return out_layer


init = tf.global_variables_initializer()

saver = tf.train.Saver()

model = multilayer_perception(input_x, weights, biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=output_y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)
#graph_writer = tf.summary.FileWriter("./logs", sess.graph)

mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={input_x: train_x, output_y: train_y})
    cost = sess.run(cost_function,feed_dict={input_x:train_x,output_y: train_y})
    cost_history = np.append(cost_history,cost)
    correct_prediction = tf.equal(tf.arg_max(model,1),tf.arg_max(output_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    predict_y = sess.run(model,feed_dict={input_x:test_x})
    mse = tf.reduce_mean(tf.square(predict_y-test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy,feed_dict = {input_x:train_x,output_y:train_y}))
    accuracy_history.append(accuracy)

    print("epoch: ",epoch," - ","cost: ",cost,"- MSE: ",mse_,"- Train Accuracy: ",accuracy)


save_path = saver.save(sess,model_path)
print("Model saved in file: %s" %save_path)


#print the final accuracy
correct_prediction = tf.equal(tf.arg_max(model,1),tf.arg_max(output_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("Test Accuracy: ",(sess.run(accuracy,feed_dict={input_x:test_x,output_y:test_y})))


#Print the final mean squared error
predict_y = sess.run(model,feed_dict={input_x:test_x})
mse = tf.reduce_mean(tf.square(predict_y-test_y))
print("MSE: %.4f" % sess.run(mse))
