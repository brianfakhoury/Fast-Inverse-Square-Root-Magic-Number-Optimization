import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# Model input and output
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# training data
lines = open("best_values.txt", "r").read().split(',')[:-1]
values = list(map(int, lines))
xs = [0.1 * x for x in range(1, len(values) + 1)]
x_plot = xs
x_train = np.array([[x] for x in xs]).reshape([1000, 1])
y_train_tf = np.array(values).reshape([1000, 1])

# Model parameters
W1 = tf.Variable(tf.ones([1, 10]) * .3, dtype=tf.float32)
b1 = tf.Variable(tf.ones([10]) * (-.3), dtype=tf.float32)
W2 = tf.Variable(tf.ones([10, 3]) * .3, dtype=tf.float32)
b2 = tf.Variable(tf.ones([3]) * (-.3), dtype=tf.float32)
W3 = tf.Variable(tf.ones([3, 1]) * .3, dtype=tf.float32)
b3 = tf.Variable(tf.ones([1]) * (-.3), dtype=tf.float32)

layer1 = tf.tanh(tf.multiply(x, W1) + b1)
layer2 = tf.tanh(tf.matmul(layer1, W2) + b2)
linear_model = tf.reduce_sum(tf.matmul(layer2, W3), 1, keepdims=True) + b3

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong

fig, ax = plt.subplots()

x_train_tf = tf.convert_to_tensor(x_train)
y_train_tf = tf.convert_to_tensor(y_train_tf)
for i in range(40000):
    y_train = sess.run(y_train_tf, {x: x_train})
    f_predict, _ = sess.run([linear_model, train], feed_dict={
                            x: x_train, y: y_train})
    curr_layer1, curr_layer2, curr_W1, curr_b1, curr_W2, curr_b2, curr_W3, curr_b3, curr_loss = sess.run([layer1, layer2, W1, b1, W2, b2, W3, b3, loss],
                                                                                                         {x: x_train, y: y_train})
    if i % 1000 == 999:
        print("step ", i)
        print("W1: %s b1: %s" % (curr_W1, curr_b1))
        print("W2: %s b2: %s" % (curr_W2, curr_b2))
        print("W3: %s b3: %s" % (curr_W3, curr_b3))
        print("layer1: %s layer2: %s" % (curr_layer1, curr_layer2))
        print("linear_model: %s loss: %s" % (f_predict, curr_loss))
        print(" ")
        y_plot = y_train.reshape(1, -1)[0]
        pred_plot = f_predict.reshape(1, -1)[0]
        ax.plot(x_plot, y_train[:])
        ax.plot(x_plot, f_predict, 'o-')
        ax.set(xlabel='X Value', ylabel='Y / Predicted Value',
               title=[str(i), " Loss: ", curr_loss])
        plt.pause(0.001)

fig.savefig("fig1.png")
plt.show()
