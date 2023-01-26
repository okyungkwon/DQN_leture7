import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import gym
env = gym.make('CartPole-v0')

# Constants defining our neural network
learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.placeholder(tf.float32, [None, input_size], name='input_x')

#First layer of weights
W1 = tf.get_variable("W1", shape=[input_size, output_size],
                     initializer=tf.compat.v1.keras.initializers.glorot_normal)
Qpred = tf.matmul(X, W1)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

# Loss function
loss = tf.reduce_sum(tf.square(Y-Qpred))
# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Values for q learning
num_episodes = 5000
dis = 0.9
rList = []

# Setting up our environment
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    e = 1. / ((i / 10) + 1)
    step_count = 0
    s = env.reset()[0]
    done = False

    # The Q-Network training
    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])
        # Choose an action by greedily (with e chance of random action) from
        # the Q-network
        Qs = sess.run(Qpred, feed_dict={X: x})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # Get new state and reward from environment
        s1, reward, done, _, _ = env.step(a)
        if done:
            # Update Q, and no Qs+1, since it's a terminal state
            Qs[0, a] = -100
        else:
            x1 = np.reshape(s1, [1, input_size])
            # Obtain the Q_s1 values by feeding the new state through our network
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
            # Update Q
            Qs[0, a] = reward + dis * np.max(Qs1)

        # Train our network using target (Y) and predicted Q (Qpred) values
        sess.run(train, feed_dict={X: x, Y: Qs})
        s = s1

    rList.append(step_count)
    print("Episodes: {} steps: {}".format(i, step_count))
    # If last 10's avg steps are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

# See our trained network in action
observation = env.reset()[0]
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict={X:x})
    a = np.argmax(Qs)

    observation, reward, done, _, _ = env.step(a)
    reward_sum += reward
    if done:
        print('Total score: {}'.format(reward_sum))
        break