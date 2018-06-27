import random

import numpy as np
import pandas as pd
import tensorflow as tf

import re
import time

from sklearn import preprocessing

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_gpus', 4, "How many GPUs to use.")

k = 10

epochs = 10000
display_step = 10

learning_rate = 0.03

batch_size = 250

train_data = "./train-1m.tsv"
test_data = "./test-1m.tsv"


def inference(x):
    num_hidden_1 = 10   # 1st layer num features
    num_hidden_2 = 5    # 2nd layer num features (the latent dim)

    initializer = tf.random_normal_initializer

    encoder_h1 = tf.get_variable('encoder_h1', [num_input, num_hidden_1], initializer=initializer,
                                      dtype=tf.float64)

    encoder_h2 = tf.get_variable('encoder_h2', [num_hidden_1, num_hidden_2], initializer=initializer,
                                      dtype=tf.float64)

    decoder_h1 = tf.get_variable('decoder_h1', [num_hidden_2, num_hidden_1], initializer=initializer,
                                      dtype=tf.float64)

    decoder_h2 = tf.get_variable('decoder_h2', [num_hidden_1, num_input], initializer=initializer,
                                      dtype=tf.float64)

    encoder_b1 = tf.get_variable('encoder_b1', [num_hidden_1], initializer=initializer, dtype=tf.float64)
    encoder_b2 = tf.get_variable('encoder_b2', [num_hidden_2], initializer=initializer, dtype=tf.float64)
    decoder_b1 = tf.get_variable('decoder_b1', [num_hidden_1], initializer=initializer, dtype=tf.float64)
    decoder_b2 = tf.get_variable('decoder_b2', [num_input], initializer=initializer, dtype=tf.float64)

    def encoder(x_encoder):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_encoder, encoder_h1), encoder_b1))

        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, encoder_h2), encoder_b2))

        return layer_2

    # Building the decoder

    def decoder(x_decoder):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_decoder, decoder_h1), decoder_b1))

        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, decoder_h2), decoder_b2))

        return layer_2

    # Construct model

    encoder_op = encoder(x)
    decoder_op = decoder(encoder_op)

    return decoder_op


def model_loss(y_hat, y):
    # Calculate the average loss across the batch.

    mse = tf.losses.mean_squared_error(y, y_hat)
    mean = tf.reduce_mean(mse, name='mse')

    tf.add_to_collection('losses', mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, inputs, y):
    """Calculate the total loss on a single tower running the model.
    Args:
      scope: unique prefix string identifying the tower, e.g. 'tower_0'
      inputs: 4D tensor of shape [batch_size, users, items].
      y: Labels. 4D tensor of shape [batch_size, users, items].
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    y_hat = inference(inputs)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = model_loss(y_hat, y)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % "tower", '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# Reading dataset

df = pd.read_csv(train_data, sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None)
df = df.drop('timestamp', axis=1)

num_items = df.item.nunique()
num_users = df.user.nunique()

print("USERS: {} ITEMS: {}".format(num_users, num_items))


# Normalize in [0, 1]

r = df['rating'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(r.reshape(-1, 1))
df_normalized = pd.DataFrame(x_scaled)
df['rating'] = df_normalized


# Convert DataFrame in user-item matrix

matrix = df.pivot(index='user', columns='item', values='rating')
matrix.fillna(0, inplace=True)


# Users and items ordered as they are in matrix

users = matrix.index.tolist()
items = matrix.columns.tolist()

matrix = matrix.as_matrix()

print("Matrix shape: {}".format(matrix.shape))

# Network Parameters

num_input = num_items   # num of items

X = tf.placeholder(tf.float64, [None, num_input])

# Targets are the input data.

y_true = X


# Define loss and optimizer, minimize the squared error

optimizer = tf.train.RMSPropOptimizer(learning_rate)

predictions = pd.DataFrame()

# Define evaluation metrics

eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)

with tf.device("/cpu:0"):
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    num_batches = int(matrix.shape[0] / batch_size)

    matrix = np.array_split(matrix, num_batches)

    current_batch = 0

    # Calculate the gradients for each model tower
    tower_grads = []
    reuseVariables = False
    for j in range(0, len(matrix), FLAGS.num_gpus):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuseVariables):
            r = list(range(FLAGS.num_gpus))
            random.shuffle(r)
            for i in r:
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                        # Dequeues one batch for the GPU
                        loss = tower_loss(scope, matrix[current_batch], matrix[current_batch])

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this tower.
                        grads = optimizer.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

                current_batch = j + i
        reuseVariables = True

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op) #variables_averages_op

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as session:
        session.run(init)
        session.run(local_init)

        tf.train.start_queue_runners(sess=session)

        start = time.time()

        for i in range(epochs):
            avg_cost = 0

            _, l = session.run([train_op, loss])
            avg_cost += l

            print("Epoch: {} Loss: {}".format(i + 1, avg_cost))

        print(time.time()-start)

        print("Predictions...")

        matrix = np.concatenate(matrix, axis=0)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            preds = session.run(inference(X), feed_dict={X: matrix})

            predictions = predictions.append(pd.DataFrame(preds))

            predictions = predictions.stack().reset_index(name='rating')
            predictions.columns = ['user', 'item', 'rating']
            predictions['user'] = predictions['user'].map(lambda value: users[value])
            predictions['item'] = predictions['item'].map(lambda value: items[value])

            print("Filtering out items in training set")

            keys = ['user', 'item']
            i1 = predictions.set_index(keys).index
            i2 = df.set_index(keys).index

            recs = predictions[~i1.isin(i2)]
            recs = recs.sort_values(['user', 'rating'], ascending=[True, False])
            recs = recs.groupby('user').head(k)
            recs.to_csv('recs.tsv', sep='\t', index=False, header=False)

