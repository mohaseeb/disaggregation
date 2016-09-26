from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator



class LstmDisaggregator(BaseEstimator):
    def __init__(self, init_scale=0.08, max_grad_norm=25, num_steps=200, drop_prob=0.2, num_layers=2,
                 hidden_size=70, max_iterations=1000, batch_size=20, eta=8e-3, models_dir="models/",
                 model_name_prefix="trained_model"):
        tf.reset_default_graph()
        self.output_classes = 2
        self.y_val = None
        self.X_val = None
        self.keep_prob = None
        self.initial_state = None
        self.targets = None
        self.input_data = None
        self.test_accuracy = None
        self.state = None
        self.predictions = None
        self.session = None
        self.models_dir = models_dir
        self.model_name_prefix = model_name_prefix
        """Hyperparamaters"""
        self.init_scale = init_scale  # Initial scale for the states
        self.max_grad_norm = max_grad_norm  # Clipping of the gradient before update
        self.num_steps = num_steps  # Number of steps to backprop over at every batch
        self.num_outputs = 2
        self.drop_prob = drop_prob
        self.num_layers = num_layers  # Number of stacked LSTM layers
        self.hidden_size = hidden_size  # Number of entries of the cell state of the LSTM
        self.max_iterations = max_iterations  # Maximum iterations to train
        self.batch_size = batch_size  # Batch size
        self.eta = eta

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        if self.X_val is None:
            print("using the test data as validation set")
            self.X_val = X
            self.y_val = y
        """Place holders"""
        self.input_data = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, 1], name='input_data')
        self.targets = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, self.num_outputs], name='Targets')

        # Used later on for drop_out. At testtime, we pass 1.0
        self.keep_prob = tf.placeholder("float", name='Drop_out_keep_prob')

        initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)
        with tf.variable_scope("model", initializer=initializer):
            """Define the basis LSTM"""
            with tf.name_scope("LSTM_setup"):
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
                cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
                # Initialize the zero_state. Note that it has to be run in session-time
                self.initial_state = cell.zero_state(self.batch_size, tf.float32)

            print(self.input_data.get_shape())
            print(self.targets.get_shape())

            """Define the recurrent nature of the LSTM"""
            cell_outputs = []
            with tf.name_scope("LSTM"):
                self.state = self.initial_state
                with tf.variable_scope("LSTM_state"):

                    for time_step in range(self.num_steps):
                        # Re-use variables only after first time-step
                        if time_step > 0:
                            tf.get_variable_scope().reuse_variables()
                        # Now cell_output is size [batch_size x hidden_size]
                        (cell_output, self.state) = cell(self.input_data[:, time_step, :], self.state)
                        cell_outputs.append(cell_output)

        """Generate a classification from the last cell_output"""
        # Note, this is where timeseries classification differs from sequence to sequence
        # modelling. We only output to Softmax at last time step
        with tf.name_scope("Softmax"):
            with tf.variable_scope("Softmax_params"):
                softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.output_classes])
                softmax_b = tf.get_variable("softmax_b", [self.output_classes])
                loss = 0.0
                predictions = []
                for time_step in range(self.num_steps):
                    logits = tf.matmul(cell_outputs[time_step], softmax_w) + softmax_b
                    predictions.append(logits)
                    loss = tf.add(loss, tf.nn.l2_loss(logits - self.targets[:, time_step, :]))
                self.predictions = tf.pack(predictions)
            cost = tf.reduce_sum(loss) / self.batch_size

        """Optimizer"""
        with tf.name_scope("Optimizer"):
            tvars = tf.trainable_variables()
            # We clip the gradients to prevent explosion
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
            gradients = zip(grads, tvars)
            train_op = optimizer.apply_gradients(gradients)

        # Collect the costs in a numpy fashion
        epochs = np.floor(self.batch_size * self.max_iterations / X.shape[0])
        print('Train with approximately %d epochs' % (epochs))

        """Training"""
        self.session = tf.Session()
        # initialize all variables
        tf.initialize_all_variables().run(session=self.session)

        step = 0
        for i in range(self.max_iterations):

            # Calculate some sizes
            N = X.shape[0]

            # Sample batch for training
            X_batch, y_batch = self._sample_batch(X, y, self.batch_size)
            self.state = self.initial_state.eval(session=self.session)  # Fire up the LSTM

            # Next line does the actual training
            self.session.run(train_op,
                             feed_dict={self.input_data: X_batch, self.targets: y_batch,
                                        self.initial_state: self.state,
                                        self.keep_prob: self.drop_prob})

            if i % 100 == 0:
                # Evaluate training performance
                X_batch, y_batch = self._sample_batch(X, y, self.batch_size)

                train_cost = self.session.run(cost, feed_dict={self.input_data: X_batch, self.targets: y_batch,
                                                               self.initial_state: self.state,
                                                               self.keep_prob: 1})

                # Evaluate validation performance
                X_batch, y_batch = self._sample_batch(self.X_val, self.y_val, self.batch_size)
                val_cost = self.session.run(cost,
                                            feed_dict={self.input_data: X_batch, self.targets: y_batch,
                                                       self.initial_state: self.state,
                                                       self.keep_prob: 1})

                print(
                    'At %d out of %d train cost is %.3f and val acc is %.3f' % (
                        i, self.max_iterations, train_cost, val_cost))

                step += 1

        """Saving the trained model"""
        tf.train.write_graph(self.session.graph_def, self.models_dir,
                             self.model_name_prefix + "_" + str(self.max_iterations) + ".pb", as_text=False)

    def predict(self, X):
        num_batch = np.floor(X.shape[0] / self.batch_size)
        predictions = np.zeros((X.shape[0], X.shape[1], 2))
        for batch in range(int(num_batch)):
            x = X[batch * self.batch_size:(batch + 1) * self.batch_size, :, :]
            batch_predictions = self.session.run(self.predictions,
                                                 feed_dict={
                                                     self.input_data: x,
                                                     self.initial_state: self.state,
                                                     self.keep_prob: 1})
            # above has shape (200,20,2) want to convert it to (20,200,2)

            predictions[batch * self.batch_size:(batch + 1) * self.batch_size, :, :] = \
                np.swapaxes(batch_predictions, 0, 1)

        return predictions

    def score(self, X, y):
        """the graph supports only data on the batch_size. This is why here i need to pad the test set so i can test all
         the data; this needs to be general"""
        N = X.shape[0]
        pad_length = self.batch_size - np.mod(N, self.batch_size)
        if pad_length == 0:
            return np.mean(np.equal(self.predict(X), y[:, 0]))
        else:
            X_pad = np.repeat([X[-1]], pad_length, axis=0)
            X_padded = np.concatenate((X, X_pad), axis=0)
            y_pad = np.repeat([y[-1]], pad_length, axis=0)
            y_padded = np.concatenate((y, y_pad), axis=0)
            y_predicted = self.predict(X_padded)[0:N]
            return np.mean(np.equal(y_predicted, y_padded[0:N, 0]))

    def _sample_batch(self, X_train, y_train, batch_size):
        """ Function to sample a batch for training"""
        N, data_len, _ = X_train.shape
        ind_N = np.random.choice(N, batch_size, replace=False)
        # form batch
        X_batch = X_train[ind_N, :, :]
        y_batch = y_train[ind_N, :, :]
        return X_batch, y_batch
