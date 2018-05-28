import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time

from sample_index_code import sample_regen, industry_gen, whole_stocks_gen
from data_logger import get_logger

THREE_SELECTED_STOCKS_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', 'nav1.xlsx')
TRAIN_TEST_SPLIT_DATE = '2015-12-01'


class StockRNN(object):

    def __init__(self, seq_size=200, small_seq_size=20, input_dimension=1, output_dimension=1, hidden_layer_size=64):
        self.seq_size = seq_size  # default is 200
        self.small_seq_size = small_seq_size # default is 20
        self.input_dimension = input_dimension  # default is 6
        self.hidden_layer_size = hidden_layer_size  # default is 128
        self.output_dimension = output_dimension  # default is 1
        self.batch_id = 0

    def next_batch(self, x, y, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(x):
            self.batch_id = 0
        batch_x = (x[self.batch_id:min(self.batch_id + batch_size, len(x))])
        batch_y = (y[self.batch_id:min(self.batch_id + batch_size, len(y))])
        self.batch_id = min(self.batch_id + batch_size, len(x))
        return batch_x, batch_y

    def random_next_batch(self, x, y, batch_size):
        """
        Return a random indexed batch
        """
        indeces = np.random.randint(0, len(x), batch_size)
        batch_x = x[indeces]
        batch_y = y[indeces]
        return batch_x, batch_y

    def _read_stock_data(self):
        """
        df columns:
        index_code	    trade_date	nav_base	10VOL	20VOL	30VOL	40VOL	50VOL   diff
        002264.SZ	  2011-01-28	1.0000	    1.0000	1.0000	1.0000	1.0000	1.0000  NaN
        """
        df = pd.read_excel(THREE_SELECTED_STOCKS_PATH)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        end_date = TRAIN_TEST_SPLIT_DATE
        start_date = df.date[0]
        print(start_date)
        print(end_date)

        training_data = df[df['date'] < end_date].reset_index()
        test_data = df[df['date'] >= end_date].reset_index()

        return training_data, test_data

    def _gen_small_batch(self, df):
        train_x = np.array([])
        train_y = np.array([])

        for j in range(len(df) - self.small_seq_size + 1):
            small_block = df[j:j + self.small_seq_size]
            train_x = np.append(train_x, small_block[:-1])
            train_y = np.append(train_y, small_block[-1])

        train_x = train_x.reshape((-1, self.small_seq_size - 1, self.input_dimension))
        train_y = train_y.reshape((-1, self.output_dimension))

        return train_x, train_y

    def _data_prepare(self):
        # [['date', 'nav']]
        training_data, test_data = self._read_stock_data()

        print('\n')
        print(len(training_data), len(training_data[training_data['nav'] >= 0]),
              len(training_data[training_data['nav'] < 0]))
        print(len(test_data), len(test_data[test_data['nav'] >= 0]), len(test_data[test_data['nav'] < 0]))

        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        # self.test_y_date = []
        # self.test_y_index_code = []

        print('\n')
        print('Generating matrix data')
        start_time = time.time()
        print('Start timing generating matrix data time')

        self.training_data = np.asarray(training_data['nav'])
        # training_data = np.asarray(training_data['nav'])
        test_data = np.asarray(test_data['nav'])



        # for i in range(len(training_data)-self.seq_size+1):
        #     # self.batch_id = i
        #     large_block = training_data[i:i+self.seq_size]
        #     for j in range(len(large_block)-self.small_seq_size+1):
        #         small_block = training_data[j:j+self.small_seq_size]
        #         self.train_x = np.append(self.train_x, small_block[:-1])
        #         self.train_y = np.append(self.train_y, small_block[-1])

        for j in range(len(test_data)-self.small_seq_size+1):
            small_block = test_data[j:j+self.small_seq_size]
            self.test_x = np.append(self.test_x, small_block[:-1])
            self.test_y = np.append(self.test_y, small_block[-1])

        # for code in self.stock_code:
        #     df = training_data[training_data['index_code'] == code]
        #     train_x = np.asmatrix(df[['nav_base', '10VOL', '20VOL', '30VOL', '40VOL', '50VOL']])
        #     train_y = np.asmatrix(df[['diff']])
        #     for i in range(len(df) - self.seq_size + 1):
        #         self.train_x = np.append(self.train_x, train_x[i:i + self.seq_size])
        #         self.train_y = np.append(self.train_y, train_y[i + self.seq_size - 1])

        # self.train_x = self.train_x.reshape((-1, self.small_seq_size-1, self.input_dimension))
        self.test_x = self.test_x.reshape((-1, self.small_seq_size-1, self.input_dimension))
        # self.train_y = self.train_y.reshape((-1, self.output_dimension))
        self.test_y = self.test_y.reshape((-1, self.output_dimension))

        print("The generating matrix data time --- {} seconds ---".format(time.time() - start_time))
        # print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape)
        # print(len(self.test_y_date))

    def _create_placeholders(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None, self.small_seq_size-1, self.input_dimension], name='X_input')
            self.Y = tf.placeholder(tf.float32, [None, self.output_dimension], name='Y_input')

    def _create_rnn(self):
        W = tf.Variable(tf.random_normal([self.hidden_layer_size, self.output_dimension]), name='W')
        b = tf.Variable(tf.random_normal([self.output_dimension]), name='b')
        with tf.variable_scope('cell'):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_layer_size)
        with tf.variable_scope('rnn'):
            x = tf.unstack(self.X, self.small_seq_size-1, 1)
            outputs, states = tf.nn.static_rnn(cell, x, dtype=tf.float32)
        print('The outputs is {}'.format(tf.shape(outputs)))

        out = tf.matmul(outputs[-1], W) + b
        print(out)

        return out

    def train_pred_rnn(self):

        self._create_placeholders()
        y_hat = self._create_rnn()
        self._data_prepare()

        loss = tf.losses.mean_squared_error(self.Y, y_hat)
        class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(self.Y > 0, tf.float32),
            logits=tf.cast(y_hat > 0, tf.float32)))  # default is y_hat, no tf.cast

        total_loss = loss + class_loss * 0.01  # default is 0.1
        train_optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)  # default lr is 0.001

        # MAPE = tf.reduce_mean(tf.abs(tf.div((self.Y - y_hat), self.Y)))
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(self.Y - y_hat)))

        # saver = tf.train.Saver(tf.global_variables())

        # batch_size = 200

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(4000):
                if self.batch_id == len(self.training_data)-self.seq_size+1:
                    self.batch_id = 0
                x_batch, y_batch = self._gen_small_batch(self.training_data[self.batch_id: self.batch_id+self.seq_size])
                # x_batch, y_batch = self.random_next_batch(self.train_x, self.train_y, batch_size)
                self.batch_id += 1
                feed_dict = {self.X: x_batch, self.Y: y_batch}
                _, train_loss, train_class_loss, train_total_loss, rmse = sess.run(
                    [train_optim, loss, class_loss, total_loss, RMSE], feed_dict=feed_dict)
                # _, train_loss = sess.run([train_optim,loss], feed_dict=feed_dict)
                if step % 50 == 0:
                    print("Step: {0}, regression loss: {1}, class loss: {2}, total_loss: {3}, RMSE: {4}".format(step,
                                                                                                                train_loss,
                                                                                                                train_class_loss,
                                                                                                                train_total_loss,
                                                                                                                rmse))
                    # saver.save(sess, save_path=os.path.dirname(__file__))

            test_loss = sess.run(total_loss, feed_dict={self.X: self.test_x, self.Y: self.test_y})
            print("test loss: {}".format(test_loss))

            # predictions = sess.run(tf.squeeze(y_hat))
            predictions = y_hat.eval(feed_dict={self.X: self.test_x}, session=sess)
            predictions = predictions.ravel()
            # pred_df = pd.DataFrame(
            #     {'stock_code': self.test_y_index_code, 'date': self.test_y_date, 'predictions': predictions},
            #     columns=['stock_code', 'date', 'predictions'])

            # writer = pd.ExcelWriter('lstm_pred_random_5.xlsx')
            # pred_df.to_excel(writer, 'Sheet1')
            # writer.save()
            # print(pred_df)
            print(list(predictions))
            print(len(predictions))


if __name__ == "__main__":
    stock = StockRNN()

    start_time = time.time()
    stock.train_pred_rnn()
    print('The total time is ----- {} ----- seconds'.format(time.time() - start_time))
