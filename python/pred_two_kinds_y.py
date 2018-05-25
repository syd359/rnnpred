import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time

from sample_index_code import sample_regen, industry_gen, whole_stocks_gen
from data_logger import get_logger


THREE_SELECTED_STOCKS_PATH = r'C:\Users\syd13065\PycharmProjects\rnnnav\data\output.xlsx'



class StockRNN(object):

    def __init__(self, seq_size=100, input_dimension=6, output_dimension=2, hidden_layer_size=128):
        self.seq_size = seq_size # default is 200
        self.input_dimension = input_dimension # default is 6
        self.hidde_layer_size = hidden_layer_size # default is 128
        self.output_dimension = output_dimension # default is 1

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
        '''
        Return a random indexed batch
        '''
        indeces = np.random.randint(0, len(x), batch_size)
        batch_x = x[indeces]
        batch_y = y[indeces]
        return batch_x, batch_y

    def _read_stock_data(self):
        '''
        df columns:
        index_code	    trade_date	nav_base	10VOL	20VOL	30VOL	40VOL	50VOL   diff
        002264.SZ	  2011-01-28	1.0000	    1.0000	1.0000	1.0000	1.0000	1.0000  NaN
        '''
        # num = 5
        # df = sample_regen(num)

        df = pd.read_excel(THREE_SELECTED_STOCKS_PATH)

        train_test_split_date = '2015-08-01'

        end_date = train_test_split_date
        start_date = df.trade_date[0]
        print(start_date)
        print(end_date)

        self.stock_code = list(set(df.index_code))
        print(self.stock_code)

        training_data = df[df['trade_date'] < end_date].reset_index()
        test_data = df[df['trade_date'] >= end_date].reset_index()

        return training_data, test_data

        # return df, train_test_split_date

    def _data_prepare(self):
        # [['nav_base', '10VOL', '20VOL', '30VOL', '40VOL', '50VOL', 'diff']]
        training_data, test_data = self._read_stock_data()

        print('\n')
        print(len(training_data), len(training_data[training_data['diff'] >= 0]),
              len(training_data[training_data['diff'] < 0]))
        print(len(test_data), len(test_data[test_data['diff'] >= 0]), len(test_data[test_data['diff'] < 0]))

        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.test_y_date = []
        self.test_y_index_code = []

        print('\n')
        print('Generating matrix data')
        start_time = time.time()
        print('Start timing generating matrix data time')

        # for code in self.stock_code:
        #     df = df[df['index_code'] == code]

        for code in self.stock_code:
            df = training_data[training_data['index_code'] == code]
            train_x = np.asmatrix(df[['nav_base', '10VOL', '20VOL', '30VOL', '40VOL', '50VOL']])
            train_y = np.asmatrix(df[['diff']])
            for i in range(len(df) - self.seq_size + 1):
                self.train_x = np.append(self.train_x, train_x[i:i + self.seq_size])
                self.train_y = np.append(self.train_y, train_y[i + self.seq_size - 1])

        for code in self.stock_code:
            df = test_data[test_data['index_code'] == code]
            test_x = np.asmatrix(df[['nav_base', '10VOL', '20VOL', '30VOL', '40VOL', '50VOL']])
            test_y = np.asmatrix(df[['nav_base','diff']])
            for i in range(len(df) - self.seq_size + 1):
                self.test_x = np.append(self.test_x, test_x[i:i + self.seq_size])
                self.test_y = np.append(self.test_y, test_y[i + self.seq_size - 1])
                self.test_y_date.append(df.iloc[i + self.seq_size - 1]['trade_date'])
                self.test_y_index_code.append(df.iloc[i + self.seq_size - 1]['index_code'])

        self.train_x = self.train_x.reshape((-1, self.seq_size, self.input_dimension))
        self.test_x = self.test_x.reshape((-1, self.seq_size, self.input_dimension))
        self.train_y = self.train_y.reshape((-1, 1))
        self.test_y = self.test_y.reshape((-1, 1))

        print("The generating matrix data time --- {} seconds ---".format(time.time() - start_time))
        print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape)
        print(len(self.test_y_date))

    def _create_placeholders(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None, self.seq_size, self.input_dimension], name='X_input')
            self.Y = tf.placeholder(tf.float32, [None, self.output_dimension], name='Y_input')

    def _create_rnn(self):
        W = tf.Variable(tf.random_normal([self.hidde_layer_size, self.output_dimension]), name='W')
        b = tf.Variable(tf.random_normal([self.output_dimension]), name='b')
        with tf.variable_scope('cell'):
            cell = tf.contrib.rnn.LSTMCell(self.hidde_layer_size)
        with tf.variable_scope('rnn'):
            x = tf.unstack(self.X, self.seq_size, 1)
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
            labels=tf.cast(self.Y > 0, tf.float32), logits=tf.cast(y_hat > 0, tf.float32))) # default is y_hat, no tf.cast

        total_loss = loss + class_loss*0.01 # default is 0.1
        train_optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss) # default lr is 0.001

        # MAPE = tf.reduce_mean(tf.abs(tf.div((self.Y - y_hat), self.Y)))
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(self.Y - y_hat)))


        # saver = tf.train.Saver(tf.global_variables())

        batch_size = 200

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(3000):
                x_batch, y_batch = self.random_next_batch(self.train_x, self.train_y, batch_size)
                feed_dict = {self.X: x_batch, self.Y: y_batch}
                _, train_loss, train_class_loss, train_total_loss, rmse = sess.run([train_optim, loss, class_loss, total_loss, RMSE], feed_dict=feed_dict)
                # _, train_loss = sess.run([train_optim,loss], feed_dict=feed_dict)
                if step % 50 == 0:
                    print("Step: {0}, regression loss: {1}, class loss: {2}, total_loss: {3}, RMSE: {4}".format(step, train_loss, train_class_loss, train_total_loss,
                                                                                       rmse))
                    # saver.save(sess, save_path=os.path.dirname(__file__))

            test_loss = sess.run(total_loss, feed_dict={self.X: self.test_x, self.Y: self.test_y})
            print("test loss: {}".format(test_loss))

            # predictions = sess.run(tf.squeeze(y_hat))
            predictions = y_hat.eval(feed_dict={self.X: self.test_x}, session=sess)
            predictions = predictions.ravel()
            pred_df = pd.DataFrame(
                {'stock_code': self.test_y_index_code, 'date': self.test_y_date, 'predictions': predictions},
                columns=['stock_code', 'date', 'predictions'])

            writer = pd.ExcelWriter('lstm_pred_random_5.xlsx')
            pred_df.to_excel(writer, 'Sheet1')
            writer.save()
            # print(pred_df)
            print(len(pred_df), len(pred_df[pred_df['predictions'] >= 0]), len(pred_df[pred_df['predictions'] < 0]))


if __name__ == "__main__":
    stock = StockRNN()

    start_time = time.time()
    stock.train_pred_rnn()
    print('The total time is ----- {} ----- seconds'.format(time.time() - start_time))
