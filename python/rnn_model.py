import tensorflow as tf
import pandas as pd
import numpy as np
import os

# from sample_index_code import sample_regen
# from data_logger import get_logger


class StockRNN(object):

    def __init__(self, seq_size=200, input_dimension=6, hidden_layer_size=3):
        self.seq_size = seq_size
        self.input_dimension = input_dimension
        self.hidde_layer_size = hidden_layer_size
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

    def _read_stock_data(self):
        '''
        df columns:
        index_code	    trade_date	nav_base	10VOL	20VOL	30VOL	40VOL	50VOL   diff
        002264.SZ   	2011-01-28	1.0000	    1.0000	1.0000	1.0000	1.0000	1.0000  NaN
        '''
        # num = 3
        # df = sample_regen(num)
        df = pd.read_excel(r'/Users/yudongsi/Desktop/Workspace/rnnpred/data/output.xlsx')

        train_test_split_date = '2016-06-01'

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

        # self.train_x = np.ones(shape=(1000,200,6))
        # extra = np.zeros(shape=(1000, 200, 6))
        # self.train_x = np.append(self.train_x, extra)
        # self.train_x = self.train_x.reshape(2000, 200, 6)
        # self.train_y = np.ones(shape=(2000,1))
        # self.test_x = np.ones(shape=(500,200,6))
        # self.test_y = np.ones(shape=(500,1))

        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.test_y_date = []
        self.test_y_index_code = []

        print('Generating matrix data')

        # for code in self.stock_code:
        #     df = df[df['index_code'] == code]

        for code in self.stock_code:
            df = training_data[training_data['index_code'] == code]
            train_x = np.asmatrix(df[['nav_base', '10VOL', '20VOL', '30VOL', '40VOL', '50VOL']])
            train_y = np.asmatrix(df[['diff']])
            for i in range(len(df) - 200 + 1):
                self.train_x = np.append(self.train_x, train_x[i:i + 200])
                self.train_y = np.append(self.train_y, train_y[i + 200 - 1])

        for code in self.stock_code:
            df = test_data[test_data['index_code'] == code]
            test_x = np.asmatrix(df[['nav_base', '10VOL', '20VOL', '30VOL', '40VOL', '50VOL']])
            test_y = np.asmatrix(df[['diff']])
            for i in range(len(df) - 200 + 1):
                self.test_x = np.append(self.test_x, test_x[i:i + 200])
                self.test_y = np.append(self.test_y, test_y[i + 200 - 1])
                self.test_y_date.append(df.iloc[i + 200 - 1]['trade_date'])
                self.test_y_index_code.append(df.iloc[i + 200 - 1]['index_code'])

        self.train_x = self.train_x.reshape((-1, 200, 6))
        self.test_x = self.test_x.reshape((-1, 200, 6))
        self.train_y = self.train_y.reshape((-1, 1))
        self.test_y = self.test_y.reshape((-1, 1))

        print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape)
        print(len(self.test_y_date))

    def _create_placeholders(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None, self.seq_size, self.input_dimension], name='X_input')
            self.Y = tf.placeholder(tf.float32, [None, 1], name='Y_input')

    def _create_rnn(self):
        W = tf.Variable(tf.random_normal([self.hidde_layer_size, 1]), name='W')
        b = tf.Variable(tf.random_normal([1]), name='b')
        with tf.variable_scope('cell'):
            cell = tf.contrib.rnn.BasicRNNCell(self.hidde_layer_size)
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
        train_optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        MAPE = tf.reduce_mean(tf.abs(tf.div((self.Y - y_hat), self.Y)))
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(self.Y - y_hat)))

        # total_error = tf.reduce_sum(tf.square(self.Y - tf.reduce_mean(self.Y)))
        # unexplained_error = tf.reduce_sum(tf.square(self.Y - y_hat))
        # R_squared = (1 - tf.div(unexplained_error, total_error))

        saver = tf.train.Saver(tf.global_variables())

        batch_size = 200

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(2000):
                x_batch, y_batch = self.next_batch(self.train_x, self.train_y, batch_size)
                feed_dict = {self.X: x_batch, self.Y: y_batch}
                _, train_loss, mape, rmse = sess.run([train_optim, loss, MAPE, RMSE], feed_dict=feed_dict)
                # _, train_loss = sess.run([train_optim,loss], feed_dict=feed_dict)
                if step % 50 == 0:
                    print("Step: {0}, training loss: {1}, MAPR: {2}, RMSE: {3}".format(step, train_loss, mape, rmse))
                    saver.save(sess, save_path=os.path.dirname(__file__))

            test_loss = sess.run(loss, feed_dict={self.X: self.test_x, self.Y: self.test_y})
            print("test loss: {}".format(test_loss))

            predictions = y_hat.eval(feed_dict={self.X: self.test_x})
            predictions = predictions.ravel()
            pred_df = pd.DataFrame({'date': self.test_y_date, 'predictions': predictions, 'index_code': self.test_y_index_code},
                                   columns=['index_code', 'date', 'predictions'])
            print(pred_df)


if __name__ == "__main__":
    stock = StockRNN()

    stock.train_pred_rnn()
