import numpy as np
import tensorflow as tf
import classifier, training
tf.enable_eager_execution()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


def data_encoder(data):
  return np.array([1 - data, data]).transpose([1, 2, 0])


def to_one_hot(labels, n_labels=10):
  one_hot = np.zeros((len(labels), n_labels))
  one_hot[np.arange(len(labels)), labels] = 1
  return one_hot


n_labels = len(np.unique(y_train))

# Flatten and normalize
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) / 255.0
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) / 255.0
# Encode
x_train = data_encoder(x_train)
x_test = data_encoder(x_test)
y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

print(n_labels)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


mps = classifier.MatrixProductState(n_sites=x_train.shape[1] + 1,
                                    n_labels=n_labels,
                                    d_phys=x_train.shape[2],
                                    d_bond=12)

# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer =  tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)

mps, history = training.fit(mps, optimizer, x_train[:1000], y_train[:1000],
                            n_epochs=10, batch_size=50, n_message=1)
