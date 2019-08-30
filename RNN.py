import tensorflow as tf
import tensorflow_datasets as tfds

vocab_size = 10000
imdb = tf.keras.datasets.imdb
(train_data,
 train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
