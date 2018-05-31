import tensorflow as tf

# "/home/glory/cnn_lstm_ctc_ocr/data/train/words-000.tfrecord"
# "/home/glory/dataset/EUR_data/train/words-000.tfrecord"
for example in tf.python_io.tf_record_iterator("/home/glory/cnn_lstm_ctc_ocr/data/train/words-000.tfrecord"):
    result = tf.train.Example.FromString(example)
    print(result)
    break