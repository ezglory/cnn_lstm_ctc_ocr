import os
import tensorflow as tf
import math

"""Each record within the TFRecord file is a serialized Example proto. 
The Example proto contains the following fields:
  image/encoded: string containing JPEG encoded grayscale image
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/filename: string containing the basename of the image file
  image/labels: list containing the sequence labels for the image text
  image/text: string specifying the human-readable version of the text
"""

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

jpeg_data = tf.placeholder(dtype=tf.string)
jpeg_decoder = tf.image.decode_jpeg(jpeg_data,channels=1)

kernel_sizes = [5,5,3,3,3,3] # CNN kernels for image reduction

# Minimum allowable width of image after CNN processing
min_width = 20

def calc_seq_len(image_width):
    """Calculate sequence length of given image after CNN processing"""
    
    conv1_trim =  2 * (kernel_sizes[0] // 2)
    fc6_trim = 2*(kernel_sizes[5] // 2)
    
    after_conv1 = image_width - conv1_trim 
    after_pool1 = after_conv1 // 2
    after_pool2 = after_pool1 // 2
    after_pool4 = after_pool2 - 1 # max without stride
    after_fc6 =  after_pool4 - fc6_trim
    seq_len = 2*after_fc6
    return seq_len

seq_lens = [calc_seq_len(w) for w in range(1024)]

def gen_data(input_base_dir, image_list_filename, output_filebase,
             num_shards=1000,start_shard=0):
    """ Generate several shards worth of TFRecord data """
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    sess = tf.Session(config=session_config)
    image_filenames = get_image_filenames(os.path.join(input_base_dir,
                                                       image_list_filename))
    if num_shards > 1:
        num_digits = math.ceil( math.log10( num_shards - 1 ))
    else:
        num_digits = 3

    num_digits = max(3,num_digits)
    shard_format = '%0'+ ('%d'%num_digits) + 'd' # Use appropriate # leading zeros
    images_per_shard = int(math.ceil( len(image_filenames) / float(num_shards) ))
    
    for i in range(start_shard,num_shards):
        start = i*images_per_shard
        end   = (i+1)*images_per_shard
        out_filename = output_filebase+'-'+(shard_format % i)+'.tfrecord'
        #if os.path.isfile(out_filename): # Don't recreate data if restarting
        #    continue
        print (str(i),'of',str(num_shards),'[',str(start),':',str(end),']',out_filename)
        gen_shard(sess, input_base_dir, image_filenames[start:end], out_filename)
    # Clean up writing last shard
    start = num_shards*images_per_shard
    out_filename = output_filebase+'-'+(shard_format % num_shards)+'.tfrecord'
    print (str(i+1),'of',str(num_shards),'[',str(start),':]',out_filename)
    gen_shard(sess, input_base_dir, image_filenames[start:], out_filename)

    sess.close()

def gen_shard(sess, input_base_dir, image_filenames, output_filename):
    """Create a TFRecord file from a list of image filenames"""
    if len(image_filenames) == 0:
        return
    writer = tf.python_io.TFRecordWriter(output_filename)
    
    for filename in image_filenames:
        path_filename = os.path.join(input_base_dir,filename)
        if os.stat(path_filename).st_size == 0:
            print('SKIPPING',filename)
            continue
        try:
            image_data,height,width = get_image(sess,path_filename)
            text,labels = get_text_and_labels(filename)
            if is_writable(width,text):
                example = make_example(filename, image_data, labels, text, 
                                       height, width)
                writer.write(example.SerializeToString())
            else:
                print('SKIPPING',filename)
        except Exception as e:
            print(e)
            # Some files have bogus payloads, catch and note the error, moving on
            print('ERROR',filename)
    writer.close()


def get_image_filenames(image_list_filename):
    """ Given input file, generate a list of relative filenames"""
    filenames = []
    with open(image_list_filename) as f:
        for line in f:
            filenames.append(line.strip())
    return filenames

def get_image(sess,filename):
    """Given path to an image file, load its data and size"""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    image = sess.run(jpeg_decoder,feed_dict={jpeg_data: image_data})
    height = image.shape[0]
    width = image.shape[1]
    return image_data, height, width

def is_writable(image_width,text):
    """Determine whether the CNN-processed image is longer than the string"""
    return (image_width > min_width) and (len(text) <= seq_lens[image_width])
    
def get_text_and_labels(filename):
    """ Extract the human-readable text and label sequence from image filename"""
    # Ground truth string lines embedded within base filename between underscores
    # 2697/6/466_MONIKER_49537.jpg --> MONIKER
    text = os.path.basename(filename).split('_',2)[1]
    # Transform string text to sequence of indices using charset, e.g.,
    # MONIKER -> [12, 14, 13, 8, 10, 4, 17]
    labels = [out_charset.index(c) for c in list(text)]
    return text,labels

def make_example(filename, image_data, labels, text, height, width):
    """Build an Example proto for an example.
    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_data: string, JPEG encoding of grayscale image
    labels: integer list, identifiers for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
        'image/labels': _int64_feature(labels),
        'image/height': _int64_feature([height]),
        'image/width': _int64_feature([width]),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'text/string': _bytes_feature(tf.compat.as_bytes(text)),
        'text/length': _int64_feature([len(text)])
    }))
    return example


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def main(argv=None):
    gen_data('/home/glory/dataset/EUR_imgs/train', 'list.txt', '/home/glory/dataset/EUR_tf/train/words', 2)
    gen_data('/home/glory/dataset/EUR_imgs/test', 'list.txt', '/home/glory/dataset/EUR_tf/test/words', 2)
    gen_data('/home/glory/dataset/EUR_imgs/val', 'list.txt', '/home/glory/dataset/EUR_tf/val/words', 2)


if __name__ == '__main__':
    main()
