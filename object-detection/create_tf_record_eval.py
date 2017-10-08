import os
import io
import tensorflow as tf
import PIL.Image
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('data_path', '', 'Path to images and labels')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

data_path = "images"

def create_tf_example(File, labels):

    height = 200 # Image height
    width = 300 # Image width
    img_path = os.path.join(FLAGS.data_path, File)
    with tf.gfile.GFile(img_path, 'rb') as fid:
      encoded_jpg = fid.read()
    filename = File.encode('utf8') # Filename of the image. Empty if image is not from file
    encoded_image_data = bytes(encoded_jpg) # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
               # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
               # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in labels:
        xmins.append(box[0] / width)
        ymins.append(box[1] / height)
        xmaxs.append(box[2] / width)
        ymaxs.append(box[3] / height)
        classes_text.append('Crater'.encode('utf8'))
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    all_files = os.listdir(FLAGS.data_path)
    labels_file = open(os.path.join(FLAGS.data_path, "GTF.lms"), "r")
    labels = {}
    num_files = len(all_files)
    train_num = int(num_files * 0.80)

    for line in labels_file:
        split = line.split(" : ")[:-1]
        values = []
        for i in range(2, len(split), 4):
            box = (int(split[i]) // 2, int(split[i + 1]) // 2, int(split[i + 2]) // 2, int(split[i + 3]) // 2)
            values.append(box)
        labels[split[0]] = values


    for File in all_files[train_num:]:
        if File[-3:] == "jpg":
            tf_example = create_tf_example(File, labels[File])
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
