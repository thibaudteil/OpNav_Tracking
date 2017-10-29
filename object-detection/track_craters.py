'''
Code to track craters using the Object Detection API
'''
import cv2
import numpy as np
import os
import tensorflow as tf
import time

from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model we are using
MODEL_NAME = 'faster_rcnn_resnet101_coco'

# Path to frozen detection graph
# This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join('exported-model-02','frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('label_map.pbtxt')
NUM_CLASSES = 1

# Load all labels
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Path to video
PATH_TO_TEST_IMAGES_DIR = 'test-images'
vid = 'video1.mp4'
video_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, vid)

# Read the exported graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Run the exported graph
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Get input from video
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            start = time.time()
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            end = time.time()
            # Tag for image
            tag = 3
            print(end - start)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                tag,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1)

            # Show video
            cv2.imshow('Crater Detector', cv2.resize(image_np, (800,600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
