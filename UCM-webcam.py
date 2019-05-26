
# Código adaptado tomado de la siguiente página https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html

import numpy as np
import tensorflow as tf
import sys
import os
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util

# Nombre del modelo 
UCM_MODEL = 'inference_graph'
NUM_CLASSES = 1
PATH_TO_LABELS = os.path.join(os.getcwd(),'training','labelmap.pbtxt')
PATH_TO_CKPT = os.path.join(os.getcwd(),UCM_MODEL,'frozen_inference_graph.pb')
cap = cv2.VideoCapture(0)

# Guardamos el modelo en memoria
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Tensor de imagen
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Cajas de detección
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Puntuaciones del modelo
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Diferentes clases del modelo
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Numero de detecciones
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Detección actual con los diferentes scores y clases
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualizamos los resultados
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break



video.release()
cv2.destroyAllWindows()



