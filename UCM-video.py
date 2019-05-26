
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
VIDEO_NAME = 'prueba1.mp4'
NUM_CLASSES = 1
CWD = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD,'training','labelmap.pbtxt')
PATH_TO_CKPT = os.path.join(CWD,UCM_MODEL,'frozen_inference_graph.pb')
PATH_TO_VIDEO = os.path.join(CWD,VIDEO_NAME)
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

video = cv2.VideoCapture(PATH_TO_VIDEO)
frame_width = int(video.get(3))
frame_height = int(video.get(4))

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(video.isOpened()):

    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

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
        [boxes, scores, classes, num_detections],feed_dict={image_tensor: frame_expanded})
 
    # Visualizamos los resultados
    vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
    
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
out.realease()
cv2.destroyAllWindows()



