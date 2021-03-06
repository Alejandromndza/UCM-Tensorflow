# UCM-Tensorflow


# Experimento de detección de objetos utilizando TensorFlow

Este análisis y estudio del funcionamiento de la detección de objetos en imagenes o videos se ha llevado a cabo gracias al API oficial de TensorFlow disponible en el siguiente enlace: [API](https://github.com/tensorflow/models/tree/master/research/object_detection)

Se ha utilizado el interfaz de anaconda para tener un mejor control de librerias, cuando anaconda este instalado, mediante anaconda prompt ejecutaremos los comandos que vienen en [requeriments.txt](https://github.com/Alejandromndza/UCM-Tensorflow/blob/master/requeriments.txt)

## Como se ha llevado a cabo

Hadware Utilizado:

  1. Procesador Intel Core i7-8750H CPU 2.21GHz
  2. Memoria RAM 16Gb
  3. NVIDIA GeForce GTX 1050 
 
Para llevar a cabo este experimento ha sido necesario el siguiente software:

Sistema opertivo : Windows 10

Librerias:

  1. Python 3.6
  2. Anaconda
  3. CUDA Tool Kit v8 (v9 no compatible con tensorflow v1.4)
  4. CuDNN v6
  5. TensorFlow-GPU v1.4
  6. Pillow
  7. lxml
  8. Cython
  9. Matplotlib
  10. Pandas
  11. OpenCv


## Nota previa Protocol buffers
Este API utiliza protobufs para configurar el modelo y los parámetros de entrenamiento. Por lo tanto se deben de compilar estas librerias, en adelante se detalla como hacerlo.

Estas librerias tienen un lenguaje y plataforma neutral, para serializar datos estructurados.

Para más información acerca de los protobufs [documentación](https://developers.google.com/protocol-buffers/).

Nota:

Por problemas con los protobufs se ha adjuntado la carpeta protos que se deberá situar en la siguiente [ruta](https://github.com/tensorflow/models/tree/master/research/object_detection)

Tras disponer la carpeta protos, en esta [ruta](https://github.com/tensorflow/models/tree/master/research) se deberá ejecutar los siguientes comandos:

                                          python setup.py build
                                          python setup.py install
                                          
Si tenemos todas las librerias correctamente instaladas no deberiamos tener problemas para ejecutar los scripts

## Desarrollo

El desarrollo del clasificador previo para entender el funcionamiento básico de las redes convolucionales, se encuentra en un archivo collaboratory de google esta disponible en el siguiente [enlace](https://colab.research.google.com/drive/1OcOGwLL2juSK3s4SVTmZ1DYQ-a6yyHh7#scrollTo=8Kn_nYo3bxG4)

Para la instalación de protobufs se detalla en el siguiente [enlace](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

El código de **UCM_Converter** ha sido modificado del original xml_to_csv que se encuentra en este [enlace](https://github.com/datitran/raccoon_dataset).

El código de **generate_tfrecord** se ha tomado del mismo repositorio.

Para su funcionamiento debemos de situar las imágenes en los correspondientes directorios (Entrenamiento, Test) 
tras esto ejecutar UCM_Converter.py.

Este script de python creara dos csv que nos servirán para generar los records de tensorflow.

                                    def class_text_to_int(row_label):
                                        if row_label == 'pistol':
                                            return 1
                                        else:
                                            None
            
En caso de querer añadir más clases o clases diferentes, deberemos de modificar esta función por ejemplo: 

                                    def class_text_to_int(row_label):
                                        if row_label == 'cat':
                                            return 1
                                        if row_label == 'dog':
                                            return 2
                                        else:
                                            None


Para ejecutar el script generador de los records de tensorflow debemos ejecutar lo siguiente:

    python generate_tfrecord.py --csv_input=ucm\train_labels.csv --image_dir=ucm\Entrenamiento --output_path=train.record

    python generate_tfrecord.py --csv_input=ucm\test_labels.csv --image_dir=ucm\Test --output_path=test.record

                     
Tras ejecutar el script generate_tfrecord.py tendremos train.record y test.record

Debemos tener un archivo parecido a este con nuestras clases [labelmap.pbtxt](https://github.com/Alejandromndza/UCM-Tensorflow/blob/master/training/labelmap.pbtxt)

Y asegurarnos que nuestra configuración es la adecuada, mirar los paths de inputs.

Si queremos usar un modelo pre-entrenado indicaremos en **fine_tune_checkpoint** el path donde se encuentra el modelo. Los modelos pre-entrenados con los que se ha entrenado los experimentos se encuentran en [enlace](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Todas estas carpetas deben ser utilizadas desplegandolas en la ruta del API señalada anteriormente

Para ello habrá que descargar y situar los archivos en la carpeta indicada anteriormente

Cuando todo este correcto se deberá situar el script de train.py que se encuentra en la carpeta **legacy** y ejecutar este comando. 

    python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_UCM.config
    
Podemos inspeccionar el entrenamiento abriendo otro terminal y mediante este comando tensorboard --logdir=training se generara en nuestro localhost (http://localhost:6060) una interfaz web donde podremos observar el entrenamiento. 

Cuando consideremos que el modelo ya esta entrenado utilizaremos Ctrl + C para parar o por el contrario dejarlo terminar con los steps indicados en la configuración 70.000.

Para exportar el grafo ejecutar el siguiente comando 

    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_UCM.config --trained_checkpoint_prefix training/model.ckpt-Checkpoint --output_directory inference_graph
   
En Checkpoint se deberá poner el último punto donde se guardo nuestro modelo.
   
Resultado 
------------

Para probar nuestro modelo se ha subido el modelo inferido, se pueden ejecutar los scripts UCM-video-py y UCM-webcam.py





