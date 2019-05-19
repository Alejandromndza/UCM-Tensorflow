# UCM-Tensorflow

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
                                            return 1
                                        else:
                                            None
                                            
                     
