from styx_msgs.msg import TrafficLight
import numpy as np
import os
import tensorflow as tf
import time

class TLClassifier(object):
    def __init__(self):
        self.loaded_model = False
        model = 'Models/model_for_sim/frozen_inference_graph.pb'
        
        model_path = os.path.join(os.path.dirname(__file__), model)
        self.detection_graph = tf.Graph()

        # load model
        with self.detection_graph.as_default():
             od_graph_def = tf.GraphDef()
             with tf.gfile.GFile(model_path, 'rb') as fid:
                  serialized_graph = fid.read()
                  od_graph_def.ParseFromString(serialized_graph)

             self.session = tf.Session(graph=self.detection_graph)

             # Define input and output Tensors for detection_graph
             self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
             self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
             self.detectio_classes = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.loaded_model = True
      
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_width = image.shape[1]
        cropped_image = image[0:int(round(image.shape[0]*0.7)),int(round(image.shape[1]*0.3)):image_width,:]

        image_np_expanded = np.expand_dims(cropped_image, axis = 0)

        # Traffic light detection.
        (boxes, scores, classes, num) = self.session.run([self.detection_boxes, 
                                                          self.detection_scores, 
                                                          self.detection_classes,
                                                          self.num_detections], 
                                                          feed_dict={self.image_tensor: image_np_expanded})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        score_threshold = 0.8
        detection_with_goodscore = np.argwhere(scores > score_threshold)

        if detection_with_goodscore.size > 0:
           class_number = classes[detection_with_goodscore[0]]
           if class_number == 2:
              print('Detected Red light')
              return TrafficLight.RED
           elif class_number == 1:
              print('Detected Green light')
              return TrafficLight.GREEN
           elif class_number == 3:
              print('Detected Yellow light')
              return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN







 




        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
