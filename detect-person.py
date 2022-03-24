import argparse
parser = argparse.ArgumentParser(description='person detection v1.0')
parser.add_argument("input", help="Input image file name")
parser.add_argument("-s","--save", help="Save processed image",action="store_true")
parser.add_argument("-m","--mode", help="Select mode 1-image, 2-video",default="1")
args = parser.parse_args()

print("person detection v1.0, loading..")

import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import time
import imutils
import os

import io
from PIL import Image
from mjpeg.client import MJPEGClient

import socket
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

url='http://192.168.88.23:8080/stream?topic=/camera/rgb/image_raw&width=640&height=480&quality=50'


def send_data (data):
    conn.send(data)
    print('Data sent: ' + data.decode())

def receive_data():
    data = ''
    while connectionIsOpen:
        rcvd_data = conn.recv(1)
        if rcvd_data.decode() == '\n':
            # print(data)
            data = ''
        else:
            data += rcvd_data.decode()


connectionIsOpen = False
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connectionIsOpen = True
# receive_thread = threading.Thread(target=receive_data)
# receive_thread.start()



def left_slip():
    conn.connect(('192.168.88.23', 7777))
    time.sleep(1)
    send_data(b'LUA_Base(0,0.5,0)^^^')
    time.sleep(1)
    conn.shutdown(socket.SHUT_RDWR)
    conn.close()
    

def right_slip():
    conn.connect(('192.168.88.23', 7777))
    time.sleep(1)
    send_data(b'LUA_Base(0,-0.5,0)^^^')
    time.sleep(1)
    conn.shutdown(socket.SHUT_RDWR)
    conn.close()


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
			int(boxes[0,i,1]*im_width),
			int(boxes[0,i,2] * im_height),
			int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


if __name__ == "__main__":
    model_path = './model.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7

    if int(args.mode) == 1:
        print('image mode')
        image = cv2.imread(args.input)
        image = imutils.resize(image,width=720)
        boxes, scores, classes, num = odapi.processFrame(image)

        # for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),(134,235,52),2)
            cv2.rectangle(image, (box[1],box[0]-30),(box[1]+125,box[0]),(134,235,52), thickness=cv2.FILLED)
            cv2.putText(image, '  Person '+str(round(scores[i],2)), (box[1],box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225,255,225), 1)

        cv2.imshow("preview", image)
        if args.save:
            print("saving...")
            cv2.imwrite("processed-"+args.input,image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if int(args.mode) == 2:
        
        # cap = cv2.VideoCapture(url)
        # cap.set(cv2.CAP_PROP_FPS,10)
        # print(type(cap))

        # # Create a new client thread
        client = MJPEGClient(url)

        # # Allocate memory buffers for frames
        bufs = client.request_buffers(65536, 50)
        for b in bufs:
            client.enqueue_buffer(b)

        client.start()

        r_i = 0
        l_i = 0
        x_pos = 500
        # time.sleep(7)
        while True:

            buf = client.dequeue_buffer()
            test_buf = client.dequeue_buffer()
            client.enqueue_buffer(test_buf)
            image = Image.open(io.BytesIO(buf.data))
             # print(type(image))
            image = np.array(image) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
            # r, image = cap.read()
            # print(type(image))
            # image = imutils.resize(image,width=720)
            boxes, scores, classes, num = odapi.processFrame(image)

            
            client.enqueue_buffer(buf)
            


            for i in range(len(boxes)):
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),(134,235,52),2)
                    cv2.rectangle(image, (box[1],box[0]-30),(box[1]+125,box[0]),(134,235,52), thickness=cv2.FILLED)
                    cv2.putText(image, '  Person '+str(round(scores[i],2)), (box[1],box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225,255,225), 1)
                    
                    buffer = box[1]
                    if (buffer < x_pos-10):
                        r_i += 1
                        x_pos = buffer
                    elif (buffer > x_pos+10):
                        l_i += 1
                        x_pos = buffer

                    # x_pos = buffer
                    # print(r_i, l_i, x_pos, buffer)
                    
                    if (l_i == 17):
                        r_i = 0
                        print('left')
                        left_slip()
                        cv2.putText(image, '  Person moving left'+str(round(scores[i],2)), (box[1],box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225,255,225), 1)
                    if (r_i == 17):
                        l_i = 0
                        print('right')
                        right_slip()
                        cv2.putText(image, '  Person moving right'+str(round(scores[i],2)), (box[1],box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225,255,225), 1)
            
            cv2.imshow("preview", image)
            key = cv2.waitKey(1)

            


            if key & 0xFF == ord('q'):
                break