import io
import cv2
import numpy as np
from PIL import Image
from mjpeg.client import MJPEGClient

url='http://192.168.88.23:8080/stream?topic=/camera/rgb/image_raw&width=640&height=480&quality=50'


# Create a new client thread
client = MJPEGClient(url)

# Allocate memory buffers for frames
bufs = client.request_buffers(65536, 50)
for b in bufs:
    client.enqueue_buffer(b)

# Start the client in a background thread
client.start()
i = 0
while True:
    i+=1
    buf = client.dequeue_buffer()

    image = Image.open(io.BytesIO(buf.data))
    # print(type(image))
    cvimage = np.array(image) 
    cv2.imshow('DEEZ NUTS', cvimage)
    cv2.waitKey(5) 
    client.enqueue_buffer(buf)


cv2.destroyAllWindows() 
