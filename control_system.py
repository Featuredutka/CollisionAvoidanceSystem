import time
import socket
import cv2 as cv

import drone as dp
import detect_lane as dl
import fuzzy_system as fs
import situation_rules as sr


URL='http://192.168.88.25:8080/stream?topic=/camera/rgb/image_raw&width=640&height=480&quality=50'


def send_data (data, conn):
    conn.send(data)
    # print('Data sent: ' + data.decode())

def receive_data():
    data = ''
    while connectionIsOpen:
        rcvd_data = conn.recv(1)
        if rcvd_data.decode() == '\n':
            # print(data)
            data = ''
        else:
            data += rcvd_data.decode()

def arm_def_pos(): #TODO Add arm lua script call to position it for further work
    send_data(b'LUA_ManipDeg(0, 160, 61, -139, 177, 169)^^^')

def mild_left():
    send_data(b'LUA_Base(0.2,0.2,0)^^^')
    time.sleep(1)
    send_data(b'LUA_Base(0,0,0)^^^')
    
def mild_right():
    send_data(b'LUA_Base(0.2,-0.2,0)^^^')
    time.sleep(1)
    send_data(b'LUA_Base(0,0,0)^^^')

def right_slip():
    send_data(b'LUA_Base(0.2,-0.2,0)^^^')
    time.sleep(1)
    send_data(b'LUA_Base(0,0,0)^^^')

def left_slip():
    send_data(b'LUA_Base(0.2,-0.2,0)^^^')
    time.sleep(1)
    send_data(b'LUA_Base(0,0,0)^^^')
   
def forward_movement():
    send_data(b'LUA_Base(0.2,0,0)^^^')
    time.sleep(1)
    send_data(b'LUA_Base(0,0,0)^^^')

def backward_movement():
    send_data(b'LUA_Base(-0.2,0,0)^^^')
    time.sleep(1)
    send_data(b'LUA_Base(0,0,0)^^^')

def infinite_forward():
    send_data(b'LUA_Base(0.2,0,0)^^^')

def stop(conn):
    send_data(b'LUA_Base(0,0,0)^^^', conn)

def main():
    # Connection routine
    connectionIsOpen = False
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect(('192.168.88.25', 7777))
    time.sleep(1)

    cap = cv.VideoCapture(URL)

    """
    Main loop 

    1) Human detector
        1.1) Human distance
        1.2) Human direction
    2) Lane detector
        2.1) Robot relative position
    3) Sign detector TODO
        3.1) Sign detection check

    All this info goes into the fuzzy regulator which returns 2 values TODO
    1) Robot desired speed
    2) RObot desired direction

    This info should be translated into commands for YouBot TODO
    """

    #TODO Since infinite loop is here maybe I ned to calculate direction here as well
    detector_class = dp.ObjectDetection("test.avi")
    _, frame2 = cap.read()

    r_i = 0
    l_i = 0

    x_pos = 500

    while True:

        _, frame = cap.read()

        #################################################################### 1
        frame1, x1, y1, x2, y2 = detector_class.detect_lane(frame)
        p_dist = (100000 - ((x2-x1) * (y2-y1)))/1000
        p_dir = 2
        buffer = x1

        #################################################################### 2
        try:
            frame2, left, right = dl.process(frame)
            # print(left, right)
            cv.imshow("DISPLAY WINDOW", frame2)
        except:
            cv.imshow("DISPLAY WINDOW", frame1)
        finally:
            r_relpos = 0.5

        #################################################################### 3 ZAGLUSHKA
        is_detected = 0

        #################################################################### SITUATIONAL
        if (buffer < x_pos-2):
            r_i += 1
            x_pos = buffer
        elif (buffer > x_pos+2):
            l_i += 1
            x_pos = buffer
        
        if (l_i == 17):
            r_i = 0
            l_i = 0
            print('left')
            p_dir = 0
            
        if (r_i == 17):
            l_i = 0
            r_i = 0
            print('right')
            p_dir = 2

        sit = sr.define_situation(buffer, p_dir)

        print(sit)

        #################################################################### REGULATOR
        dir_com, speed_com = fs.fuz_reg(p_dist, p_dir, r_relpos, is_detected)
        # print(dir_com, type(speed_com.item()))
        # print(buffer)

        ################################################################### COMAND INTERPRETATOR
        command = "LUA_Base(" + str(round(speed_com.item(), 1)) + ",0,0)^^^"
        b = command.encode('utf-8')
        # send_data(b, conn)
        key = cv.waitKey(1)

        if key & 0xFF == ord('q'):    
            stop(conn)
            conn.shutdown(socket.SHUT_RDWR)
            conn.close()


if __name__ == "__main__":
    main()
        
        


