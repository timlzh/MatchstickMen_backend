from typing import NoReturn
import argparse
import yaml
import queue
import threading
import shutil
from cv2 import VideoCapture as capture, waitKey
import cv2
import os

# dprint only pirnt message when environment variable DEBUG=1 is setted
from dprint import dprint

# global variables
fifo_name = "/home/khadas/workspace/FIFO/OPENCV_TO_HRNET"
trigger = "/home/khadas/workspace/FIFO/TRIGGER"
willPut = False
q = queue.Queue(30)
fps = 0
width = 1920
height = 1080

def put_receiver():
    global willPut
    try:
        os.mkfifo(trigger)
    except FileExistsError:
        pass
    with open(trigger,"rb") as f:
        while True:
            f.readline()
            willPut = not willPut


def send2processor():
    global fps
    try:
        os.mkfifo(fifo_name)
    except FileExistsError:
        pass
    frame_cnt = 0
    with open(fifo_name, 'wb') as f:
        dprint("open fifo successfully.")
        f.write('{}\n'.format(fps).encode())
        f.write('{}\n'.format(height).encode())
        f.write('{}\n'.format(width).encode())
        f.flush()
        while True:
            if q.empty != True:
                frame = q.get()
                h, w = int(frame.shape[0]), int(frame.shape[1])
                dprint("{}x{}".format(h, w))
                f.write('{}\n'.format(h).encode())
                f.write('{}\n'.format(w).encode())
                f.write(frame.data)
                dprint("send a frame ok ")
            frame_cnt += 1


def frame_derives(input_file):
    print("intput file: ",input_file)
    cap = capture(input_file)
    global width, height, fps,willPut
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        else:
            if ret == True:
                if willPut==True:
                    q.put(frame)
    cap.release()


# operation costs too much is not allowed during consecutive 'vedio_capture.read()'
# so ,must to create another process to do further processing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description='please enter two parameters a and b ...'
    parser.add_argument("-i", "--inputA", help="set input file", dest="input", type=str, default="video1.mp4")
    args = parser.parse_args()
    t1 = threading.Thread(target=frame_derives, name="frame_derives",args=(args.input,))
    t2 = threading.Thread(target=send2processor, name="send2processor")
    t3 = threading.Thread(target=put_receiver,name="put_receiver")
    t1.start()
    t2.start()
