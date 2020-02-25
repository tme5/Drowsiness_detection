#python drowniness_yawn.py --webcam webcam_index
from sqlite_db import SqliteDB
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from gtts import gTTS

class BaseDetection(object):
    def __init__(self):
        self.yawn_alarm_msg = 'take some fresh air sir'
        self.eye_alarm_msg = 'wake up sir'
        self.yawn_alarm_file = 'yawn_alarm.mp3'
        self.eye_alarm_file = 'eye_alarm.mp3'
        tts = gTTS(text=self.yawn_alarm_msg, lang='en')
        tts.save(self.yawn_alarm_file)
        tts = gTTS(text=self.eye_alarm_msg, lang='en')
        tts.save(self.eye_alarm_file)
    def alarm(self, msg):
        while self.eye_alarm:
            print('[ALARM] Sleeping Alert')
            os.system("omxplayer -o local %s" %self.eye_alarm_file)

        if self.yawn_alarm:
            print('[WARNING] Drowsiness Alert')
            self.saying = True
            os.system("omxplayer -o local %s" %self.yawn_alarm_file)
            self.saying = False

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def final_ear(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)

    def lip_distance(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = abs(top_mean[1] - low_mean[1])
        return distance
    
    def get_face_of_interest(self, rects, window_area):
        if len(rects)>=1:
            max_area = 0
            max_face = None
            _data = [((face.right()-face.left())*(face.bottom()-face.top()), face) for face in rects]
            (max_area, max_face) = max(_data,key=lambda item:item[1])
            if (max_area/window_area)>=0.06:
                return max_face
            else:
                print("[INFO] Face area: {}, Window area: {}".format(max_area, window_area))
                print("[INFO] You are not sufficiently close to camera")
        return None

class MyDectection(BaseDetection):
    def __init__(self):
        super(MyDectection, self).__init__()
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
        self.args = vars(self.ap.parse_args())
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 5
        self.YAWN_CONSEC_FRAMES = 7
        self.EYE_AR_INV_THRESH = 75
        self.ear_list = []
        self.YAWN_THRESH = 16
        self.eye_alarm = False
        self.yawn_alarm = False
        self.saying = False
        self.COUNTER = 0
        self.window_width = 700
        self.show_enhanced_image = False
        self.enhance_image = True
        self.save_videostream = False
        print("[INFO] Loading the predictor and detector...")
        self.detector = dlib.get_frontal_face_detector()
        #detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.sqlite_obj = SqliteDB()
        print("[INFO] Starting Video Stream")
        try:
            self.vs = VideoStream(usePiCamera=True).start()       #//For Raspberry Pi
        except:
            self.vs = VideoStream(src=self.args["webcam"]).start()
        time.sleep(0.5)

    def run_dectection(self):
        if self.save_videostream:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
            
        blink_start = blink_stop = 0
        fc = 0
        self.ear_inv_list = []
        self.bw_list = []
        self.bi_list = []
        while True:
            _frame = self.vs.read()
            self.frame = imutils.resize(_frame, width=self.window_width)

            if self.enhance_image:
                # Converting image to LAB Color model
                lab = cv2.cvtColor(self.frame, cv2.COLOR_BGR2LAB)
                # Splitting the LAB image to different channels
                l, a, b = cv2.split(lab)
                # Applying CLAHE to L-channel---
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                # Merge the CLAHE enhanced L-channel with the a and b channel
                limg = cv2.merge((cl, a, b))
                # Converting image from LAB Color model to RGB model
                final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            else:
                final = self.frame

            if self.show_enhanced_image:
                self.frame = final

            gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
            (w, h) = gray.shape
            rects = self.detector(gray, 0)
            
            rect = self.get_face_of_interest(rects, (w*h))
            
            if rect:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                eye = self.final_ear(shape)
                ear = eye[0]
                ear_inv = 100 - (ear*100)
                self.ear_list.append(ear_inv)
                leftEye = eye [1]
                rightEye = eye[2]

                yawn_index = self.lip_distance(shape)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(self.frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(self.frame, [rightEyeHull], -1, (0, 255, 0), 1)

                lip = shape[48:60]
                cv2.drawContours(self.frame, [lip], -1, (0, 255, 0), 1)

                # Insert indices in database table
                #  self.sqlite_obj.insert_data(int(time.time()), ear, yawn_index)
                self.sqlite_obj.insert_data(time.time(), ear, ear_inv, yawn_index)
                self.BGR = (0, 255, 0)


                if len(self.ear_list) >= 2:
                    if (ear_inv >= self.EYE_AR_INV_THRESH
                            and self.ear_list[-2] <= self.EYE_AR_INV_THRESH):
                        blink_start = time.time()
                        try:
                            bi = blink_start - blink_stop
                            self.bi_list.append(bi)
                        except:
                            pass
                    elif (ear_inv <= self.EYE_AR_INV_THRESH
                            and self.ear_list[-2] >= self.EYE_AR_INV_THRESH):
                        blink_stop = time.time()
                        bw = blink_stop - blink_start
                        self.sqlite_obj.insert_blink_width(time.time(), bw)
                        self.bw_list.append(bw)
                        blink_start = 0

                if blink_start and (time.time() - blink_start) > 0.5:
                    if self.eye_alarm == False:
                        self.eye_alarm = True
                        print('wake up sir')
                        t = Thread(target=self.alarm, args=('wake up sir',))
                        t.daemon = True
                        t.start()

                    self.BGR = (0, 0, 255)
                    cv2.putText(self.frame, "DROWSINESS ALERT!", (450, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                else:
                    self.eye_alarm = False

                if len(self.bw_list) > 1:
                    cv2.putText(self.frame, "Blink width: {:.2f}".format(self.bw_list[-1]), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.BGR, 2)
                    if self.bw_list[-1] > 0.19:
                        cv2.putText(self.frame, "Getting drowsy", (500, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if len(self.bi_list) > 1:
                    cv2.putText(self.frame, "Blink interval: {:.2f}".format(self.bi_list[-1]), (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.BGR, 2)
                    if self.bi_list[-1] <= 0.7:
                        cv2.putText(self.frame, "Getting drowsy", (500, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                if yawn_index > self.YAWN_THRESH:
                    self.COUNTER += 1
                    if self.COUNTER >= self.YAWN_CONSEC_FRAMES:
                        self.BGR = (0, 255, 255)
                        cv2.putText(self.frame, "Yawn Alert", (550, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.BGR, 2)
                        if self.yawn_alarm == False and self.saying == False:
                            self.yawn_alarm = True
                            print('take some fresh air sir')
                            t = Thread(target=self.alarm, args=('take some fresh air sir',))
                            t.start()
                else:
                    self.COUNTER = 0
                    self.yawn_alarm = False
                cv2.putText(self.frame, "YAWN: {:.2f}".format(yawn_index), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Save video stream if flag is True.
            if self.save_videostream:
                out.write(self.frame)
                
            cv2.imshow("Frame", self.frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                
        if self.save_videostream:
            out.release()
        print("[INFO] Closing VideoStream")
        self.vs.stop()
        cv2.destroyAllWindows()
        self.sqlite_obj.conn_close()

if __name__ == '__main__':
    test_obj = MyDectection()
    test_obj.run_dectection()
