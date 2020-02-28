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
import sys

class BaseDetection(object):
    def __init__(self):
        self.yawn_alarm_msg = 'take some fresh air sir'
        self.eye_alarm_msg = 'wake up sir'
        self.yawn_alarm_file = 'yawn_alarm.mp3'
        self.eye_alarm_file = 'eye_alarm.mp3'
        #tts = gTTS(text=self.yawn_alarm_msg, lang='en')
        #tts.save(self.yawn_alarm_file)
        #tts = gTTS(text=self.eye_alarm_msg, lang='en')
        #tts.save(self.eye_alarm_file)
        
    def alarm(self, msg):
        while self.eye_alarm:
            print('[ALARM] Sleeping Alert')
            try:
                assert os.system("omxplayer -o local %s > /dev/null" % self.eye_alarm_file
                                 ) == 0
            except Exception:
                from playsound import playsound
                playsound(self.eye_alarm_file)

        if self.yawn_alarm:
            print('[WARNING] Drowsiness Alert')
            self.saying = True
            try:
                assert os.system("omxplayer -o local %s > /dev/null" % self.yawn_alarm_file
                                 ) == 0
            except Exception:
                from playsound import playsound
                playsound(self.yawn_alarm_file)
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

    def mar(self, shape):
        (start, end) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
        mouth = shape[start:end]
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])
        D = dist.euclidean(mouth[0], mouth[4])
        mar = (A + B + C) / (3.0 * D)
        return (mar, mouth)
    
    def get_face_of_interest(self, rects):
        if len(rects)>1:
            max_area = 0
            max_face = None
            #face = x,y,w,h
            _data = [(face[2]*face[3], face) for face in rects]
            #_data = [((face.right()-face.left())*(face.bottom()-face.top()), face) for face in rects]
            #(max_area, max_face) = max(_data,key=lambda item:item[1])
            max_area = 0
            max_face = None
            for f_area , face in _data:
                if max_area < f_area:
                    max_area = f_area
                    max_face = face
            #return max_face
            return dlib.rectangle(int(max_face[0]), int(max_face[1]), int(max_face[0] + max_face[2]),int(max_face[1] + max_face[3]))
        elif len(rects)==1:
            max_face = rects[0]
            return dlib.rectangle(int(max_face[0]), int(max_face[1]), int(max_face[0] + max_face[2]),int(max_face[1] + max_face[3]))
        else:
            return None

class MyDetection(BaseDetection):
    def __init__(self):
        super(MyDetection, self).__init__()
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
        self.args = vars(self.ap.parse_args())
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 5
        self.YAWN_CONSEC_FRAMES = 6
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
        self.register_flag = False
        print("[INFO] Loading the predictor and detector...")
        #self.detector = dlib.get_frontal_face_detector()
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.sqlite_obj = SqliteDB()
        print("[INFO] Starting Video Stream")
        try:
            self.vs = VideoStream(usePiCamera=True, resolution=(320, 240), framerate=32).start()       #//For Raspberry Pi
        except:
            self.vs = VideoStream(src=self.args["webcam"]).start()
        time.sleep(0.5)
    
    def get_ear_inv(self, msgs, gray, rect):
        ear_inv = 0
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        eye = self.final_ear(shape)
        ear = eye[0]
        ear_inv = 100 - (ear*100)
        self.ear_list.append(ear_inv)
        leftEye = eye [1]
        rightEye = eye[2]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(self.frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(self.frame, [rightEyeHull], -1, (0, 255, 0), 1)

        cv2.putText(self.frame, "### Calibrating EAR threshold ###", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        for idx, msg in enumerate(msgs, 1):
            cv2.putText(self.frame, msg, (10, 450+(30*idx)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
        return ear_inv
    
    def get_mar_inv(self, msgs, gray, rect):
        mar_inv = 0
        if rect is not None:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mar, mouth = self.mar(shape)
            mar_inv = 100 - (mar*100)
            cv2.drawContours(self.frame, [mouth], -1, (0, 255, 0), 1)

            cv2.putText(self.frame, "### Calibrating MAR threshold ###", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            for idx, msg in enumerate(msgs, 1):
                cv2.putText(self.frame, msg, (10, 450+(30*idx)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return mar_inv
        
    def run_detection(self):
        if self.save_videostream:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
            
        blink_start = blink_stop = 0
        fc = 0
        self.ear_inv_list = []
        self.bw_list = []
        self.bi_list = []
        space_count = 0
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
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, 
                	minNeighbors=5, minSize=(30, 30),
                	flags=cv2.CASCADE_SCALE_IMAGE)

            #rects = self.detector(gray, 0)
            rect = self.get_face_of_interest(rects)
            
            if rect is not None:
                if not self.register_flag:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        space_count+=1
                    if space_count==0:
                        self.open_ear_inv = self.get_ear_inv(["Keep your eyes wide open, hold for a moment","& press spacebar"], gray, rect)
                    elif space_count==1:
                        self.close_ear_inv = self.get_ear_inv(["Keep your eyes close, hold for a moment","& press spacebar"], gray, rect)
                    elif space_count==2:                    
                        self.open_mar_inv = self.get_mar_inv(["Keep your mouth wide open, hold for a moment","& press spacebar"], gray, rect)
                    elif space_count==3:
                        self.close_mar_inv = self.get_mar_inv(["Keep your mouth close, hold for a moment","& press spacebar"], gray, rect)
                    elif space_count==4:
                        self.EYE_AR_INV_THRESH = (self.open_ear_inv+self.close_ear_inv)/2
                        self.YAWN_THRESH = (self.open_mar_inv+self.close_mar_inv)/2
                        self.register_flag = True
                else:
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    eye = self.final_ear(shape)
                    ear = eye[0]
                    ear_inv = 100 - (ear*100)
                    self.ear_list.append(ear_inv)
                    leftEye = eye [1]
                    rightEye = eye[2]

                    yawn_index, mouth = self.mar(shape)
                    mar = 100 - (yawn_index*100)

                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(self.frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(self.frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    lip = shape[48:60]
                    cv2.drawContours(self.frame, [mouth], -1, (0, 255, 0), 1)

                    # Insert indices in database table
                    #  self.sqlite_obj.insert_data(int(time.time()), ear, yawn_index)
                    #self.sqlite_obj.insert_data(time.time(), ear, ear_inv, mar)
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

                    cv2.putText(self.frame,
                            "EAR threshold: %.2f" % self.EYE_AR_INV_THRESH,
                            (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if mar < self.YAWN_THRESH:
                        self.COUNTER += 1
                        if self.COUNTER >= self.YAWN_CONSEC_FRAMES:
                            self.BGR = (0, 255, 255)
                            cv2.putText(self.frame, "Yawn Alert", (550, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.BGR, 2)
                            if self.yawn_alarm == False and self.saying == False:
                                self.yawn_alarm = True
                                print('take some fresh air sir')
                                t = Thread(target=self.alarm, args=('take some fresh air sir',))
                                t.daemon = True
                                t.start()
                    else:
                        self.COUNTER = 0
                        self.yawn_alarm = False
                    cv2.putText(self.frame, "YAWN: {:.2f}".format(mar), (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(self.frame, "YAWN threshold: {:.2f}".format(self.YAWN_THRESH), (10, 480),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                            
            # Save video stream if flag is True.
            if self.save_videostream:
                out.write(self.frame)
                
            cv2.imshow("Frame", self.frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.register_flag = False
                self.eye_alarm = False
                space_count = 0
                
        if self.save_videostream:
            out.release()
        print("[INFO] Closing VideoStream")
        self.vs.stop()
        cv2.destroyAllWindows()
        self.sqlite_obj.conn_close()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit()

if __name__ == '__main__':
    test_obj = MyDetection()
    test_obj.run_detection()
