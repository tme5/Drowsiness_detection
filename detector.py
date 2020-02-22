#python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

class BaseDetection():
    def __init__(self):
        pass
    
    def alarm(self, msg):
        global eye_alarm
        global yawn_alarm
        global saying

        while eye_alarm:
            print('call')
            s = 'espeak "'+msg+'"'
            os.system(s)

        if yawn_alarm:
            print('call')
            saying = True
            s = 'espeak "' + msg + '"'
            os.system(s)
            saying = False

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

class MyDectection(BaseDetection):
    def __init__(self):
        super(MyDectection, self).__init__()
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
        self.args = vars(self.ap.parse_args())
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 10
        self.YAWN_THRESH = 20
        self.eye_alarm = False
        self.yawn_alarm = False
        self.saying = False
        self.COUNTER = 0
        print("[INFO] Loading the predictor and detector...")
        self.detector = dlib.get_frontal_face_detector()
        #detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        print("[INFO] Starting Video Stream")
        #vs = VideoStream(src=args["webcam"]).start()
        self.vs = VideoStream(usePiCamera=True).start()       #//For Raspberry Pi
        time.sleep(0.5)

    def run_dectection(self):
        while True:
            _frame = self.vs.read()
            self.frame = imutils.resize(_frame, width=450)
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            #rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            #    	minNeighbors=5, minSize=(30, 30),
            #    	flags=cv2.CASCADE_SCALE_IMAGE)

            for rect in rects:
            #for (x, y, w, h) in rects:
            #    rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
               
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
            
                eye = self.final_ear(shape)
                ear = eye[0]
                leftEye = eye [1]
                rightEye = eye[2]
            
                distance = self.lip_distance(shape)
            
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(self.frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(self.frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
                lip = shape[48:60]
                cv2.drawContours(self.frame, [lip], -1, (0, 255, 0), 1)
            
                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
            
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        if self.eye_alarm == False:
                            self.eye_alarm = True
                            print('wake up sir')
                            #t = Thread(target=self.alarm, args=('wake up sir',))
                            #t.deamon = True
                            #t.start()
            
                        cv2.putText(self.frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
                else:
                    self.COUNTER = 0
                    self.eye_alarm = False
            
                if (distance > self.YAWN_THRESH):
                        cv2.putText(self.frame, "Yawn Alert", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if self.yawn_alarm == False and self.saying == False:
                            self.yawn_alarm = True
                            print('take some fresh air sir')
                            #t = Thread(target=self.alarm, args=('take some fresh air sir',))
                            #t.deamon = True
                            #t.start()
                else:
                    self.yawn_alarm = False
            
                cv2.putText(self.frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(self.frame, "YAWN: {:.2f}".format(distance), (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.vs.stop()

if __name__ == '__main__':
    test_obj = MyDectection()
    test_obj.run_dectection()
