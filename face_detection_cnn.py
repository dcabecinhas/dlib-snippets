# Dlib
import dlib

import cv2
cap = cv2.VideoCapture(0)

import timeit
import numpy as np
  
# From https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# Dlib
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

N = 5
list_fps_det = [30]*N
list_fps_cam = [30]*N

k = 0
while(True):
  
  k += 1

  # Capture frame-by-frame
  cam_start = timeit.default_timer()
  ret, frame = cap.read()

  cam_stop = timeit.default_timer()
  
  ## Dlib face detection

  start = timeit.default_timer()

  # resize image to half-size and flip as webcam
  # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
  # frame = cv2.resize(frame, (0,0), fx=0.333, fy=0.333)
  frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
  frame = cv2.flip(frame,1)

  # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  if(k==1):
    print("(h,w,c) = ",frame.shape)

  # dets = detector(frame)
  # print("Number of faces detected: {}".format(len(dets)))

  dets = cnn_face_detector(frame)
  # for i, d in enumerate(dets):
  #   print("Detection {}, score: {}, face_type:{}".format(
  #     d, scores[i], idx[i]))
             
  for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()))
    cv2.rectangle(frame,(d.rect.left(),d.rect.top()),(d.rect.right(),d.rect.bottom()),(0,255,255),1)

  # Our operations on the frame come here
  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  stop = timeit.default_timer()
  fps = 1/(stop-start)
  cam_fps = 1/(cam_stop-cam_start)

  list_fps_det[k % N] = fps
  list_fps_cam[k % N] = cam_fps

  # print("FPS:", 1/(stop-start))

  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(frame, "fps detection: {0:.1f}".format(sum(list_fps_det)/N), (0, 13), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
  cv2.putText(frame, "fps camera:   {0:.1f}".format(sum(list_fps_cam)/N), (0, 30), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

  # Display the resulting frame
  cv2.imshow('frame',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()

# Hack not to "hang" the window in *nix systems (Linux,Mac)
cv2.waitKey(1)

