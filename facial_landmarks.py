# Dlib
import dlib

import cv2

import timeit
import numpy as np

# From https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# From imutils - https://github.com/jrosebr1/imutils

def rect_to_bb(rect):
  # take a bounding predicted by dlib and convert it
  # to the format (x, y, w, h) as we would normally do
  # with OpenCV
  x = rect.left()
  y = rect.top()
  w = rect.right() - x
  h = rect.bottom() - y
 
  # return a tuple of (x, y, w, h)
  return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
  # initialize the list of (x, y)-coordinates
  N = shape.num_parts
  coords = np.zeros((N, 2), dtype=dtype)
 
  # loop over the N facial landmarks and convert them
  # to a 2-tuple of (x, y)-coordinates
  for i in range(0, N):
    coords[i] = (shape.part(i).x, shape.part(i).y)
  # coords = [(shape.part(i).x, shape.part(i).y) for i in range(N)]
  # return the list of (x, y)-coordinates
  return coords


# OpenCV
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

# Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

N = 5
list_fps_det = [30]*N
list_fps_cam = [30]*N

k = 0
cam_stop = timeit.default_timer()
while(True):
  
  k += 1

  # Capture frame-by-frame
  ret, frame = cap.read()
  cam_start = cam_stop
  cam_stop = timeit.default_timer()
  start = cam_stop
  
  frame = cv2.flip(frame,1)

  ## Dlib face detection

  # scale = 0.5
  # scale = 0.33
  scale = 0.25

  frame_small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
  frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

  if(k==1):
    print("(h,w,c) original = ",frame.shape)
    print("(h,w,c) small = ",frame_small.shape)

  rects = detector(frame_small)

  for rect in rects:

    rect = dlib.rectangle(round(rect.left()/scale),round(rect.top()/scale),round(rect.right()/scale),round(rect.bottom()/scale))

    # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #     i, d.left(), d.top(), d.right(), d.bottom()))
    # cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,255),1)

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(frame, rect)
    shape = shape_to_np(shape)
 
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
      cv2.circle(frame, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)

  # Our operations on the frame come here
  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  stop = timeit.default_timer()

  det_fps = 1/(stop-start)
  cam_fps = 1/(cam_stop-cam_start)

  list_fps_det[k % N] = det_fps
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

