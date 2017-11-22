# Dlib
import dlib

import cv2

import timeit
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys


#window dimensions
width = 1280
height = 720
nRange = 1.0

global capture
capture = None

# Dlib
global detector
global predictor
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


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



  # Capture frame-by-frame




def idle():
  #capture next frame
  global capture
  _,frame = capture.read()
  
  # Timing
  t = timeit.default_timer()
  fps_camera = 1/(t-idle.t_prev)
  idle.t_prev = t

  frame = cv2.flip(frame,1)

  ## Dlib face detection

  # scale = 0.5
  # scale = 0.3333
  scale = 0.25

  frame_small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
  frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

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
  
  t_stop = timeit.default_timer()
  fps_detection = 1/(t_stop-t)

  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(frame, "fps camera:   {0:.1f}".format(fps_camera), (0, 13), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
  cv2.putText(frame, "fps detection: {0:.1f}".format(fps_detection), (0, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

  
  # Create Texture
  glTexImage2D(GL_TEXTURE_2D, 
    0, 
    GL_RGB, 
    1280,720,
    0,
    GL_BGR, 
    GL_UNSIGNED_BYTE, 
    frame)
  glutPostRedisplay()

idle.t_prev = timeit.default_timer()


def init():
  #glclearcolor (r, g, b, alpha)
  glClearColor(0.0, 0.0, 0.0, 1.0)

  glutDisplayFunc(display)
  glutReshapeFunc(reshape)
  glutKeyboardFunc(keyboard)
  glutIdleFunc(idle)


def display():
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glEnable(GL_TEXTURE_2D)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

  # Set Projection Matrix
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  gluOrtho2D(0, width, height, 0)

  # Switch to Model View Matrix
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

  # Draw textured Quads
  glBegin(GL_QUADS)
  glTexCoord2f(0.0, 0.0)
  glVertex2f(0.0, 0.0)
  glTexCoord2f(1.0, 0.0)
  glVertex2f(width, 0.0)
  glTexCoord2f(1.0, 1.0)
  glVertex2f(width, height)
  glTexCoord2f(0.0, 1.0)
  glVertex2f(0.0, height)
  glEnd()

  glFlush()
  glutSwapBuffers()


def reshape(w, h):
  # if h == 0:
  #   h = 1

  glViewport(0, 0, w, h)
  # glMatrixMode(GL_PROJECTION)

  # glLoadIdentity()
  # # allows for reshaping the window without distoring shape

  # if w <= h:
  #   glOrtho(-nRange, nRange, -nRange*h/w, nRange*h/w, -nRange, nRange)
  # else:
  #   glOrtho(-nRange*w/h, nRange*w/h, -nRange, nRange, -nRange, nRange)

  # glMatrixMode(GL_MODELVIEW)
  # glLoadIdentity()


def keyboard(key, x, y):
  if key == b'\x1b' or key == b'q':
    sys.exit()


def main():
  global capture
  #start openCV capturefromCAM
  capture = cv2.VideoCapture(0)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
  glutInit(sys.argv)
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
  glutInitWindowSize(width, height)
  glutInitWindowPosition(100, 100)
  glutCreateWindow("OpenGL + OpenCV")

  init()
  glutMainLoop()

main()

