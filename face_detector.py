import cv2, dlib, sys
import numpy as np


find_face = dlib.get_frontal_face_detector()
find_face_elements = dlib.shape_predictor('input/face_data.dat')

input_vid = cv2.VideoCapture('input/set1_input.mp4')

filter = cv2.imread('input/set1_input_filter.png', cv2.IMREAD_UNCHANGED)
scaler = 0.32

def face_filter(final_output, mask_png, center_x, center_y, size=None):
  try:
    orig_vid = final_output.copy()


    orig_vid = cv2.cvtColor(orig_vid, cv2.COLOR_BGR2BGRA)

    mask_png = cv2.resize(mask_png.copy(), size)

    b, g, r, a = cv2.split(mask_png)

    mask = cv2.medianBlur(a, 3)

    h, w, n = mask_png.shape
    mask_size = orig_vid[int(center_y - h / 2):int(center_y + h / 2), int(center_x - w / 2):int(center_x + w / 2)]

    img1_bg = cv2.bitwise_and(mask_size.copy(), mask_size.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(mask_png, mask_png, mask=mask)

    orig_vid[int(center_y - h / 2):int(center_y + h / 2), int(center_x - w / 2):int(center_x + w / 2)] = cv2.add(img1_bg, img2_fg)


    orig_vid = cv2.cvtColor(orig_vid, cv2.COLOR_BGRA2BGR)
    return orig_vid
  except Exception:
    return final_output

elements = []
size_images = []

while True:

  frame, img = input_vid.read()
  if not frame:
    break

  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
  init = img.copy()

  if len(elements) == 0:
    dectect = find_face(img, 1)
  else:
    image_first = img[elements[0]:elements[1], elements[2]:elements[3]]

    dectect = find_face(image_first)


  img_array = []
  for image in dectect:
    if len(elements) == 0:
      face = find_face_elements(img, image)
      np_convert = np.array([[i.x, i.y] for i in face.parts()])
    else:
      face = find_face_elements(image_first, image)
      np_convert = np.array([[i.x + elements[2], i.y + elements[0]] for i in face.parts()])

    for j in np_convert:
      cv2.circle(img, center=tuple(j), radius=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


    center_x, center_y = np.mean(np_convert, axis=0).astype(np.int)


    min_coords = np.min(np_convert, axis=0)
    max_coords = np.max(np_convert, axis=0)

    cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)


    size_img = int(max(max_coords - min_coords))
    size_images.append(size_img)
    final_size = int(np.mean(size_images))

    elements = np.array([int(min_coords[1] - size_img / 2), int(max_coords[1] + size_img / 2), int(min_coords[0] - size_img / 2), int(max_coords[0] + size_img / 2)])
    elements = np.clip(elements, 0, 10000)

    # Filter on face
# set1_input use below code
    result = face_filter(init, filter, center_x + 5, center_y - 90, size=(final_size, final_size))

# set2_input use below code
    #result = face_filter(init, filter, center_x, center_y - 20, size=(final_size *2, final_size *2))

# set3_input use below code
    #result = face_filter(init, filter, center_x, center_y - 50, size=(final_size, final_size))
  # visualize
  cv2.imshow('input', init)
  cv2.imshow('face recog', img)
  cv2.imshow('output', result)
  cv2.imwrite('output_original.png', init)
  cv2.imwrite('output_face_recog.png', img)
  cv2.imwrite('output_final_output.png', result)

  if cv2.waitKey(1) == ord('q'):
    sys.exit(1)

























