import os
import cv2
from glob import glob
import argparse

__version__ = "1.0.0"

def cascade_gen(HAAR_FILE):
  cascade = cv2.CascadeClassifier(HAAR_FILE)
  return cascade

def img_list_gen(path):
  # generate file list
  images = glob(f"./{path}/*")
  return images

def face_cutter(images, cascade, directory):
  os.makedirs(directory, exist_ok=True)
  for i,image in enumerate(images):
    # read image
    img = cv2.imread(image)
    # detect faces
    face_pos = cascade.detectMultiScale(img)
    if len(face_pos) > 0:
      # cut face
      for j,(x,y,w,h) in enumerate(face_pos):
        face = img[y:y+h, x:x+w]
        # output image
        cv2.imwrite(f'{directory}/face{i}-{j}.jpg', face)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=f"face-cutter v{__version__}")
  parser.add_argument(
    "-t",
    "--target",
    dest="target_path",
    help="Target path",
    type=str,
    required=True,
  )
  parser.add_argument(
    "-c",
    "--cascade",
    dest="cascade_path",
    help="Cascade file path. Get from https://github.com/opencv/opencv/tree/master/data/haarcascades",
    type=str,
    required=True,
  )
  parser.add_argument(
    "-o",
    "--output",
    dest="output_path",
    help="Output directory path",
    type=str,
    required=True,
  )
  args = parser.parse_args()
  cascade = cascade_gen(args.cascade_path)
  img_list = img_list_gen(args.target_path)
  face_cutter(img_list, cascade, args.output_path)
