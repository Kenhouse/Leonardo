import os
import argparse
from skimage import io

def main(images_dir,output_dir):
    image_files = os.listdir(images_dir)
    for f in image_files:
      filename, file_extension = os.path.splitext(f)
      if f == ".DS_Store":
          continue

      if file_extension == '.jpg':
          continue

      img = io.imread(args.images_dir + '/' + f)
      if img.shape[2] == 4:
          img = img[:, :, :3]

      io.imsave(args.output_dir + '/' + filename + '.jpg',img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', default=None, help='path to the src dir')
    parser.add_argument('output_dir', default=None, help='path to the src dir')
    args = parser.parse_args()
    main(args.images_dir,args.output_dir)
