import os
import argparse
import numpy as np

import skimage.io
from skimage.color import rgb2gray

from skimage.transform import hough_line, hough_line_peaks
from skimage import morphology
from skimage import exposure


def get_args():
    parser = argparse.ArgumentParser(description='This tool accepts an image '
                                                 'and attempts to fine the '
                                                 'centerline of railroad '
                                                 'tracks. It should return '
                                                 'the same image, renamed, '
                                                 'with a red line for the '
                                                 'tracks centerline',

                                     epilog='If you are having problems with '
                                            'this script please contact Zach '
                                            'Raymer - rayme1zb@gmail.com')
    parser.add_argument('input_file')
    parser.add_argument('-output_file')
    parser.add_argument('-line_width', type=int, default=2)
    return parser.parse_args()


def process_image(input_file, output_file, line_width):
    data = skimage.io.imread(input_file)
    gray = rgb2gray(data)

    hist_eq = exposure.equalize_hist(gray)  # data float btwn 0-1
    so_removed = morphology.remove_small_objects(hist_eq > 0.8,
                                                 10000)  # data bool
    mask = 1 * morphology.remove_small_holes(so_removed,
                                             1000)  # data bool, multiply by 1 to turn to int

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    h, theta, d = hough_line(mask)

    sid = {'slope': [], 'intercept': []}
    origin = np.array((0, gray.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=2)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        s, i = np.polyfit((origin[0], origin[1]), (y0, y1), 1)
        sid['slope'].append(s)
        sid['intercept'].append(i)

    nys = list(range(0, gray.shape[0], 1))
    nxs = (nys - np.mean(sid['intercept'])) / np.mean(sid['slope'])
    nxs = np.floor(nxs).astype(np.int16)

    for x in range(-line_width, line_width+1):
        data[nys, nxs + x, :] = [255, 0, 0]

    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = base + "_wLine" + ext

    skimage.io.imsave(output_file, data)

    return


def main():
    args = get_args()
    process_image(**vars(args))


if __name__ == '__main__':
    main()
