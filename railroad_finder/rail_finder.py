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
    # Read the data into a numpy array
    data = skimage.io.imread(input_file)

    # Convert the 3 band image to grayscale
    gray = rgb2gray(data)

    # Use equalize histogram as a method of contrast streching
    hist_eq = exposure.equalize_hist(gray)  # data float btwn 0-1

    # Threshold the data to isolate pixels above 0.8 and use a morphology
    # filter to remove groups of pixels w/ less than 10k members.
    # Note: if the image was small, this could cause an error - better to use
    # a percentage of n_pixels here
    so_removed = morphology.remove_small_objects(hist_eq > 0.8, 10000)

    # Use another morphology filter to fill in holes. Data here is bool,
    # multiply by 1 to turn to int
    mask = 1 * morphology.remove_small_holes(so_removed, 1000)  #

    # Classic straight-line Hough transform
    h, theta, d = hough_line(mask)

    # Use a dictionary to hold the slope and intercept values of each foudn line
    sid = {'slope': [], 'intercept': []}
    # Set the origin coords
    origin = np.array((0, gray.shape[1]))
    # Bin the number of extracted peaks to 2, as we know railways only
    # have 2 tracks.
    # Note: if multiple railways were present, a maximum of two lines would be
    # extracted
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=2)):
        # Use extracted angle + distance from origin to find y-points
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        # Calculate slope/intercept
        s, i = np.polyfit((origin[0], origin[1]), (y0, y1), 1)
        # record slope/intercept in dictionary
        sid['slope'].append(s)
        sid['intercept'].append(i)

    # Create range of y values that correspond to number of rows in the image
    nys = list(range(0, gray.shape[0], 1))
    # Reverse y=mx+b equation to find x values... x=(y-b)/m
    nxs = (nys - np.mean(sid['intercept'])) / np.mean(sid['slope'])
    # Pixels are integers not float, so take the floor and convert dtype
    nxs = np.floor(nxs).astype(np.int16)

    # using line_width parameter, burn the line into the image using a red color
    for x in range(-line_width, line_width+1):
        data[nys, nxs + x, :] = [255, 0, 0]

    # Create output file if none was provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = base + "_wLine" + ext

    # Save the image 
    skimage.io.imsave(output_file, data)

    return


def main():
    args = get_args()
    process_image(**vars(args))


if __name__ == '__main__':
    main()
