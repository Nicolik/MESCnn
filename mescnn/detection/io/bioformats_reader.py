import logging
import os
from collections import defaultdict

import javabridge
import bioformats
import bioformats.formatreader
import numpy as np
import cv2

from mescnn.detection.io.config import OP_EPS, check_desired_op
from mescnn.detection.qupath.utils import find_nearest, crop


class BioformatsReader(object):
    def __init__(self, image_path, objective_power=40):
        self.reader = bioformats.formatreader.get_image_reader(0, image_path)
        self.series = self.reader.rdr.getSeriesCount()
        self.name = os.path.basename(image_path)
        self.indexes = []
        self.dimensions = []
        self.objective_power = objective_power
        self.objective_powers = defaultdict(list)
        self._rescale_factor = None
        self._init_indexes()

    def _init_indexes(self):
        X, Y = 0, 0
        largest_X, largest_Y = 0, 0
        for s in range(self.series):
            self.reader.rdr.setSeries(s)
            has_increased = (self.reader.rdr.getSizeX() > X) and (self.reader.rdr.getSizeY() > Y)
            if has_increased:
                largest_X = self.reader.rdr.getSizeX()
                largest_Y = self.reader.rdr.getSizeY()
                self.indexes.append(s)
                self.dimensions.append((largest_X, largest_Y))
            self.objective_powers[self.indexes[-1]].append(self.objective_power * self.reader.rdr.getSizeX() / largest_X)
            X, Y, Z = self.reader.rdr.getSizeX(), self.reader.rdr.getSizeY(), self.reader.rdr.getSizeZ()
            print(f"Series: {s:2d}", end=' | ')
            print(f"X, Y, Z = {X:5d}, {Y:5d}, {Z}", end=' | ')
            print(f"C, T = {self.reader.rdr.getSizeC()}, {self.reader.rdr.getSizeT()}")
        print(f"Indexes: {self.indexes}")

    def read_resolution(self, s, x, y, w, h, desired_op, read_bgr=False, do_rescale=True):
        objective_powers = self.objective_powers[s]
        closest_level, closest_op = find_nearest(objective_powers, desired_op)
        print(f"[read_resolution] Desired OP: {desired_op}, Closest Level: {closest_level}, Closest OP: {closest_op}")
        assert check_desired_op(closest_op, desired_op), "Mismatch between closest OP and desired OP"
        self.reader.rdr.setSeries(s+closest_level)
        X, Y, Z = self.reader.rdr.getSizeX(), self.reader.rdr.getSizeY(), self.reader.rdr.getSizeZ()
        print(f"Series: {s:2d}", end=' | ')
        print(f"X, Y, Z = {X:5d}, {Y:5d}, {Z}")

        rescale_factor = self.objective_power / closest_op
        self._rescale_factor = rescale_factor
        if do_rescale:
            print(f"[read_resolution] Before Rescaling: [X={X}, Y={Y}] [{x}, {y}, {w}, {h}]")
            x, y, w, h = int(x/rescale_factor), int(y/rescale_factor), int(w/rescale_factor), int(h/rescale_factor)
            X, Y = int(X/rescale_factor), int(Y/rescale_factor)
            print(f"[read_resolution] After Rescaling:  [X={X}, Y={Y}] [{x}, {y}, {w}, {h}]")
        try:
            print(f"[read_resolution] Trying to open [{x}, {y}, {w}, {h}]. [X={X}, Y={Y}]. [x+w={x+w}, y+h={y+h}]")
            if desired_op == 40:
                print(f"[read_resolution] Before Crop: {x}-{x+w} ({X}), {y}-{y+h} ({Y})")
                x, y, w, h = crop(x, y, w, h, X, Y)
                print(f"[read_resolution] After Crop:  {x}-{x+w} ({X}), {y}-{y+h} ({Y})")
            image = self.reader.rdr.openBytesXYWH(0, x, y, w, h)
            image = image.reshape(3, h, w)
            image = np.transpose(image, (1, 2, 0))
            if read_bgr:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except javabridge.jutil.JavaException as E:
            logging.warning(f"javabridge.jutil.JavaException: {E}")
            image = None
        return image
