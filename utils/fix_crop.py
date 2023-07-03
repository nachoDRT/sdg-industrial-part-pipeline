import cv2
from copy import deepcopy
import numpy as np


class fix_crop():

    def __init__(self) -> None:
        pass

    def update_crop_dims(self, crop_dims):
        """
        Input:
            - img: a picture
            - crop_dims: a vector with the dimensions
                        of the crop [x, y, width, height]

        Output:
            - fixed_crop = a vector with the dimensions of
                          the FIXED crop [x, y, width, height]
        """

        crop_width = crop_dims[2]
        crop_height = crop_dims[3]

        upper_left_corner_x = crop_dims[0]
        upper_left_corner_y = crop_dims[1]

        # No correction case
        if crop_width == crop_height:
            fixed_crop_dims = crop_dims

        # Case A
        elif crop_height > crop_width:
            # Center x
            c_x = upper_left_corner_x + int((crop_width)/2)
            left_boundary = c_x - int(crop_height/2)
            fixed_crop_dims = [left_boundary,
                               upper_left_corner_y, crop_height, crop_height]

        # Case B
        elif crop_width > crop_height:
            # Center y
            c_y = upper_left_corner_y + int((crop_height)/2)
            upper_boundary = c_y - int(crop_width/2)
            fixed_crop_dims = [upper_left_corner_x,
                               upper_boundary, crop_width, crop_width]

        return fixed_crop_dims

    def draw_new_crop(self, img, fixed_crop_dims):
        """
        Input:
            - img: Image with initial crop
        Output:
            - img: Image with the new fixed crop
        """
        x_0 = fixed_crop_dims[0]
        y_0 = fixed_crop_dims[1]
        x_f = x_0 + fixed_crop_dims[2]
        y_f = y_0 + fixed_crop_dims[3]

        img = cv2.rectangle(img, (x_0, y_0), (x_f, y_f), (255, 0, 0), 1)

        return img

    def extract_crop(self, img, crop_dims):
        """
        Obtain the end result crop, with padding in case it is needed
        Input:
            - img: original raw image
            - crop_dims: the suggested dimensions of
                         the crop (still padding needs to be checked)
        Output:
            - final_crop: the end result image crop
        """
        img_copy = deepcopy(img)
        img_height = img_copy.shape[0] - 1
        img_width = img_copy.shape[1] - 1
        total_padding = [0, 0, 0, 0]  # [Left, TOP, Right, Bottom]

        # Variables for padding
        padding_left = crop_dims[0]
        padding_up = crop_dims[1]
        padding_right = (crop_dims[0] + crop_dims[2]) - img_width
        padding_down = (crop_dims[1] + crop_dims[3]) - img_height

        # Check the need of padding in left boundary
        if padding_left < 0:
            roi_x0 = 0
            roi_xf = padding_left + crop_dims[2]
            roi_y0 = crop_dims[1]
            roi_yf = roi_y0 + crop_dims[3]

            total_padding[0] = abs(padding_left)
            raw_crop = img_copy[roi_y0:roi_yf, roi_x0:roi_xf]
            final_crop = self.padding_left(raw_crop, abs(padding_left))

        # Check the need of padding in the upper boundary
        elif padding_up < 0:
            roi_x0 = crop_dims[0]
            roi_xf = roi_x0 + crop_dims[2]
            roi_y0 = 0
            roi_yf = padding_up + crop_dims[3]

            total_padding[1] = abs(padding_up)
            raw_crop = img_copy[roi_y0:roi_yf, roi_x0:roi_xf]
            final_crop = self.padding_up(raw_crop, abs(padding_up))

        # Check the need of padding in the right boundary
        elif padding_right > 0:
            roi_x0 = crop_dims[0]
            roi_xf = img_width
            roi_y0 = crop_dims[1]
            roi_yf = roi_y0 + crop_dims[3]

            total_padding[2] = abs(padding_right)
            raw_crop = img_copy[roi_y0:roi_yf, roi_x0:roi_xf]
            final_crop = self.padding_right(raw_crop, padding_right)

        # Check the need of padding in the bottom boundary
        elif padding_down > 0:
            roi_x0 = crop_dims[0]
            roi_xf = roi_x0 + crop_dims[2]
            roi_y0 = crop_dims[1]
            roi_yf = img_height

            total_padding[3] = abs(padding_down)
            raw_crop = img_copy[roi_y0:roi_yf, roi_x0:roi_xf]
            final_crop = self.padding_down(raw_crop, padding_down)

        else:
            # Region of interest coordinates intervals
            # (initial point: 0, final point: f)
            roi_x0 = crop_dims[0]
            roi_xf = roi_x0 + crop_dims[2]
            roi_y0 = crop_dims[1]
            roi_yf = roi_y0 + crop_dims[3]

            final_crop = img_copy[roi_y0:roi_yf, roi_x0:roi_xf]

        return final_crop, total_padding

    def padding_left(self, raw_crop, padding_n):
        """
        Add extra black pad in the left boundary
        Input:
            - raw_crop: maximum crop possible (given the original
                        image restrictions)
            - padding_n: the required padding thickness
        Output:
            - final_crop: squared final crop (raw crop + the black padding)
        """
        pad_height = raw_crop.shape[0]
        color = raw_crop[:, 0]
        if len(raw_crop.shape) == 3:
            dimensions = raw_crop.shape[2]
        else:
            dimensions = 1
        extra_pad = np.empty(
            shape=(pad_height, padding_n, dimensions), dtype=raw_crop.dtype)
        for i in range(extra_pad.shape[1]):
            extra_pad[:, i] = color[:][0]
        final_crop = cv2.hconcat([extra_pad, raw_crop])
        return final_crop

    def padding_up(self, raw_crop, padding_n):
        """
        Add extra black pad in the upper boundary
        Input:
            - raw_crop: maximum crop possible (given the original
                        image restrictions)
            - padding_n: the required padding thickness
        Output:
            - final_crop: squared final crop (raw crop + the black padding)
        """
        pad_width = raw_crop.shape[1]
        color = raw_crop[0, :]
        if len(raw_crop.shape) == 3:
            dimensions = raw_crop.shape[2]
        else:
            dimensions = 1
        extra_pad = np.empty(
            shape=(padding_n, pad_width, dimensions), dtype=raw_crop.dtype)
        for i in range(extra_pad.shape[0]):
            extra_pad[i, :] = color[:][0]
        final_crop = cv2.vconcat([extra_pad, raw_crop])
        return final_crop

    def padding_right(self, raw_crop, padding_n):
        """
        Add extra black pad in the right boundary
        Input:
            - raw_crop: maximum crop possible (given the original
                        image restrictions)
            - padding_n: the required padding thickness
        Output:
            - final_crop: squared final crop (raw crop + the black padding)
        """
        pad_height = raw_crop.shape[0]
        color = raw_crop[:, -1]
        if len(raw_crop.shape) == 3:
            dimensions = raw_crop.shape[2]
        else:
            dimensions = 1
        extra_pad = np.empty(
            shape=(pad_height, padding_n, dimensions), dtype=raw_crop.dtype)
        for i in range(extra_pad.shape[1]):
            extra_pad[:, i] = color[:][0]
        final_crop = cv2.hconcat([raw_crop, extra_pad])
        return final_crop

    def padding_down(self, raw_crop, padding_n):
        """
        Add extra black pad in the bottom boundary
        Input:
            - raw_crop: maximum crop possible (given the original
                        image restrictions)
            - padding_n: the required padding thickness
        Output:
            - final_crop: squared final crop (raw crop + the black padding)
        """
        pad_width = raw_crop.shape[1]
        color = raw_crop[-1, :]
        if len(raw_crop.shape) == 3:
            dimensions = raw_crop.shape[2]
        else:
            dimensions = 1
        extra_pad = np.empty(
            shape=(padding_n, pad_width, dimensions), dtype=raw_crop.dtype)
        for i in range(extra_pad.shape[0]):
            extra_pad[i, :] = color[:][0]
        final_crop = cv2.vconcat([raw_crop, extra_pad])
        return final_crop
