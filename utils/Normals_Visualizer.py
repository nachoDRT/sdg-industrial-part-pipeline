import numpy as np
import cv2
import os
import h5py
import sys
import curses
import math
from copy import deepcopy


# Creates and opens all necesary directories and files.
def files_management():
    input_path = 'output'
    input_aruco_path = 'output_aruco'
    # Visualize all given files
    print('Extracting from: ', input_path, input_aruco_path)
    path = os.path.abspath(os.getcwd())
    base_path = os.path.join(path, input_path)
    base_aruco_path = os.path.join(path, input_aruco_path)
    return path, base_path, base_aruco_path


# Loads from a given file .png, normal images, QR image, distance image.
def data_loading(base_path, folder):
    image = np.empty(0, dtype=np.uint8)
    normals = np.empty(0, dtype=np.uint8)
    # Search of all files within folder
    for data in os.listdir(os.path.join(base_path, folder)):
        if data.endswith('.hdf5'):
            with h5py.File(os.path.join(base_path, folder, data), "r") as data:
                # List all groups
                keys = list(data.keys())
                # Get the data
                for key in keys:
                    if key == 'colors':
                        image = np.array(data[key])
                    if key == 'normals':
                        normals = np.array(data[key])
    return image, normals


# Creater all necassary windows for images display.
def windows_creation():
    cv2.destroyAllWindows()
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', 640, 480)
    cv2.setMouseCallback('Original', click_and_distance)
    cv2.namedWindow('Aruco', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Aruco', 640, 480)
    cv2.setMouseCallback('Aruco', click_and_distance)
    cv2.namedWindow('Normals', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Normals', 640, 480)
    cv2.setMouseCallback('Normals', click_and_distance)
    return


def windows_display(original, aruco, normals):
    cv2.imshow('Original', original)
    cv2.imshow('Aruco', aruco)
    cv2.imshow('Normals', normals)
    return


def click_and_distance(event, x, y, flags, param):
    # grab references to the global variables
    global mouse
    mouse = (x, y)
    return


def display_results(normals_EXR, normals_Unitary, normals_Degrees):
    global console, mouse
    mouse_str = f'Mouse position: ({mouse})'
    exr = f'EXR format: {normals_EXR}'
    unitary = f'Unitary format: {normals_Unitary}'
    degrees = f'Unitary format: {normals_Degrees}'
    console.clear()
    console.addstr(0, 0, mouse_str)
    console.addstr(1, 0, exr)
    console.addstr(2, 0, unitary)
    console.addstr(3, 0, degrees)
    console.refresh()
    return


if __name__ == "__main__":
    # initialize with (0,0) point
    mouse = (0, 0)

    # Configuration
    path, base_path, base_aruco_path = files_management()
    windows_creation()
    console = curses.initscr()

    for folder in sorted(os.listdir(base_path)):
        # Obtain only the folders that are numbers
        if os.path.isdir(os.path.join(base_path, folder)) and folder.isdigit():
            # Obtain all files inside each folder
            image, normals_map = data_loading(base_path, folder)
            image_aruco, _ = data_loading(base_aruco_path, folder)
            normals_display = deepcopy(normals_map)

            while True:
                # Display mouse position
                windows_display(image, image_aruco, normals_display)

                # Obtain normals and transform to angles
                normals_point = normals_map[mouse[1], mouse[0], :]
                normals_EXR = np.around(normals_point, 2)
                # normals = [-normals[1], normals[0], normals[2]]
                normals_point = np.asanyarray(
                    [normals_point[1], 1 - normals_point[0], normals_point[2]])
                normals_Unitary = (normals_point - [0.5, 0.5, 0.5])*2
                modulo = math.sqrt(np.power(normals_Unitary, 2).sum())
                normals_Unitary = np.around(normals_Unitary, 2)
                normals_Degrees = (normals_point - [0.5, 0.5, 0.5])*180
                normals_Degrees = np.around(normals_Degrees, 2)

                # Print results
                display_results(normals_EXR, normals_Unitary, normals_Degrees)

                # Read keyboard
                key = cv2.waitKey(100) & 0xFF

                # Press Q or Esq on keyboard to exit
                if key == ord('q') or key == 27:
                    curses.endwin()
                    sys.exit(0)
                elif key == ord('n'):
                    break
