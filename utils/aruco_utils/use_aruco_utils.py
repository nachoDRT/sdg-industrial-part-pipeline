from aruco import aruco_utils
import cv2
import os

NUM_MARKERS = 6
MARKERS_NAME = 'G3'
COEF_MATRIX = True
DIST_COEFF = True
DISPLAY = True

Create_markers = False
Read_markers = True

aruco_instance = aruco_utils()


# Generate as many markers as 'NUM_MARKERS'
if Create_markers:
    aruco_instance.generate_markers(NUM_MARKERS, MARKERS_NAME)

# Analize images from a folder
if Read_markers:
    folder_path = os.path.abspath(os.getcwd()) + '/examples/'
    folder_path = '/home/msi/Documentos/database_generator/G1_a_Point_2_flatten_dataset/box_0.8_to_0.9_projection/output_aruco/000000/images/'
    lst = os.listdir(folder_path)
    lst.sort()

    for files in lst:
        img_path = folder_path + files
        marker_img = cv2.imread(img_path)
        corners, ids, Point_name, image = aruco_instance.detect_markers(
            marker_img, COEF_MATRIX, DIST_COEFF, DISPLAY)

        # Display the resulting frame
        cv2.namedWindow('Detection', cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(
            'Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Detection', image)
        while cv2.getWindowProperty('Detection', cv2.WND_PROP_VISIBLE) > 0:
            if cv2.waitKey(100) > 0:
                break

        cv2.destroyAllWindows()
