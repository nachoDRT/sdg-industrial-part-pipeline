"""
Main Author: Ignacio Ortiz de Zúñiga Mingot
Date: February 2022
Course: 2º MII + MIC

Modified by: Ignacio de Rodrigo Tobías
Date: August 2022

Purpose: Dataset pipeline to work with blenderproc workflow. Extract all
information from blenderproc output and transform it for CNN training.

Functions:
  - files_management():   Creates and opens all necessary directories and
                          files.

  - windows_creation():   Create all necessary windows for images display.

  - data_loading():       Loads from a given file .png, normal images,
                          doted image, distance image.

  - coco_treatment():     Loads and treats coco annotations. It extracts the
                          necessary information.

  - get_category():       Obtains from the full list of coco categories a
                          category for a given id.

  - crop_creation():      Creates augmented crops from the given images based
                          on the bboxes.

  - dots_detection():     Detects on the doted image the green dots. Select
                          the preferred dot (anchor point) and determines it's
                          center.

  - data_saving():        Determines the optimal point and saves the cropped
                          image and the information of the point.

Prompt commands:
python Prepare_Dataset_Regressor.py output output_aruco G1_a Regressor_dataset
"""

import os
import numpy as np
import cv2
import h5py
import json
import copy
import multiprocessing
from array import array
from PIL import Image
from typing import Tuple
from utils.fix_crop import fix_crop
from utils.aruco_utils.aruco import aruco_utils
from tqdm import tqdm

# Configuration
SINGLE_THREAD = False
DISPLAY = False
POSTPROCESS_BASE_DATASET = False


class Postprocess():

    def __init__(self, arucos_part):
        self.path = None
        self.input_paths = 'output'
        self.input_aruco_paths = 'output_aruco'
        self.input_category = arucos_part
        self.output_paths = 'dataset_4_regressor'
        self.previous_images = None
        self.base_case_images = None
        self.id = None

    def set_path(self, path):
        """ Set the path preceeding the classic "dataset_4_regressor" folder

        Args:
            path (str): the path preceeding the "dataset_4_regressor" folder

        """
        self.path = path

    def set_base_case_previous_images(self, path: str = None, hardcode: bool = False, hardvalue: int = None):
        if not hardcode:
            numbers = []
            if os.path.isdir(path):
                for file in sorted(os.listdir(path)):
                    if file.endswith(".png"):
                        numbers.append(int(file[:-4]))
            if not numbers:
                numbers.append(0)
            self.base_case_images = max(numbers) + 1
            self.previous_images = self.base_case_images

        else:
            self.base_case_images = hardvalue
            self.previous_images = self.base_case_images

    def update_num_imgs(self):
        imgs_path = os.path.join(self.path, self.output_paths, "images")
        if os.path.isdir(imgs_path):
            imgs_counter = 0
            for file in sorted(os.listdir(imgs_path)):
                if file.endswith(".png"):
                    imgs_counter += 1
            self.previous_images = self.base_case_images + imgs_counter

    def files_management(self) -> tuple[int, str, str, str, str]:
        """
        Creates and opens all necessary directories and files.

        Args:
            None

        Returns:
            num_images: Number of images detected on save path [int]
            path: Current working directory [str]
            base_path: Input directory of regular images [str]
            base_aruco_path: Input directory of aruco images [str]
            save_path: output directory to store treated data [str]
        """
        # Visualize all given files
        path = self.path
        base_path = os.path.join(path, self.input_paths)
        base_aruco_path = os.path.join(path, self.input_aruco_paths)
        save_path = os.path.join(
            path, self.output_paths)

        # Detect previous files on output
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.mkdir(save_path + '/images')
            os.mkdir(save_path + '/labels')
        try:
            num_images = len([name for name in os.listdir(
                save_path) if name.endswith('.png')])
        except NameError:
            print("No previous images detected")
            num_images = 0

        return num_images, path, base_path, base_aruco_path, save_path

    def windows_creation():
        """
        Handler of cv2 windows. Creates all necessary windows for images display.

        Args:
            None

        Returns:
            None
        """
        cv2.destroyAllWindows()
        cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Original', 640, 480)
        cv2.namedWindow('Crop', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Crop', 640, 480)
        return

    def data_loading(self, base_path: str, folder: str) \
            -> Tuple[np.array, np.array, list[dict], np.array]:
        """
        Loads from a given directory the RGB, normal images and
        segmentation map contained inside a hdf5 file.

        Args:
            - base_path: Input directory of regular images [str]
            - folder: Input folder to load from [str]

        Returns:
            - image: RGB numpy array [uint8]
            - normals: normals numpy array [float32]
            - instance: relation between object and instances
            - instance_segmap: segmentation map based on instances
        """
        image = np.empty(0, dtype=np.uint8)
        normals = np.empty(0, dtype=np.uint8)
        instance = np.empty(0, dtype=np.uint8)
        instance_segmaps = np.empty(0, dtype=np.uint8)

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
                        if key == 'instance_attribute_maps':
                            instance = json.loads(np.array(data[key]).tobytes())
                        if key == 'instance_segmaps':
                            instance_segmaps = np.array(data[key])
        return image, normals, instance, instance_segmaps

    def coco_treatment(self, base_path: str, folder: str) \
            -> Tuple[list[Tuple], list[str], list[int]]:
        """
        Loads and treats coco annotations. It extracts the
        necessary information.

        Args:
            base_path: Input directory of regular images [str]
            folder: Input folder to load from [str]

        Returns:
            bboxes: List of bboxes obtained [x, y, width, height] [list]
            categories: List of strings containing the categories names [list]
        """
        # Load of coco annotations
        for data in os.listdir(os.path.join(base_path, folder)):
            if data.startswith('coco'):
                # Read coco_annotations configuration
                with open(os.path.join(base_path, folder, data)) as f:
                    annotations_all = json.load(f)
                    annotations_categories = annotations_all['categories']
                    annotations = annotations_all['annotations']

        # Extraction of information
        image_idx = 0
        bboxes = []
        categories = []
        instances = []
        for annotation in annotations:
            if annotation["image_id"] == image_idx:
                bb = annotation['bbox']
                bboxes.append((bb[0], bb[1], bb[2], bb[3]))
                if POSTPROCESS_BASE_DATASET == False:
                    categories.append(self.get_category(
                                      annotation["category_id"], annotations_categories))
                else:
                    categories.append(annotation['id'])
        # if POSTPROCESS_BASE_DATASET:
        #     print(categories)
        #     print(a)
        return bboxes, categories

    def get_category(self, id: int, categories: list[dict]) -> str:
        """
        Obtains from the full list of coco categories a category for a given id

        Args:
            id: identification tag to look for
            categories: full list of coco categories

        Returns:
            category: the expected category if it's found
        """
        category = [category["name"]
                    for category in categories if category["id"] == id]
        if len(category) != 0:
            return category[0]
        else:
            raise Exception("Category {} is not defined".format(id))

    def draw_bboxes(image: array, bboxes: list[Tuple], categories) -> array:
        """
        Draw bboxes on a image given the category and the bboxes

        Args:
            - image: to be used as template to draw on
            - bboxes: list of [x, y, width, height]
            - categories: full list of coco categories

        Returns:
            - bboxes_image: copy of image with bboxes drawn on top
        """
        bboxes_image = copy.deepcopy(image)
        for i in range(len(categories)):
            category = categories[i]
            # Change of format: x1,y1,h,w -> x1,y1,x2,y2
            bbox = [bboxes[i][0], bboxes[i][1],
                    (bboxes[i][0] + bboxes[i][2]), (bboxes[i][1] + bboxes[i][3])]
            center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
            cv2.rectangle(
                bboxes_image, (bbox[0], bbox[1]),
                (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(bboxes_image, category,
                        (center[0], center[1]), 0, 2, (0, 255, 0), 4)
        return bboxes_image

    def crop_creation(self, img: array, bbox: Tuple, padding: bool,
                      respect_id: int = None, seg_map: array = []) -> array:
        """
        Creates crops from the given images based on the bboxes

        Args:
            - image: to be used as template to draw on
            - bboxes: list of [x, y, width, height]

        Returns:
            - final_crop: copy of image with crop applied
        """
        image = copy.deepcopy(img)

        if respect_id:
            image = self.hide_other_parts(image=image, respect_id=respect_id, seg_map=seg_map)

        if padding:
            instance_fix_crop = fix_crop()
            # Given a crop, fix dimensions (no padding checked still here)
            fixed_crop_dims = instance_fix_crop.update_crop_dims(bbox)

            # Extract the crop as an isolated image (padding is checked here)
            final_crop = instance_fix_crop.extract_crop(image, fixed_crop_dims)
        else:
            final_crop = image[bbox[1]:(bbox[1] + bbox[3]),
                               bbox[0]:(bbox[0] + bbox[2])]

        return final_crop

    def hide_other_parts(self, image, respect_id, seg_map):
        seg_map = np.where(seg_map > respect_id, 0, seg_map)
        seg_map = np.where(seg_map < respect_id, 0, seg_map)
        seg_map = np.where(seg_map == respect_id, 255, seg_map)

        # image = cv2.bitwise_and(image, image, mask=seg_map)
        # img_copy = copy.deepcopy(image)
        # width = int(img_copy.shape[1] * 0.2)
        # height = int(img_copy.shape[0] * 0.2)
        # dsize = (width, height)
        # img_copy = cv2.resize(img_copy, dsize)
        # cv2.imshow('Img', img_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image

    def segmentation_detect(self, bbox: list[Tuple],
                            corners: list[Tuple],
                            total_padding: list[int],
                            segmentation_map: array) -> Tuple:
        """
        Creates crops from the given images based on the bboxes

        Args:
            - corners: list of [Top x, Top y, Bottom x, Bottom y]
            - segmentation_map: in array format with the use of ids/instance

        Returns:
            - indexes: Tuple of the valid indexes
        """
        # Obtain the box index to remove it (The one that appears the most in the seg. map)
        box_value = np.unique(segmentation_map)[0]
        # print("")
        # print("Box value:                   ", box_value)

        indexes = []
        # TODO send crop_seg to CSV
        crop_seg, _ = self.crop_creation(segmentation_map, bbox, padding=True)
        vals, counts = np.unique(crop_seg, return_counts=True)

        remove_pos = np.where(vals == box_value)
        vals = np.delete(vals, remove_pos)
        counts = np.delete(counts, remove_pos)

        # index = np.argmax(counts)
        # print("Valores segmentacion:        ", vals)
        # print("Numero de veces que aparece: ", counts)
        # print("Index:                       ", index)

        # if vals[index] == 1:
        #     counts[index] = 0
        #     index = np.argmax(counts)
        #     center_instance = vals[index]
        # else:

        center_instance = self.id
        # print("Center instance: ", center_instance)

        # TODO Redundant from Ignacio's code. But let it be here for a while
        for i in range(len(corners)):
            point = ((corners[i][0][0] + corners[i][0][2]) / 2 + (corners[i][0][0] + corners[i][0][2]) / 2) / 2
            point_instance = crop_seg[int(point[1])][int(point[0])]
            # print(point)
            # print("Point instance:", point_instance)
            if point_instance == center_instance:
                indexes.append(i)

        return indexes

    def normal_extraction(self, normals: array, corners: list[Tuple]) -> list[Tuple]:
        """
        Extracts and formats normals from the center of a bbox and a normal image

        Args:
            - normals: array containing all normals
            - corners: list of[Top x, Top y, Bottom x, Bottom y]

        Returns:
            - normals_unitary: list of normal vector to the center of
            the given corners
        """
        normals_unitary = []
        for i in range(len(corners)):
            point = ((corners[i][0][0] + corners[i][0][2]) / 2 +
                     (corners[i][0][0] + corners[i][0][2]) / 2) / 2
            x = int(point[0])
            y = int(point[1])
            normals_cords = np.asanyarray(
                [normals[y][x][0], normals[y][x][1], normals[y][x][2]])
            normals_cords = np.asanyarray(
                [normals_cords[1], 1 - normals_cords[0], normals_cords[2]])
            normals_unitary.append(
                np.around((normals_cords - [0.5, 0.5, 0.5]) * 2, 2))

        return normals_unitary

    def data_saving(self, lock, save_path: str, image_crop: array,
                    corners: array, Point_names: list[str], normal_coords: array,
                    name: int):
        """
        Saves the cropped image and the information of the point.

        Args:
            - num_images: int value for the image to be named after
            - save_path: directory where to store the image
            - image_crop: array to be saved
            - corners: list of[Top x, Top y, Bottom x, Bottom y]
            - Point_names: list of points present on the image
            - normals_coords: normal vectors to the center of each detected point

        Returns:
            - None
        """
        width, height, _ = image_crop.shape
        target_resolution = (640, 640)

        # Save image in new directory
        image = Image.fromarray(image_crop.astype(np.uint8))
        image = image.resize(target_resolution)

        name = '{:06d}.png'.format(name)
        save_image_path = os.path.join(save_path, "images/", "{}".format(name))
        image.save(save_image_path)

        # Data for Regression
        for i in range(len(Point_names)):
            # Creation/open of txt
            Point = Point_names[i]
            txt_name = Point + ".txt"
            save_txt_path = os.path.join(
                save_path, "labels/", "{}".format(txt_name))
            file_exists = os.path.exists(save_txt_path)

            bbox = corners[i][0]

            center = (bbox[0] + bbox[1] + bbox[2] + bbox[3]) / 4
            center = (center[0] / width, center[1] / height)

            u = normal_coords[i][0]
            v = normal_coords[i][1]
            w = normal_coords[i][2]

            lock.acquire()
            txt = open(save_txt_path, "a")

            # Creation of headers
            if not file_exists:
                txt.write("Name, width, height, u, v, w\n")

            # Write on txt
            txt.write("{} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(
                name[:], center[0], center[1], u, v, w))
            txt.close()
            lock.release()
        return

    def images_processing(self, total_proccess, lock, aruco_instance,
                          base_path, base_aruco_path, save_path, folder) -> None:
        # Obtain all files inside each folder
        image, normals, attributes, instance_segmap = self.data_loading(
            base_path, folder)

        image_aruco, _, _, _ = self.data_loading(base_aruco_path, folder)

        # Load and treatment of Coco Annotations
        bboxes, categories = self.coco_treatment(
            base_path, folder)

        # Analyze and detect indexes of interest
        id_category = self.input_category

        if POSTPROCESS_BASE_DATASET:
            indexes = np.unique(instance_segmap)
            # Remove the first two positions corresponding to the cube and the plane
            indexes = indexes[2:]
        else:
            indexes = [index for index, value in enumerate(
                categories) if id_category in value]

        # Analysis of each piece
        for j, i in enumerate(indexes):
            # Creation of crops and category
            if POSTPROCESS_BASE_DATASET:
                category = i
                bbox = [bboxes[j][0], bboxes[j][1], bboxes[j][2], bboxes[j][3]]
            else:
                category = categories[i]
                bbox = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]

            if POSTPROCESS_BASE_DATASET:
                self.id = category
            else:
                self.id = self.get_id(attributes=attributes, category=category)

            image_crop, total_padding = self.crop_creation(
                image, bbox, padding=True)
            image_aruco_crop, _ = self.crop_creation(
                image_aruco, bbox, padding=True, respect_id=self.id, seg_map=instance_segmap)
            image_normals_crop, _ = self.crop_creation(
                normals, bbox, padding=True)

            # Aruco detection
            corners, ids, Point_names, img_r \
                = aruco_instance.detect_markers(image_aruco_crop,
                                                matrix_flag=True,
                                                distortion_flag=True,
                                                display=True)

            # Post process with segmentation to only detect the points
            # corresponding the detected piece
            indexes = self.segmentation_detect(
                bbox, corners, total_padding, instance_segmap)
            corners_clean = []
            ids_clean = []
            Point_names_clean = []

            for i in indexes:
                corners_clean.append(corners[i])
                ids_clean.append(ids[i][0])
                Point_names_clean.append(Point_names[i])

            # cv2.imshow("aruco_detection", img_r)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imwrite("".join([os.getcwd(), str(random.randint(0, 5000)).zfill(6), ".png"]), img_r)

            # Extraction of normals and saving
            if ids_clean:
                # Extraction of normals
                normal_coords = self.normal_extraction(
                    image_normals_crop, corners_clean)

                # Count total of images post-processed
                with total_proccess.get_lock():
                    if not self.saliency:
                        name = total_proccess.value + self.previous_images
                        total_proccess.value += 1
                    else:
                        name = self.name

                # Saving of images and data
                self.data_saving(lock, save_path, image_crop,
                                 corners_clean, Point_names_clean,
                                 normal_coords, name)

            # Display images
            if DISPLAY and SINGLE_THREAD:
                cv2.imshow('Original', image[..., ::-1])
                cv2.imshow('Crop', image_crop[..., ::-1])
                cv2.waitKey(0)

        if self.previous_images == self.base_case_images:
            self.previous_images += total_proccess.value
        return

    def get_id(self, attributes, category):
        for attribute in attributes:
            if attribute['name'] == category:
                id = attribute['idx']
                break
        return id

    def extract_labels(self, raw_img_counter, ignore_past=False, angle=None):

        # Initialize image count
        total_proccess = multiprocessing.Value('i', 0)

        base_path = os.path.join(self.path, self.input_paths)

        folders_list = []
        folder = str(raw_img_counter).zfill(6)
        folders_list.append(folder)

        # Lock when saving images
        lock = multiprocessing.Lock()

        # Manege file systems
        num_images, path, base_path, base_aruco_path, \
            save_path = self.files_management()

        # Initialize Aruco
        aruco_instance = aruco_utils()

        # Initialize multiprocessing
        if not SINGLE_THREAD:
            num_cores = multiprocessing.cpu_count() - 4
            pool = multiprocessing.Pool(processes=num_cores)

        # Initialize image visualization
        if DISPLAY and SINGLE_THREAD:
            self.windows_creation()

        # Obtain the number of previous images
        if not ignore_past:
            self.update_num_imgs()
        else:
            self.previous_images = 0
            self.base_case_images = 0
            self.name = int(angle)
            self.saliency = ignore_past
        # Obtain everything inside input_paths
        try:
            if SINGLE_THREAD:
                for folder in folders_list:
                    self.images_processing(
                        total_proccess=total_proccess,
                        lock=lock,
                        aruco_instance=aruco_instance,
                        base_path=base_path,
                        base_aruco_path=base_aruco_path,
                        save_path=save_path,
                        folder=folder
                    )
            else:
                for folder in folders_list:
                    pool.apply_async(self.images_processing(
                        total_proccess=total_proccess, 
                        lock=lock, 
                        aruco_instance=aruco_instance, 
                        base_path=base_path, 
                        base_aruco_path=base_aruco_path, 
                        save_path=save_path, 
                        folder=folder))

        except KeyboardInterrupt:  # Press Ctrl+C to break the for loop and gracefully stop the simulation
            print(f'Keyboard interrupt')

        if not SINGLE_THREAD:
            pool.close()
            pool.join()

        # Final report
        print(f'Obtained a total of {total_proccess.value} crops')


if __name__ == "__main__":

    PART = 'G1_a'

    if POSTPROCESS_BASE_DATASET:
        SSD_PATH = os.path.join('/media', 'msi', 'SSD-2TB')
        ROOT_PATH = os.path.join(SSD_PATH, 'database', 'BlenderProc', PART)
        NO_ARUCO_PATH = os.path.join(ROOT_PATH, 'output')
        ARUCO_PATH = os.path.join(ROOT_PATH, 'output_aruco')
        SAVE_PATH = os.path.join(ROOT_PATH, 'new_postprocess')

        imgs_path = os.path.join(SAVE_PATH, 'images')
        lbls_path = os.path.join(SAVE_PATH, 'labels')
        intermediate_path = os.path.join(SAVE_PATH, 'dataset_4_regressor')

        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        if not os.path.isdir(intermediate_path):
            os.mkdir(intermediate_path)
        if not os.path.isdir(imgs_path):
            os.mkdir(imgs_path)
        if not os.path.isdir(lbls_path):
            os.mkdir(lbls_path)

        post_instance = Postprocess(PART)
        post_instance.set_base_case_previous_images(imgs_path)

        post_instance.set_path(os.path.join(ROOT_PATH))

        img_counter = 1
        for img in tqdm(sorted(os.listdir(NO_ARUCO_PATH))):
            print(img)
            post_instance.extract_labels(img_counter)
            img_counter += 1

    else:
        COUNTER = 400000 #TODO

        SSD = os.path.join("/media", "msi", "SSD_CIC")

        PATH = os.path.join(SSD, "".join([PART, "_flatten_dataset"]),
                            "2_finde_Nic_box_0.8_to_0.9_projection") #TODO

        ARUCO_PATH = os.path.join(PATH, "output")
        NO_ARUCO_PATH = os.path.join(PATH, "output_aruco")

        post_instance = Postprocess(PART)
        post_instance.set_base_case_previous_images(hardcode = True, hardvalue = COUNTER)

        post_instance.set_path(os.path.join(PATH))

        img_counter = 0
        for img in tqdm(sorted(os.listdir(NO_ARUCO_PATH))):
            print(img)
            post_instance.extract_labels(img_counter)
            img_counter += 1