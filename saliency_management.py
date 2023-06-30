import os
import sys
import time
import random

# This enables python to look for the Postprocess module
sys.path.append(os.getcwd())
from postprocess import Postprocess
from time_it import Time_it

PART = "G1_a"
ARUCO_PART = "".join([PART, "_ArUco"])
BOX_NAME = "flat"
FLATTEN_POINT = 2
SEED = 0
SALIENCY = True

PATH = os.path.join(
    os.getcwd(), "".join([PART, "_Saliency"])
)

post_instance = Postprocess(PART)
time_instance = Time_it()

post_instance.set_path(os.path.join(PATH, BOX_NAME))

space = " "
produced_imgs_in_this_run = 0

for z_angle in range(0, 360, 1):
    init_tic = time.time()
    img_counter = z_angle
    x_angle = random.randint(0,61)
    print(FLATTEN_POINT)
    command = "".join(["blenderproc run generator.py", space, PART, space, ARUCO_PART, space, str(FLATTEN_POINT),
                      space, BOX_NAME, space, str(SEED), space, str(img_counter), space, str(SALIENCY), space, str(z_angle), space, str(x_angle)])
    os.system(command)
    time_instance.write_to_txt(path=os.path.join(PATH, BOX_NAME), label="Attempts", message=0)
    check_path = os.path.join(PATH, BOX_NAME, "output_aruco")
    post_tic = time.time()
    if os.path.isdir(check_path):
        post_instance.extract_labels(img_counter, SALIENCY, z_angle)

    final_toc = time.time()
    time_instance.time_it(tic=post_tic, toc=final_toc, path=os.path.join(PATH, BOX_NAME), run="Post")
    time_instance.time_it(tic=init_tic, toc=final_toc, path=os.path.join(PATH, BOX_NAME), run=produced_imgs_in_this_run)

    produced_imgs_in_this_run += 1
