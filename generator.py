import blenderproc as bproc

import os
import math
import random
import numpy as np
import argparse
import time
import sys

sys.path.append(os.getcwd())
from time_it import Time_it

parser = argparse.ArgumentParser()
parser.add_argument("PART", nargs="?", help="For instance: G1_a")
parser.add_argument("ARUCO_PART", nargs="?", help="For instance: G1_a_ArUco")
parser.add_argument("FLATTEN_POINT", nargs="?", help="For instance: 2")
parser.add_argument(
    "BOX_NAME", nargs="?", help="For instance: box_0.8_to_0.9_projection"
)
parser.add_argument("SEED", nargs="?", help="For instance: 1000")
parser.add_argument("IMG_COUNTER", nargs="?", help="For instance: 1")
parser.add_argument("SALIENCY", nargs="?", help="True or False", default=False)
parser.add_argument("ANGLE", nargs="?", help="An angle in [deg.]", default=None)
parser.add_argument("X_ANGLE", nargs="?", help="An angle in [deg.]", default=None)
args = parser.parse_args()

# PATH = os.path.join(
#     os.getcwd(), "".join([args.PART, "_flatten_dataset"])
# )

PATH = os.path.join(os.getcwd(), "".join([args.PART, "_Saliency"]))

# SSD = os.path.join("/media", "msi", "SSD-2TB")
# SSD = os.path.join("/media", "msi", "SSD_CIC")

# PATH = os.path.join(
#     SSD, "".join([args.PART, "_flatten_dataset"])
# )

# PATH = os.path.join("".join([args.PART, "_flatten_dataset"]))

OBJECTS_PATH = os.path.join(os.getcwd(), "objects", "Big", "Scenes")
PIECE_PATH = os.path.join(os.getcwd(), "objects")

DOT_PRODUCT = True
INTERVALS = 10
BOXES = 5
if DOT_PRODUCT:
    STEP = 1 / INTERVALS
else:
    STEP = 90 / INTERVALS

# Normal vector projections:
# if projection in x -> PROJECTION = 1,
# if projection in Y -> PROJECTION = 2,
# if projection in Z -> PROJECTION = 3
PROJECTION = 3

PART_AND_POINT = "".join([args.PART, "_Point_", str(args.FLATTEN_POINT)])

COMPLETE = True

RAND_MAX = 1
RAND_MIN = 0
DICE_LIM = 0.75

MULTIPLE = False


def preload_materials():
    preloaded_materials = bproc.loader.load_ccmaterials(
        "BlenderProc/resources/cctextures", preload=True
    )

    return preloaded_materials


def load_materials():
    bproc.loader.load_ccmaterials(
        "BlenderProc/resources/cctextures", fill_used_empty_materials=True
    )


def apply_material_to_box(*, box, materials):
    box[0].new_material("Random")
    for i in range(len(box[0].get_materials())):
        # Replace the material with a random one
        box[0].set_material(i, random.choice(materials))
    return box


def load_box(*, box_name):
    for data in sorted(os.listdir(OBJECTS_PATH)):
        if data.endswith(".blend") and data.replace(".blend", "") == box_name:
            blender_box = bproc.loader.load_blend(os.path.join(OBJECTS_PATH, data))
            blender_box[0].enable_rigidbody(False, collision_shape="MESH")
            dimensions = blender_box[0].blender_obj.dimensions
            print("".join([box_name, ".blend: successfully loaded"]))

    return blender_box, dimensions


def create_bckgrnd_plane(*, materials):
    plane = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
    # plane.new_material("Random")
    # for i in range(len(plane.get_materials())):
    #     # Replace the material with a random one
    #     plane.set_material(i, random.choice(materials))
    return plane


def change_plane_material(*, plane, materials):
    for i in range(len(plane.get_materials())):
        # Replace the material with a random one
        plane.set_material(i, random.choice(materials))
    return plane


def check_piece():
    blender_piece = False
    for data in sorted(os.listdir(PIECE_PATH)):
        if data.endswith(".blend") and data.replace(".blend", "") == args.PART:
            print("".join([args.PART, ".blend: successfully found"]))
            blender_piece = True
    if blender_piece is False:
        raise Exception(
            "".join(
                [
                    "Sorry, no Blender file was found with this name: ",
                    args.PART,
                    ".blend",
                ]
            )
        )


def load_pieces():
    blender_pieces = []
    for data in sorted(os.listdir(PIECE_PATH)):
        if data.endswith(".blend") and data.replace(".blend", "") == args.PART:
            # Load it multiple times
            n_pieces = 0

            if MULTIPLE:
                dice = random.uniform(RAND_MIN, RAND_MAX)
                if dice <= DICE_LIM:
                    min_pieces = 3
                    max_pieces = 5
                else:
                    min_pieces = 7
                    max_pieces = 10

                for id in range(random.randrange(min_pieces, max_pieces)):
                    blender_piece = bproc.loader.load_blend(
                        os.path.join(PIECE_PATH, data)
                    )
                    blender_piece = turn_on_physics(piece=blender_piece)
                    blender_pieces.append(blender_piece[0])
                    name = blender_piece[0].blender_obj.name
                    n_pieces += 1
                    if not name.__contains__("Point"):
                        blender_piece[0].set_cp("category_id", id)
            else:
                blender_piece = bproc.loader.load_blend(os.path.join(PIECE_PATH, data))
                blender_piece = turn_on_physics(piece=blender_piece)
                blender_pieces.append(blender_piece[0])
                name = blender_piece[0].blender_obj.name
                n_pieces += 1
                if not name.__contains__("Point"):
                    blender_piece[0].set_cp("category_id", 0)

    return blender_pieces, n_pieces


def check_aruco_piece():
    blender_aruco_piece = False
    for data in sorted(os.listdir(PIECE_PATH)):
        if data.endswith(".blend") and data.replace(".blend", "") == args.ARUCO_PART:
            print("".join([args.ARUCO_PART, ".blend: successfully found"]))
            blender_aruco_piece = True
    if blender_aruco_piece is False:
        raise Exception(
            "".join(
                [
                    "Sorry, no Blender file was found with this name: ",
                    args.ARUCO_PART,
                    ".blend",
                ]
            )
        )


def load_aruco_pieces(*, n_pieces):
    blender_aruco_pieces = []
    for data in sorted(os.listdir(PIECE_PATH)):
        if data.endswith(".blend") and data.replace(".blend", "") == args.ARUCO_PART:
            # Load the piece the exact number of times
            for id in range(n_pieces):
                blender_piece = bproc.loader.load_blend(os.path.join(PIECE_PATH, data))
                blender_piece = turn_on_physics(piece=blender_piece)
                blender_aruco_pieces.append(blender_piece[0])
                name = blender_piece[0].blender_obj.name
                if not name.__contains__("Point"):
                    blender_piece[0].set_cp("category_id", id)

    return blender_aruco_pieces


def create_scene(*, flag):
    global n_pieces
    if flag == "no_ArUco":
        blender_pieces, n_pieces = load_pieces()
        if not args.SALIENCY:
            physics_simulations(pieces=blender_pieces, flag=flag)
        else:
            place_object(blender_pieces)
        pieces = blender_pieces
    elif flag == "ArUco":
        blender_aruco_pieces = load_aruco_pieces(n_pieces=n_pieces)
        if not args.SALIENCY:
            physics_simulations(pieces=blender_aruco_pieces, flag=flag)
        else:
            place_object(blender_aruco_pieces)
        pieces = blender_aruco_pieces
    return pieces


def delete_scene(*, blender_pieces, box):
    delete_pieces(blender_pieces=blender_pieces)
    bproc.object.delete_multiple(box)


def delete_pieces(*, blender_pieces):
    bproc.object.delete_multiple(blender_pieces)


def sample_pose(obj: bproc.types.MeshObject):
    global locations
    global rotations
    global dimensions
    min_dim = [-dimensions[0] / 2.5, -dimensions[1] / 2.5, dimensions[2]]
    max_dim = [dimensions[0] / 2.5, dimensions[1] / 2.5, dimensions[2] + 0.05]

    location = np.random.uniform(min_dim, max_dim)
    if MULTIPLE:
        rotation = bproc.sampler.uniformSO3()
    else:
        rotation = bproc.sampler.uniformSO3(
            around_x=False, around_y=False, around_z=True
        )

    locations.append(location)
    rotations.append(rotation)

    obj.set_location(location)
    obj.set_rotation_euler(rotation)


# TODO 360
def place_object(objs: list):
    """Set hardcoded location and rotation

    Args:
        objs (list): contains bproc.types.MeshObject elements
    """

    for obj in objs:
        location = np.asarray([0, 0, 0.1])
        rotation = np.asarray([math.radians(float(args.X_ANGLE)), 0, math.radians(float(args.ANGLE))])

        obj.set_location(location)
        obj.set_rotation_euler(rotation)


def sample_pose_aruco(obj: bproc.types.MeshObject):
    global locations
    global rotations

    item = 0
    location = locations.pop(item)
    rotation = rotations.pop(item)
    obj.set_location(location)
    obj.set_rotation_euler(rotation)


def turn_on_physics(*, piece):
    piece[0].enable_rigidbody(
        active=True,
        collision_shape="CONVEX_HULL",
        mass=1.0,
        friction=1.0,
        angular_damping=0.5,
        linear_damping=0.5,
    )
    return piece


def physics_simulations(*, pieces, flag):
    # Sample the poses of all pieces above the ground without any
    # collisions in-between
    if flag == "no_ArUco":
        bproc.object.sample_poses(pieces, sample_pose_func=sample_pose)
    elif flag == "ArUco":
        bproc.object.sample_poses(pieces, sample_pose_func=sample_pose_aruco)

    # Run the simulation and fix the poses of the spheres at the end
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=1, max_simulation_time=1.5, check_object_interval=1
    )


def create_lights():
    # TODO: improve this nested foor loop combo
    # Create a light and set its properties
    for x in (-1, 1):
        for y in (-1, 1):
            light = bproc.types.Light()
            light.set_type("POINT")
            light.set_location([x, y, 2.5])
            light.set_energy(50)


def create_camera():
    # Add a camera pose via location + euler angles
    bproc.camera.add_camera_pose(
        bproc.math.build_transformation_mat([0, 0, 3], [0, 0, math.pi / 2])
    )
    # Define the camera resolution
    bproc.camera.set_resolution(3840, 2160)
    # TODO: Set intrinsics via K matrix -> Pendiente de introducir
    """bproc.camera.set_intrinsics_from_K_matrix(
    [[1367.14, 0.0, 973.89],
    [0.0, 1368.28, 526.45],
    [0.0, 0.0, 1.0]], 1920, 1080)"""


def configure_render():
    # Activate denoiser
    bproc.renderer.set_denoiser(None)
    bproc.renderer.set_noise_threshold(0.1)
    # Define file format
    bproc.renderer.set_output_format(file_format="PNG")
    if COMPLETE:
        # Activate normal rendering
        bproc.renderer.enable_normals_output()
        # Activate depth rendering
        # bproc.renderer.enable_depth_output(activate_antialiasing=False)
        # Set the amount of samples, which should be used for the color rendering
        bproc.renderer.set_max_amount_of_samples(50)
    else:
        bproc.renderer.set_max_amount_of_samples(25)


def render(output_path):
    if COMPLETE:
        data = bproc.renderer.render(return_data=True)
        seg_data = bproc.renderer.render_segmap(map_by=["class", "instance", "name"])
        data.update(seg_data)
        write_output(seg_data=seg_data, data=data, output_path=output_path)
    else:
        data = bproc.renderer.render(return_data=True)
        write_output(data=data, output_path=output_path)


def write_output(seg_data, data, output_path):
    if COMPLETE:
        # Write data to coco file
        bproc.writer.write_coco_annotations(
            output_dir=output_path,
            instance_segmaps=seg_data["instance_segmaps"],
            instance_attribute_maps=seg_data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="PNG",
        )

        # Write the data to a .hdf5 container
        bproc.writer.write_hdf5(output_path, data)
    else:
        bproc.writer.write_hdf5(output_path, data)


if __name__ == "__main__":
    try:
        time_instance = Time_it()

        random.seed(int(args.SEED))
        np.random.seed(int(args.SEED))

        # Check the piece to render exists
        check_piece()
        check_aruco_piece()

        # Start BlenderProc
        bproc.init(clean_up_scene=True)
        img_counter = 0

        # Start the scene creation + rendering + saving
        pre_materials = preload_materials()
        create_lights()
        create_camera()
        configure_render()
        textured_plane = create_bckgrnd_plane(materials=pre_materials)

        output_path = os.path.join(
            PATH, args.BOX_NAME, "output", args.IMG_COUNTER.zfill(6)
        )
        aruco_output_path = os.path.join(
            PATH, args.BOX_NAME, "output_aruco", args.IMG_COUNTER.zfill(6)
        )
        locations = []
        rotations = []
        n_pieces = 0
        if not args.SALIENCY:
            box, dimensions = load_box(box_name=args.BOX_NAME)
            textured_box = apply_material_to_box(box=box, materials=pre_materials)
        load_materials()

        # Render with no ArUco
        simple_scene_tic = time.time()
        blender_pieces = create_scene(flag="no_ArUco")
        simple_scene_toc = time.time()
        time_instance.time_it(
            tic=simple_scene_tic,
            toc=simple_scene_toc,
            path=os.path.join(PATH, args.BOX_NAME),
            run="Simple_Scene",
        )

        simple_render_tic = time.time()
        render(output_path)
        simple_render_toc = time.time()
        time_instance.time_it(
            tic=simple_render_tic,
            toc=simple_render_toc,
            path=os.path.join(PATH, args.BOX_NAME),
            run="Simple_Render",
        )

        # Render the ArUco image if the previous 'no ArUco' image was rendered, so no time is wasted
        check_path = os.path.join(PATH, args.BOX_NAME, "output")
        if os.path.isdir(check_path):
            # Delete no ArUco parts
            bproc.object.delete_multiple(blender_pieces)

            # Render with ArUco
            aruco_scene_tic = time.time()
            blender_aruco_pieces = create_scene(flag="ArUco")
            aruco_scene_toc = time.time()
            time_instance.time_it(
                aruco_scene_tic,
                aruco_scene_toc,
                os.path.join(PATH, args.BOX_NAME),
                "ArUco_Scene",
            )

            aruco_render_tic = time.time()
            render(aruco_output_path)
            aruco_render_toc = time.time()
            time_instance.time_it(
                tic=aruco_render_tic,
                toc=aruco_render_toc,
                path=os.path.join(PATH, args.BOX_NAME),
                run="ArUco_Render",
            )

        time_instance.write_to_txt(
            path=os.path.join(PATH, args.BOX_NAME), label="Parts", message=n_pieces
        )

    except (
        KeyboardInterrupt
    ):  # Press Ctrl+C to break the for loop and gracefully stop the simulation
        print(f"Keyboard interrupt")
