import time
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
EXTRA_SITE = os.environ.get("BLENDER_PYTHONPATH", "")
if EXTRA_SITE and EXTRA_SITE not in sys.path:
    sys.path.insert(0, EXTRA_SITE)
import argparse
from argparse import Namespace
from zendoworld.paths import RENDER_CONFIG, output_path
import csv
import gc
import yaml
import ast
import platform
from zendoworld.rendering.zendo_objects import *
from zendoworld.rendering.generate import generate_structure, get_image_bounding_box
from zendoworld.rendering.encoding import convert_to_tensor
import zendoworld.rendering.utils as utils


def render(args, name):
    """
    Renders a scene using Blender's Cycles engine with specified settings.

    This function sets up the rendering configuration, including the compute device,
    resolution, sampling, and output file format. It then performs the rendering
    and optionally saves the Blender scene file.

    :param args: Configuration arguments for rendering, including resolution,
                 sample count, output directory, and rendering options.
    :param output_path: The subdirectory within the output directory where
                        the rendered image will be saved.
    :param name: The name of the rendered image file (without extension).
    """

    #######################################################
    # Initialize render settings
    #######################################################

    # Detect system OS and configure the best rendering settings
    system = platform.system()
    preferences = bpy.context.preferences.addons["cycles"].preferences

    # Set the best compute device type based on the OS. CUDA works in our
    # container (verified via manual enumeration); OPTIX silently hangs.
    override = os.environ.get("ZENDO_CYCLES_DEVICE")
    if override:
        preferences.compute_device_type = override.upper() if override.upper() != "CPU" else "NONE"
    elif system == "Darwin":
        preferences.compute_device_type = "METAL"
    elif system in ["Windows", "Linux"]:
        preferences.compute_device_type = "CUDA"
    else:
        preferences.compute_device_type = "NONE"

    preferences.get_devices()

    # Set render device to GPU if available; otherwise, use CPU
    if preferences.compute_device_type in ["OPTIX", "CUDA", "METAL"]:
        bpy.context.scene.cycles.device = "GPU"
    else:
        bpy.context.scene.cycles.device = "CPU"

    # Explicitly activate the available devices based on compute_device_type
    for device in preferences.devices:
        if preferences.compute_device_type == "OPTIX" and device.type in ["OPTIX", "CUDA"]:
            device.use = True
        elif preferences.compute_device_type == "CUDA" and device.type == "CUDA":
            device.use = True
        elif preferences.compute_device_type == "METAL" and device.type in ["GPU", "CPU"]:
            device.use = True
        elif preferences.compute_device_type == "NONE" and device.type == "CPU":
            device.use = True
        else:
            device.use = False

    # Get the directory of the executing Python script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.filepath = os.path.join(name)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.cycles.samples = int(args.render_num_samples)
    bpy.context.scene.render.resolution_x = args.width
    bpy.context.scene.render.resolution_y = args.height
    bpy.context.scene.render.resolution_percentage = 100

    debug(f"Saving output image to: {bpy.context.scene.render.filepath}")

    logfile = str(output_path('logs', 'blender_render.log'))
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    saved_stdout = os.dup(sys.stdout.fileno())
    try:
        with open(logfile, 'a') as log:
            sys.stdout.flush()
            os.dup2(log.fileno(), sys.stdout.fileno())
            render_start = time.time()
            bpy.ops.render.render(write_still=True)
            render_end = time.time()
            sys.stdout.flush()
    finally:
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.close(saved_stdout)


    render_duration = render_end - render_start

    if args.save_blendfile:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.output_dir, f"{name}.blend"))

    return render_duration

def generate_blender_examples(args, scene):
    total_start = time.time()
    render_time_total = 0.0

    collection = bpy.data.collections.new("Structure")
    bpy.context.scene.collection.children.link(collection)

    rule_output_dir = args.output_dir
    os.makedirs(rule_output_dir, exist_ok=True)

    rule_name = f"query"
    scene_path = os.path.join(rule_output_dir, args.path, rule_name + ".txt")
    os.makedirs(os.path.dirname(scene_path), exist_ok=True)
    scene_name = os.path.join(rule_output_dir, args.path)
    examples_data = []
    try:
        generate_structure(args, scene, collection, grounded="grounded" in scene)
        render_time = render(args, scene_name)
        render_time_total += render_time

        scene_objects = ZendoObject.instances
        for obj in scene_objects:
            x_min, y_min, x_max, y_max = get_image_bounding_box(obj, bpy.context.scene)

            examples_data.append([
                obj.name, obj.touching, obj.pointing,
                x_min, y_min, x_max, y_max
            ])

        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        ZendoObject.instances.clear()

        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

    except Exception as e:
        print(f"❌ Exception during scene generation {scene_name}: {e}")
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        ZendoObject.instances.clear()

    if collection.name in bpy.data.collections:
        bpy.data.collections.remove(collection, do_unlink=True)
    for col in bpy.data.collections:
        if col.users == 0:
            bpy.data.collections.remove(col)
    bpy.ops.outliner.orphans_purge(do_recursive=True)

    gc.collect()

    if examples_data:
        convert_to_tensor(examples_data, os.path.join(scene_name + ".pt"))

    total_end = time.time()
    cpu_time = total_end - total_start - render_time_total
    return bool(examples_data), render_time_total, cpu_time

def main(args):
      start_time = time.time()
      script_dir = os.path.dirname(bpy.data.filepath)
      if script_dir not in sys.path:
            sys.path.append(script_dir)

      bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
      os.makedirs(args.output_dir, exist_ok=True)

      total_gpu_time = 0.0
      total_cpu_time = 0.0
      total_failed_time = 0.0
      failed_attempts = 0
      try:
            scene = ast.literal_eval(args.scene)
      except Exception as e:
            print("Failed to parse argument using ast.literal_eval:", e)
            return
      collection = bpy.data.collections.new("Structure")
      bpy.context.scene.collection.children.link(collection)

      attempt_start = time.time()
      generated_successfully, render_time, cpu_time = generate_blender_examples(args, scene)
      attempt_end = time.time()

      # If result is not true, then prolog query took to long, therefore try again
      if not generated_successfully:
            total_failed_time += (attempt_end - attempt_start)
            failed_attempts += 1

      total_gpu_time += render_time
      total_cpu_time += cpu_time

      debug(f"\nDataset generation complete.")

      debug(f"\nTime to complete: {(time.time() - start_time):.2f}s")
      debug(f"Total GPU time: {total_gpu_time:.2f}s")
      debug(f"Total CPU time: {total_cpu_time:.2f}s")
      debug(f"Total failed attempts time: {total_failed_time:.2f}s")
      debug(f"Total failed attempts: {failed_attempts}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, default=str(RENDER_CONFIG),
                        help='config file for rendering')
    parser.add_argument("--scene", type=str, default=None,
                    help="Scene to render")
    parser.add_argument("--path", type=str, default=None,
                    help="Path to save the rendered scene")
    conf = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    with open(conf.config_file) as f:
        args = yaml.safe_load(f.read())
    args = Namespace(**args)
    # Resolve asset paths relative to the selected config, including custom configs.
    config_dir = Path(conf.config_file).resolve().parent
    for field in ("base_scene_blendfile", "properties_json", "shape_dir", "rules_prolog_file"):
        path = Path(getattr(args, field))
        if not path.is_absolute():
            setattr(args, field, str((config_dir / path).resolve()))
    if args.output_dir == "outputs/rendering":
        args.output_dir = str(output_path("rendering"))
    if conf.scene is not None:
        args.scene = conf.scene
    if conf.path is not None:
        args.path = conf.path

    utils.DEBUG_PRINTING = args.debug_printing

    main(args)
