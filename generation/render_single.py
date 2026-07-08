# This file contains code derived from:
# - https://github.com/CapArrow/zendo_game_dataset_generator
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from argparse import Namespace
import csv
import gc
import yaml
import ast
import platform
from generation.zendo_objects import *
from generation.generate import generate_structure, get_image_bounding_box
from generation.encoding import convert_to_tensor
import generation.utils as utils


def render(args, name):
    """Render a single scene with Blender's Cycles engine; returns elapsed render time."""

    # ── Compute device ────────────────────────────────────────────────────
    system = platform.system()
    preferences = bpy.context.preferences.addons["cycles"].preferences

    if system == "Darwin":
        preferences.compute_device_type = "METAL"
    elif system in ["Windows", "Linux"]:
        preferences.compute_device_type = "OPTIX"
    else:
        preferences.compute_device_type = "NONE"

    preferences.get_devices()

    if preferences.compute_device_type in ["OPTIX", "METAL"]:
        bpy.context.scene.cycles.device = "GPU"
    else:
        bpy.context.scene.cycles.device = "CPU"

    for device in preferences.devices:
        if preferences.compute_device_type == "OPTIX" and device.type in ["OPTIX", "CUDA"]:
            device.use = True
        elif preferences.compute_device_type == "METAL" and device.type in ["GPU", "CPU"]:
            device.use = True
        elif preferences.compute_device_type == "NONE" and device.type == "CPU":
            device.use = True
        else:
            device.use = False

    debug(f"Using compute_device_type: {preferences.compute_device_type}")
    debug(f"Render device set to: {bpy.context.scene.cycles.device}")
    for device in preferences.devices:
        debug(f"Device: {device.name}, Type: {device.type}, Active: {device.use}")

    # ── Render ────────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.realpath(__file__))

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.filepath = os.path.join(name)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.cycles.samples = int(args.render_num_samples)
    bpy.context.scene.render.resolution_x = args.width
    bpy.context.scene.render.resolution_y = args.height
    bpy.context.scene.render.resolution_percentage = 100

    debug(f"Saving output image to: {bpy.context.scene.render.filepath}")

    # Redirect Blender's chatty stdout to a logfile.
    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.close(sys.stdout.fileno())
    fd = os.open(logfile, os.O_WRONLY)

    render_start = time.time()
    bpy.ops.render.render(write_still=True)
    render_end = time.time()

    render_duration = render_end - render_start

    os.close(fd)
    os.dup(old)
    os.close(old)

    if args.save_blendfile:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.output_dir, f"{name}.blend"))

    return render_duration

def generate_blender_examples(args, scene):
    """Build, render, and tear down a Blender scene; return (success, render_time, cpu_time)."""
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
      """Render the scene given by ``args.scene`` and write the encoded tensor next to the image."""
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

      # Failure path is generally a Prolog timeout; the caller can retry.
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
    parser.add_argument("-c", "--config-file", type=str, default="generation/configs/simple_config.yml",
                        help='config file for rendering')
    parser.add_argument("--scene", type=str, default=None,
                    help="Scene to render")
    parser.add_argument("--path", type=str, default=None,
                    help="Path to save the rendered scene")
    conf = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    with open(conf.config_file) as f:
        args = yaml.safe_load(f.read())
    args = Namespace(**args)
    if conf.scene is not None:
        args.scene = conf.scene
    if conf.path is not None:
        args.path = conf.path

    utils.DEBUG_PRINTING = args.debug_printing

    main(args)