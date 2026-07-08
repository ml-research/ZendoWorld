# This file contains code derived from:
# - https://github.com/CapArrow/zendo_game_dataset_generator
import ast
import re
from collections import defaultdict, deque
from math import cos, sin, pi
from generation.utils import debug
from bpy_extras.object_utils import world_to_camera_view
import json

import generation.zendo_objects as zendo_objects
from generation.structure import *


def check_pointing(observer: ZendoObject):
    """Return the list of ZendoObjects ``observer`` is pointing at via raycasts."""
    bpy.context.view_layer.update()
    results = []

    for origin, direction in observer.get_rays():
        ray_path = [origin.copy()]
        direction = direction.normalized()
        current_location = origin.copy()
        hit_location = origin.copy()
        while True:
            hit, hit_location, _, _, obj, _ = bpy.context.scene.ray_cast(
                bpy.context.view_layer.depsgraph, current_location, direction
            )
            if not hit:
                break
            else:
                zendo_obj = zendo_objects.get_from_blender_obj(obj)
                if zendo_obj is not None and zendo_obj not in results and zendo_obj is not observer:
                    ray_path.append(hit_location)
                    results.append(zendo_obj)
                current_location = hit_location + direction * 0.01
                hit_location *= 1.01

    return results


def check_collision(zendo_object: ZendoObject, omit: ZendoObject = None, margin: float = 0.0):
    """Return ZendoObjects whose AABB overlaps ``zendo_object``'s (with optional ``margin`` and ``omit``)."""
    bpy.context.view_layer.update()

    obj_bb = [zendo_object.obj.matrix_world @ Vector(corner) for corner in zendo_object.obj.bound_box]
    obj_bb_min = Vector((min(v.x for v in obj_bb), min(v.y for v in obj_bb), min(v.z for v in obj_bb))) - Vector(
        (margin,) * 3)
    obj_bb_max = Vector((max(v.x for v in obj_bb), max(v.y for v in obj_bb), max(v.z for v in obj_bb))) + Vector(
        (margin,) * 3)

    colliding_objects = []

    for other_obj in ZendoObject.instances:
        if other_obj is zendo_object or other_obj is omit:
            continue

        other_bb = [other_obj.obj.matrix_world @ Vector(corner) for corner in other_obj.obj.bound_box]
        other_bb_min = Vector(
            (min(v.x for v in other_bb), min(v.y for v in other_bb), min(v.z for v in other_bb))) - Vector(
            (margin,) * 3)
        other_bb_max = Vector(
            (max(v.x for v in other_bb), max(v.y for v in other_bb), max(v.z for v in other_bb))) + Vector(
            (margin,) * 3)

        if (obj_bb_min.x <= other_bb_max.x and obj_bb_max.x >= other_bb_min.x and
                obj_bb_min.y <= other_bb_max.y and obj_bb_max.y >= other_bb_min.y and
                obj_bb_min.z <= other_bb_max.z and obj_bb_max.z >= other_bb_min.z):
            colliding_objects.append(other_obj)

    return colliding_objects


def get_random_position(anchor, radius):
    """Sample a position uniformly inside a circle of given radius around ``anchor`` (xy-plane)."""
    angle = random.uniform(0, 2 * pi)
    distance = random.uniform(0, radius)
    x = distance * cos(angle)
    y = distance * sin(angle)

    random_position = Vector(anchor) + Vector((x, y, 0))

    return random_position


def get_grounded(instructions):
    grounded_objects = [i for i in instructions if i.get('action') == 'grounded']
    grounded_objects = sorted(grounded_objects, key=lambda x: x['id'])
    return grounded_objects


def get_relations(instructions):
    """Topologically sort ``instructions`` so each target is placed before any object referencing it."""

    def extract_target_id(action_str):
        if '(' in action_str and ')' in action_str:
            try:
                return int(action_str.split('(')[1].split(')')[0])
            except Exception:
                return None
        return None

    id_to_instruction = {obj['id']: obj for obj in instructions}
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for obj in instructions:
        source_id = obj['id']
        target_id = extract_target_id(obj['action'])
        if target_id is not None and target_id in id_to_instruction:
            # Target must be placed before the source.
            graph[target_id].append(source_id)
            in_degree[source_id] += 1
        else:
            # Touch the key so isolated nodes still appear in the dict.
            in_degree[source_id]

    queue = deque([i for i in instructions if in_degree[i['id']] == 0])
    sorted_result = []

    while queue:
        current = queue.popleft()
        sorted_result.append(current)
        for dependent_id in graph[current['id']]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(id_to_instruction[dependent_id])

    return sorted_result


def get_free_face(args, obj: ZendoObject):
    random_faces = args.random_face_choice
    faces = obj.get_free_face()
    if random_faces:
        return random.choice(faces)
    else:
        return faces[0]


def generate_relation(instruction):
    """Return ``(relation_type, target_object)`` parsed from an instruction's ``action`` field."""
    relation = instruction['action']
    relation_type = relation.split('(')[0]
    relation_target = int(relation.split('(')[1][0])
    target = zendo_objects.get_object(relation_target)

    return relation_type, target


def generate_creation(args, instructions, index, collection):
    """Instantiate the ZendoObject described by ``instructions[index]`` and add it to ``collection``."""
    instruction = instructions[index]
    idx = instruction['id']
    shape = instruction['shape']
    color = instruction['color']
    orientation = instruction['orientation']
    action = instruction['action']

    if shape == 'block':
        obj = zendo_objects.Block(args, idx, color, orientation)
    elif shape == 'wedge':
        obj = zendo_objects.Wedge(args, idx, color, orientation)
    elif shape == 'pyramid':
        obj = zendo_objects.Pyramid(args, idx, color, orientation)
    collection.objects.link(obj.obj)

    if args.random_object_rotation:
        d = random.uniform(0, 360)
        obj.rotate_z(d)
    return obj


def generate_structure(args, items: list[str], collection, attempt: int = 1, grounded=False):
    """
    Generates a structured scene based on a set of item descriptions and configuration parameters.

    :param args: Configuration arguments for scene generation.
    :param items: A list of item descriptions in Prolog format.
    :param collection: The Blender collection where the generated objects will be stored.
    :param attempt: The current attempt count for scene generation (used for retries).
    :raises Exception: If the maximum number of generation attempts is exceeded.
    """

    if attempt > args.generation_attempts:
        raise Exception(f"Exceeded generation attempts, unable to generate a scene for rule:\n {items}")

    placement_radius = args.placement_radius
    anchor_position = args.anchor_position
    collision_margin = args.collision_margin
    touching_margin = args.touching_margin
    placement_attempts = args.placement_attempts
    los_threshold = args.los_threshold

    # items = prolog_string
    instructions = []
    for item in items:
        match = re.match(r"item\((\d+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(.+)\)", item)
        if match:
            item_id = int(match.group(1))
            color = match.group(2)
            shape = match.group(3)
            orientation = match.group(4)
            action = match.group(5)
            instructions.append({
                'id': item_id,
                'color': color,
                'shape': shape,
                'orientation': orientation,
                'action': action
            })

    related_objects = get_relations(instructions)
    if len(related_objects) != len(instructions):
        raise Exception(f"Rule not resolvable!\n {instructions}\n {related_objects}")

    integrity = True

    for instruction in related_objects:
        idx = instructions.index(instruction)
        current_object = generate_creation(args, instructions, idx, collection)

        # Random 90° rotations around Z (constrained for upside-down wedges).
        if current_object.shape == "wedge" and current_object.pose == "upside_down":
            random_rotation = 90
        else:
            random_rotation = random.choice([0, 90, 180, 270])
        current_object.rotate_z(random_rotation)

        if instruction['action'] == 'grounded':
            attempts = 0
            while True:
                if attempts >= placement_attempts:
                    debug(f"Exceeded maximum placement attempts!")
                    integrity = False
                    break
                if idx == 0:
                    pos = Vector(anchor_position)
                else:
                    pos = get_random_position(anchor=anchor_position, radius=placement_radius)

                current_object.set_position_xy(pos)
                colliding_objects = check_collision(current_object, margin=collision_margin)
                if len(colliding_objects) == 0:
                    break
                else:
                    attempts += 1
                    debug(f"{current_object.get_namestring()} colliding with {[o.get_namestring() for o in colliding_objects]}!")
        else:
            relation_type, target = generate_relation(instruction)
            attempts = 0
            while True:
                if attempts >= placement_attempts:
                    debug(f"Exceeded maximum placement attempts!")
                    integrity = False
                    break

                if relation_type == 'touching':
                    random_faces = args.random_face_choice
                    faces = target.get_free_face()
                    if random_faces and not grounded:
                        face = random.choice(faces)
                    else:
                        face = faces[0]
                    touching(current_object, target, face=face, margin=touching_margin)
                    colliding_objects = check_collision(current_object, target)
                    if len(colliding_objects) == 0:
                        target.set_touching(face, current_object)
                        axis, direction = face_map[face]
                        object_1_face = list(face_map.keys())[list(face_map.values()).index((axis, direction * (-1)))]
                        current_object.set_touching(object_1_face, target)
                        break
                    else:
                        attempts += 1
                        debug(f"{current_object.get_namestring()} colliding with {[o.get_namestring() for o in colliding_objects]}!")

                elif relation_type == 'pointing':
                    pos = get_random_position(anchor=anchor_position, radius=placement_radius)
                    current_object.set_position_xy(pos)
                    pointing(current_object, target)
                    colliding_objects = check_collision(current_object, margin=collision_margin)
                    pointing_objects = check_pointing(current_object)
                    if len(colliding_objects) == 0 and len(pointing_objects) == 1:
                        break
                    else:
                        attempts += 1
                        debug(f"{current_object.get_namestring()} pointing towards {[o.get_namestring() for o in pointing_objects]}!")

                elif relation_type == 'on_top_of':
                    on_top(current_object, target, margin=touching_margin)
                    break

    # ── Integrity check ───────────────────────────────────────────────────
    for instruction in related_objects:
        current_object = zendo_objects.get_object(instruction['id'])
        if current_object.pose == 'upright' or current_object.pose == 'upside_down':
            continue
        pointing_objects = check_pointing(current_object)
        colliding_objects = check_collision(current_object)
        debug(
            f"{current_object.get_namestring()} pointing towards {[o.get_namestring() for o in pointing_objects]}!\n"
            f"{current_object.get_namestring()} colliding with {[o.get_namestring() for o in colliding_objects]}!"
        )

        if instruction['action'].split('(')[0] == 'pointing':
            if len(pointing_objects) != 1:
                integrity = False
        else:
            if len(pointing_objects) != 0:
                integrity = False

    if check_scene_occlusion(los_threshold):
        debug("Scene is occluded, trying to generate again!")
        integrity = False

    debug(f"Integrity: {integrity}")

    if not integrity:
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        ZendoObject.instances.clear()

        generate_structure(args, items, collection, attempt=attempt + 1, grounded=grounded)


def check_scene_occlusion(threshold):
    """Return True if any object in the scene is more than ``threshold`` fraction occluded from the camera."""
    cam = bpy.data.objects["Camera.001"]
    camera_loc = cam.location
    targets = [
        o for o in bpy.data.objects
        if o.type == 'MESH' and any(k in o.name for k in ["Pyramid", "Wedge", "Block"])
    ]

    for obj in targets:
        vertices = obj.data.vertices
        if not vertices:
            continue

        mw = obj.matrix_world
        total = len(vertices)
        blocked = 0

        for v in vertices:
            wv = mw @ v.co
            dir_vec = wv - camera_loc
            hit, hit_loc, _, _, hit_obj, _ = bpy.context.scene.ray_cast(
                bpy.context.view_layer.depsgraph, camera_loc, dir_vec
            )

            if (
                    hit
                    and hit_obj != obj
                    and "Ground" not in hit_obj.name
                    and (hit_loc - camera_loc).length < dir_vec.length - 1e-5
            ):
                aligned = (
                    abs(hit_obj.location.x - obj.location.x) < 0.1 and
                    abs(hit_obj.location.y - obj.location.y) < 0.1
                )

                obj_above = obj.location.z > hit_obj.location.z
                hit_above = hit_obj.location.z > obj.location.z

                # Allowed stacks: pyramid-on-block, or two upside-down pieces resting on each other.
                if aligned:
                    if (((obj_above and "Pyramid" in hit_obj.name) or (hit_above and "Pyramid" in obj.name))
                        or (obj_above and "Block" in hit_obj.name and "upside_down" in hit_obj.name and "Pyramid" in obj.name and "upside_down" in obj.name)
                        or (hit_above and "Block" in obj.name and "upside_down" in obj.name and "Pyramid" in hit_obj.name and "upside_down" in hit_obj.name)):
                        continue

                blocked += 1

        debug(f"Actual threshold {obj.name}: {blocked / total:.2f}")
        if blocked / total >= threshold:
            return True
    return False

def compute_3d_bbox_corners(min_bb, max_bb):
    return [
        mathutils.Vector((x, y, z))
        for x in [min_bb.x, max_bb.x]
        for y in [min_bb.y, max_bb.y]
        for z in [min_bb.z, max_bb.z]
    ]

def project_point_3d_to_2d(point, cam, scene):
    co_ndc = world_to_camera_view(scene, cam, point)
    render = scene.render
    x = round(co_ndc.x * render.resolution_x)
    y = round((1.0 - co_ndc.y) * render.resolution_y)
    return x, y

def get_image_bounding_box(obj, scene):
    bpy.context.view_layer.update()
    cam = bpy.data.objects["Camera.001"]
    min_bb, max_bb = obj.get_world_bounding_box()
    corners = compute_3d_bbox_corners(min_bb, max_bb)
    projected = [project_point_3d_to_2d(corner, cam, scene) for corner in corners]
    xs, ys = zip(*projected)
    x_min = max(0, min(xs))
    x_max = min(scene.render.resolution_x - 1, max(xs))
    y_min = max(0, min(ys))
    y_max = min(scene.render.resolution_y - 1, max(ys))
    return x_min, y_min, x_max, y_max