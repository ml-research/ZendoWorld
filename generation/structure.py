# This file contains code derived from:
# - https://github.com/CapArrow/zendo_game_dataset_generator
import bpy
import mathutils
from mathutils import Vector
from generation.zendo_objects import ZendoObject, Pyramid, Block, Wedge
import math
import copy
import random

face_map = {
    "front": ('X', 1),
    "back": ('X', -1),
    "right": ('Y', 1),
    "left": ('Y', -1),
    "top": ('Z', 1),
    "bottom": ('Z', -1),
}


def check_beneath(object: ZendoObject):
    """Return Blender objects whose top is below ``object``'s bottom (i.e. directly beneath)."""
    beneath_objects = []

    target_min, target_max = object.get_world_bounding_box()

    for obj in bpy.data.objects:
        if obj == object.obj:
            continue

        obj_min, obj_max = obj.calculate_world_bounding_box()

        if obj_max.z <= target_min.z:
            beneath_objects.append(obj)

    return beneath_objects


def on_top(object_1: ZendoObject, target: ZendoObject, margin: float = 0.0):
    """Place ``object_1`` on top of ``target``; pyramid pairs in matching poses are nested instead."""
    bpy.context.view_layer.update()
    if type(target) is Pyramid and object_1.pose == 'upright' and target.pose == 'upright' or \
        type(object_1) is Pyramid and object_1.pose == 'upside_down' and target.pose == 'upside_down':
        nested(object_1, target)
    else:
        touching(object_1, target, face='top', margin=margin)


def get_restricted_bounds(obj_a, obj_b, direction):
    """Min/max of ``obj_a`` along ``direction`` (x or y) restricted to verts inside ``obj_b``'s footprint."""
    direction = direction.lower()
    if direction not in ('x', 'y'):
        print(f"Invalid direction: {direction}.")
        raise ValueError("Direction must be 'x' or 'y'.")

    axis_idx = {'x': 0, 'y': 1}
    move_axis = axis_idx[direction]
    filter_axis = 1 - move_axis

    verts_a = [obj_a.matrix_world @ v.co for v in obj_a.data.vertices]
    verts_b = [obj_b.matrix_world @ v.co for v in obj_b.data.vertices]

    b_min = min(v[filter_axis] for v in verts_b)
    b_max = max(v[filter_axis] for v in verts_b)

    c_min = min(v[2] for v in verts_b)
    c_max = max(v[2] for v in verts_b)

    filtered = [v for v in verts_a if b_min <= v[filter_axis] <= b_max and c_min <= v[2] <= c_max]

    if not filtered:
        return None, None

    min_a = min(v[move_axis] for v in filtered)
    max_a = max(v[move_axis] for v in filtered)
    return min_a, max_a


def touching(object_1: ZendoObject, object_2: ZendoObject, face: str = 'left', margin: float = 0.0):
    """Place ``object_1`` flush against ``object_2``'s ``face``; raises if face is invalid or occupied."""
    bpy.context.view_layer.update()
    if face not in face_map:
        raise ValueError(
            f"{face} is not a valid face! "
            f"Valid faces are: {[f for f in face_map]}"
        )

    if object_2.get_touching()[face] is not None:
        raise ValueError(
            f"{face} of {object_2.name} is already occupied!"
        )

    axis, direction = face_map.get(face.lower(), None)
    loc_object_2 = object_2.get_position()
    object_1.set_position(Vector((loc_object_2[0], loc_object_2[1], object_1.get_position()[2])))

    obj1_min, obj1_max = object_1.get_world_bounding_box()
    obj2_min, obj2_max = object_2.get_world_bounding_box()

    axis_index = 'XYZ'.index(axis.upper())

    if axis_index == 2:
        if direction > 0:
            offset = obj2_max[axis_index] - obj1_min[axis_index]
            offset += margin
        else:
            offset = obj2_min[axis_index] - obj1_max[axis_index]
            offset -= margin
        object_1.obj.location[axis_index] += offset
    else:
        min_1, max_1 = get_restricted_bounds(object_1.obj, object_2.obj, axis)
        min_2, max_2 = get_restricted_bounds(object_2.obj, object_1.obj, axis)
        if direction > 0:
            offset = max_2 - min_1
            offset += margin
        else:
            offset = min_2 - max_1
            offset -= margin
        # Empirical scaling: certain pose/shape/rotation combinations need a smaller offset
        # to avoid visible gaps when sloped faces meet.
        if ((object_1.pose == 'upside_down' and object_2.pose == 'upright') or (object_2.pose == 'upside_down' and object_1.pose == 'upright')) and object_2.shape == "pyramid" and object_1.shape == "pyramid":
            offset = offset * 0.54
        if (object_1.pose == 'upside_down' and object_2.pose == 'upright' and object_1.shape == 'wedge' and object_2.shape == "pyramid" and any(abs(object_1.get_rotation_z_degrees() - target) < 20 for target in [90, 270])) or (object_2.pose == 'upside_down' and object_1.pose == 'upright' and object_2.shape == 'wedge' and object_1.shape == "pyramid" and any(abs(object_2.get_rotation_z_degrees() - target) < 20 for target in [90, 270])):
            offset = offset * 0.54
        if (object_1.pose == 'upside_down' and object_2.pose == 'upright' and object_1.shape == 'pyramid' and object_2.shape == "wedge"  and any(abs(object_2.get_rotation_z_degrees() - target) < 20 for target in [90, 270])) or (object_2.pose == 'upside_down' and object_1.pose == 'upright' and object_2.shape == 'pyramid' and object_1.shape == "wedge" and any(abs(object_1.get_rotation_z_degrees() - target) < 20 for target in [90, 270])):
            offset = offset * 0.54
        if (((object_1.pose == 'upside_down' and object_2.pose == 'upright') or (object_1.pose == 'upright' and object_2.pose == 'upside_down')) and object_1.shape == 'wedge' and object_2.shape == "wedge"  and any(abs(object_2.get_rotation_z_degrees() - target) < 20 for target in [90, 270]) and any(abs(object_1.get_rotation_z_degrees() - target) < 20 for target in [90, 270])):
            offset = offset * 0.54
        object_1.obj.location[axis_index] += offset
        if offset is None:
            raise ValueError(
                f"Offset could not be computed for {object_1.obj.name} and {object_2.obj.name} "
            )

    if face == "top":
        object_1.grounded = False
        object_1.set_touching("bottom", object_2.obj)
        object_2.set_touching("top", object_1.obj)


def nested(object_1: ZendoObject, object_2: ZendoObject):
    """Nest ``object_1`` inside ``object_2`` (used for pyramid pairs); applies a 0.4 offset to avoid clipping."""
    if object_2.shape == "pyramid" and object_2.pose == "upright":
        bpy.context.view_layer.update()
        obj_2_pos = object_2.get_position()
        object_1.set_position(obj_2_pos)

        obj_2_rot = object_2.obj.rotation_quaternion
        object_1.set_rotation_quaternion(obj_2_rot)

        mesh = object_2.obj.data
        top_vertex = max(mesh.vertices, key=lambda v: v.co.z)
        top_world = object_2.obj.matrix_world @ top_vertex.co
        origin_world = object_2.obj.matrix_world @ mathutils.Vector((0, 0, 0))

        vector_to_top = top_world - origin_world

        scaled_vector = vector_to_top * 0.4
        object_1.move(scaled_vector)
    else:
        bpy.context.view_layer.update()
        obj_2_pos = object_2.get_position()
        object_1.set_position(obj_2_pos)

        obj_2_rot = object_2.obj.rotation_quaternion
        object_1.set_rotation_quaternion(obj_2_rot)

        mesh = object_1.obj.data
        top_vertex = max(mesh.vertices, key=lambda v: v.co.z)
        top_world = object_2.obj.matrix_world @ top_vertex.co
        origin_world = object_2.obj.matrix_world @ mathutils.Vector((0, 0, 0))
        vector_to_top = top_world - origin_world
        scaled_vector = -vector_to_top * 0.4
        object_1.move(scaled_vector)
    object_2.nested = object_1.obj.name
    object_1.nests = object_2.obj.name
    object_2.set_touching("top", object_1.obj)
    object_1.set_touching("bottom", object_2.obj)


def weird(object_1: ZendoObject, object_2: ZendoObject, face: str):
    # Placeholder for future "weird" interaction handling.
    pass


def pointing(object_1: ZendoObject, target: ZendoObject):
    """Rotate ``object_1`` around Z so its forward rays point at ``target``."""
    bpy.context.view_layer.update()

    origin = object_1.obj.matrix_world.translation

    rays = object_1.get_rays()
    avg_origin = Vector((0, 0, 0))
    avg_direction = Vector((0, 0, 0))
    for ray_origin, ray_direction in rays:
        avg_origin += ray_origin
        avg_direction += ray_direction

    avg_origin = avg_origin / len(rays)
    avg_direction = avg_direction / len(rays)

    target_position = copy.deepcopy(target.obj.matrix_world.translation)
    target_direction_xy = mathutils.Vector((target_position.x - origin.x,
                                            target_position.y - origin.y,
                                            0)).normalized()

    rotation_angle = avg_direction.angle(target_direction_xy)

    # Sign by cross-product Z to pick rotation direction.
    cross_z = avg_direction.cross(target_direction_xy).z
    if cross_z < 0:
        rotation_angle = -rotation_angle

    rotation_quaternion = mathutils.Quaternion(Vector((0, 0, 1)), rotation_angle)

    # Compose with existing rotation rather than replace it.
    object_1.obj.rotation_mode = 'QUATERNION'
    object_1.obj.rotation_quaternion = rotation_quaternion @ object_1.obj.rotation_quaternion

    object_1.pointing = target.obj.name
    bpy.context.view_layer.update()
