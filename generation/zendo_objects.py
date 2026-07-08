# This file contains code derived from:
# - https://github.com/CapArrow/zendo_game_dataset_generator
import sys
import os
from enum import Enum
import math
import bpy
import mathutils
from mathutils import Vector, Euler, Quaternion
import numpy as np
import generation.utils as utils
from generation.utils import debug
from dataclasses import dataclass


class ZendoObject:
    instances = []
    poses = {}

    def __init__(self, args, idx: int, shape: str, color: str, pose: str = "upright"):
        """
        Initializes a ZendoObject instance by loading a Blender object with specific shape, color, and pose.

        :param args: Configuration arguments including paths to resources.
        :param idx: Unique identifier for the object.
        :param shape: Shape type of the object (e.g., "Pyramid", "Block", "Wedge").
        :param color: Color name for the object, corresponding to defined color mappings.
        :param pose: Initial pose orientation ("upright" by default).
        """

        ZendoObject.instances.append(self)
        object_shapes, object_colors, object_sizes = utils.read_properties_json(args.properties_json)
        color_code = object_colors[color]
        filename = os.path.join(args.shape_dir, '%s.blend' % shape.lower(), 'Object', shape)
        bpy.ops.wm.append(filename=filename)
        unique_name = f"{idx}_{shape}_{color}_{pose}"
        bpy.data.objects[shape].name = unique_name
        self.obj = bpy.data.objects[unique_name]
        self.material = self.obj.data.materials[0]

        self.args = args
        self.shape = shape.lower()
        self.idx = idx
        self.name = unique_name
        self.pose = pose
        self.set_color(color_code)
        self.color = color
        self.set_pose(pose)
        self.grounded = True
        self.touching = {
            'left': None,
            'right': None,
            'front': None,
            'back': None,
            'top': None,
            'bottom': None,
        }
        self.nests = None
        self.pointing = None

        self.rays = []

    def remove(self):
        """
        Removes the Blender object associated with this ZendoObject instance from the scene and clears its instance reference.
        """

        bpy.data.objects.remove(self.obj, do_unlink=True)
        ZendoObject.instances.remove(self)

    def get_namestring(self):
        """
        Returns a string representation of the object's color, shape, and pose.

        :return: A formatted string with color, shape, and pose.
        """

        return f"{self.color} {self.shape} {self.pose}"

    def set_touching(self, face: str, obj):
        """
        Sets the reference of another object touching this object on a specified face.

        :param face: The face of this object being touched ('left', 'right', 'front', 'back', 'top', 'bottom').
        :param obj: The ZendoObject instance that is touching this object.
        """

        if obj is None:
            self.touching[face] = obj
        else:
            self.touching[face] = obj.name

    def get_touching(self):
        """
        Retrieves the dictionary of objects currently touching this object, keyed by face.

        :return: A dictionary mapping each face ('left', 'right', 'front', 'back', 'top', 'bottom') to the touching ZendoObject or None.
        """

        return self.touching

    def set_pose(self, pose):
        """
        Sets the pose of the object using a predefined orientation quaternion. Also ensures the object is grounded after the rotation.

        :param pose: The pose orientation to set ('upright', 'flat', etc.).
        """

        self.check_pose(pose)
        self.obj.rotation_mode = "QUATERNION"
        self.obj.rotation_quaternion = self.__class__.poses[pose]
        self.pose = pose
        self.set_to_ground()
        bpy.context.view_layer.update()

    def set_position(self, position: Vector):
        """
        Sets the object's position in 3D space.

        :param position: A Vector representing the new position.
        """

        self.obj.location = position

    def set_position_xy(self, position: Vector):
        """
        Sets only the X and Y coordinates of the object's position, leaving the Z coordinate unchanged.

        :param position: A Vector containing the new X and Y coordinates.
        """

        self.obj.location.x = position.x
        self.obj.location.y = position.y

    def get_position(self):
        """
        Returns the current position of the object in world coordinates.

        :return: A Vector representing the object's location.
        """
        bpy.context.view_layer.update()
        return self.obj.location

    def get_scaled_and_rotated_vertices(self):
        """
        Computes and returns the object's vertices after applying scaling and rotation transformations, without translation.

        :return: A list of vectors representing the object's vertices in local space after scale and rotation.
        """

        return [self.obj.rotation_quaternion @ (self.obj.scale * vertex.co) for vertex in self.obj.data.vertices]

    def set_to_ground(self):
        """
        Positions the object vertically so that its lowest vertex touches the ground plane (Z=0).

        :return: The vertical offset applied to ground the object.
        """

        self.obj.location = mathutils.Vector((0.0, 0.0, 0.0))
        transformed_vertices = self.get_scaled_and_rotated_vertices()
        bottom_p = min(transformed_vertices, key=lambda p: p.z)
        center_z = self.obj.location.z
        ground_ofset = center_z - bottom_p.z
        self.obj.location.z = ground_ofset
        return ground_ofset

    def check_pose(self, pose):
        """
        Validates if the given pose is available for this object type.

        :param pose: The pose orientation to validate.
        :raises ValueError: If the pose is not valid for the object's class.
        """

        if pose not in self.__class__.poses:
            raise ValueError(
                f"{pose} is not a valid pose for {self.__class__.__name__}. "
                f"Valid poses are: {[p for p in self.__class__.poses]}"
            )

    def get_free_face(self):
        """
        Determines and returns a list of faces ('left', 'right', 'front', 'back') of the object that are currently not touching other objects.

        :return: A list of strings representing the available (free) faces.
        """

        avail = [f for f in self.touching if self.touching.get(f) is None and f not in ('top', 'bottom')]
        return avail

    def rotate(self, axis: str, rad: float):
        """
        Rotates the object around a specified axis by the given angle (in degrees). Ensures the object remains grounded after rotation.

        :param axis: The axis ('X', 'Y', or 'Z') around which to rotate.
        :param rad: The rotation angle in degrees.
        :raises ValueError: If the specified axis is invalid.
        """

        self.obj.rotation_mode = "QUATERNION"
        if axis.upper() == 'X':
            rotation = Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(rad))
        elif axis.upper() == 'Y':
            rotation = Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(rad))
        elif axis.upper() == 'Z':
            rotation = Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(rad))
        else:
            raise ValueError(
                f"{axis} is an invalid axis!"
            )
        self.obj.rotation_quaternion = rotation @ self.obj.rotation_quaternion
        self.set_to_ground()
        bpy.context.view_layer.update()

    def set_rotation_quaternion(self, rotation: Quaternion):
        """
        Sets the object's rotation using the specified quaternion.

        :param rotation: A Quaternion representing the desired rotation.
        """

        self.obj.rotation_mode = "QUATERNION"
        self.obj.rotation_quaternion = rotation

    def rotate_z(self, rad: float):
        """
        Rotates the object around the Z-axis by a specified angle (in degrees), ensuring it remains grounded afterward.

        :param rad: The rotation angle in degrees.
        """

        self.obj.rotation_mode = "QUATERNION"
        rotation = Quaternion(Vector((0, 0, 1)), math.radians(rad))
        self.obj.rotation_quaternion = rotation @ self.obj.rotation_quaternion
        self.set_to_ground()
        bpy.context.view_layer.update()

    def get_rotation_z_degrees(self) -> float:
        """
        Returns the rotation of the object around the Z-axis in degrees, normalized to [0, 360).
        Assumes rotation_mode is QUATERNION.
        """
        quat = self.obj.rotation_quaternion
        euler = quat.to_euler('XYZ')
        return math.degrees(euler.z) % 360

    def move(self, vec: Vector):
        """
        Moves the object by the specified offset vector.

        :param vec: A Vector representing the translation offset to apply to the object's position.
        """

        self.obj.location += vec

    def set_color(self, color: list[float]):
        """
        Sets the object's color by adjusting its material's RGBA node value.

        :param color: A list containing RGBA color values (floats between 0 and 1).
        :raises Exception: If the object's material or RGB node is missing.
        """

        r, g, b, a = color
        # Check if the material exists and uses nodes
        if not self.material:
            raise Exception("No material found.")

        if not self.material.use_nodes:
            raise Exception(f"Material '{self.material.name}' does not use nodes.")

        nodes = self.material.node_tree.nodes

        # Look for an RGB node labeled "Color"
        color_node = None
        for node in nodes:
            if node.type == 'RGB':  # and (node.label == "Color" or node.name == "Color"):
                color_node = node
                break

        if not color_node:
            raise Exception("RGB node labeled 'Color' not found.")

        # Set the color
        color_node.outputs[0].default_value = (r, g, b, a)

    def get_world_bounding_box(self):
        """
        Computes the axis-aligned bounding box of the object in world coordinates, accounting for its position, rotation, and scale.

        :return: A tuple (min_coords, max_coords) representing the minimum and maximum bounding box corners as Vectors.
        """

        bpy.context.view_layer.update()

        # Get the world-space coordinates of the object's vertices
        world_vertices = [self.obj.matrix_world @ mathutils.Vector(v.co) for v in self.obj.data.vertices]

        # Find the min and max values along each axis
        min_coords = mathutils.Vector((min(v.x for v in world_vertices),
                                       min(v.y for v in world_vertices),
                                       min(v.z for v in world_vertices)))
        max_coords = mathutils.Vector((max(v.x for v in world_vertices),
                                       max(v.y for v in world_vertices),
                                       max(v.z for v in world_vertices)))

        return min_coords, max_coords
    
    def get_front_facing_direction(self) -> Vector:
        """
        Returns the direction vector that the object is facing in world space.
        Assumes +Y is the local 'front' of the model.
        """
        return self.obj.matrix_world.to_quaternion() @ Vector((0, 1, 0))  # +Y is front


class Pyramid(ZendoObject):
    poses = {
        "upright": Quaternion(Vector((0.0, 0.0, 0.0)), math.radians(0)),
        "upside_down": Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(180)),
        "flat": Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(106))
    }

    def __init__(self, args, idx: int, color: str, pose: str):
        """
        Initializes a Pyramid object by calling the parent ZendoObject initializer with the predefined shape "Pyramid".

        :param args: Configuration arguments including paths to resources.
        :param idx: Unique identifier for this Pyramid object.
        :param color: Color name for the Pyramid.
        :param pose: Initial orientation pose for the Pyramid object.
        """

        super(Pyramid, self).__init__(args, idx, "Pyramid", color, pose)
        self.nested = None

    def get_rays(self):
        """
        Calculates and returns a list of rays originating from the Pyramid's tip, accounting for its current pose and grounding status.

        :return: A list containing tuples with ray origin and direction vectors.
        """

        mesh = self.obj.data
        tip = max(mesh.vertices, key=lambda v: v.co.z)  # Highest vertex in local Z
        if self.pose == 'flat' and self.grounded:
            origin = project_to_xy(self.obj.matrix_world @ tip.co)
            origin.z = 0.01  # Ground offset because raycasting on 0 Z coordinate doesn't work reliably
            direction = project_to_xy(self.obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))).normalized()
        else:
            origin = self.obj.matrix_world @ tip.co
            direction = self.obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1)).normalized()
        return [(origin, direction)]


class Block(ZendoObject):
    poses = {
        "upright": Quaternion(Vector((0.0, 0.0, 0.0)), math.radians(0)),
        "upside_down": Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(180)),
        "flat": Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(90))
    }

    def __init__(self, args, idx: int, color: str, pose: str):
        """
        Initializes a Block object by invoking the parent ZendoObject initializer with the predefined shape "Block".

        :param args: Configuration arguments including paths to resources.
        :param idx: Unique identifier for this Block object.
        :param color: Color name for the Block.
        :param pose: Initial orientation pose for the Block object.
        """

        super(Block, self).__init__(args, idx, "Block", color, pose)

    def get_rays(self):
        """
        Calculates and returns rays originating from the top face of the Block, interpolated across its vertices based on configured density.

        :return: A list of tuples containing ray origin and direction vectors.
        :raises ValueError: If the object does not have enough vertices to interpolate rays.
        """

        mesh = self.obj.data
        highest_z = max(v.co.z for v in mesh.vertices)
        top_vertices = [v for v in mesh.vertices if v.co.z == highest_z]

        if len(top_vertices) < 4:
            raise ValueError("Not enough vertices to interpolate rays.")

        v1_world = self.obj.matrix_world @ top_vertices[0].co
        v2_world = self.obj.matrix_world @ top_vertices[1].co
        v3_world = self.obj.matrix_world @ top_vertices[2].co
        v4_world = self.obj.matrix_world @ top_vertices[3].co

        rays = []

        # Interpolate within the topmost face
        for i in range(self.args.ray_interpolation + 2):  # +2 includes the edges
            t1 = i / (self.args.ray_interpolation + 1)
            # Interpolate along the two opposite edges
            edge1_point = v1_world.lerp(v2_world, t1)
            edge2_point = v4_world.lerp(v3_world, t1)

            for j in range(self.args.ray_interpolation + 2):
                t2 = j / (self.args.ray_interpolation + 1)
                # Interpolate between the two edges to create grid points
                interpolated_point = edge1_point.lerp(edge2_point, t2)

                # Generate ray from the interpolated point
                origin = interpolated_point
                direction = self.obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1)).normalized()

                rays.append((origin, direction))

        for v in top_vertices:
            origin = self.obj.matrix_world @ v.co
            direction = self.obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1)).normalized()
            rays.append((origin, direction))
        return rays


class Wedge(ZendoObject):
    poses = {
        "upright": Quaternion(Vector((0.0, 0.0, 0.0)), math.radians(0)),
        "upside_down": Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(180)),
        "cheesecake": Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(90)),
        "doorstop": Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(104))
    }

    def __init__(self, args, idx: int, color: str, pose: str):
        """
        Initializes a Wedge object by invoking the parent ZendoObject initializer with the predefined shape "Wedge".

        :param args: Configuration arguments including paths to resources.
        :param idx: Unique identifier for this Wedge object.
        :param color: Color name for the Wedge.
        :param pose: Initial orientation pose for the Wedge object.
        """

        super(Wedge, self).__init__(args, idx, "Wedge", color, pose)

    def get_rays(self):
        """
        Computes rays originating from the highest edge of the Wedge, interpolated according to configured density, and accounts for its pose and grounding status.

        :return: A list of tuples containing ray origin and direction vectors.
        :raises ValueError: If the object lacks sufficient vertices for ray interpolation.
        """

        mesh = self.obj.data
        highest_z = max(v.co.z for v in mesh.vertices)
        top_vertices = [v for v in mesh.vertices if v.co.z == highest_z]

        if len(top_vertices) < 2:
            raise ValueError("Not enough vertices to interpolate rays.")

        v1_world = self.obj.matrix_world @ top_vertices[0].co
        v2_world = self.obj.matrix_world @ top_vertices[1].co

        rays = []
        for i in range(self.args.ray_interpolation + 2):  # +2 includes the start and end points
            t = i / (self.args.ray_interpolation + 1)  # Interpolation factor (0 to 1)
            interpolated_point = v1_world.lerp(v2_world, t)  # Linear interpolation

            if self.pose == 'flat' and self.grounded:
                origin = project_to_xy(interpolated_point)
                origin.z = 0.01  # Ground offset because raycasting on 0 Z coordinate doesn't work reliably
                direction = project_to_xy(self.obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))).normalized()
            else:
                origin = interpolated_point
                direction = self.obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1)).normalized()

            rays.append((origin, direction))
        return rays


def get_object_count(name):
    """
    Counts and returns the number of Blender objects in the scene with names starting with the specified prefix.

    :param name: The prefix string to match object names against.
    :return: The count of objects matching the given name prefix.
    """

    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1
    return count


def get_object(idx: int):
    """
    Retrieves the ZendoObject instance with the specified index.

    :param idx: The unique identifier of the desired object.
    :return: The corresponding ZendoObject instance.
    """

    obj = [o for o in ZendoObject.instances if o.idx == idx][0]
    return obj


def project_to_xy(vec):
    """
    Projects a given 3D vector onto the XY plane, removing its Z-component.

    :param vec: The original 3D vector.
    :return: A new vector projected onto the XY plane.
    """

    """Project a 3D vector to the XY plane."""
    return mathutils.Vector((vec.x, vec.y, 0))


def get_from_blender_obj(obj):
    """
    Retrieves the ZendoObject instance associated with the specified Blender object.

    :param obj: The Blender object to find the corresponding ZendoObject for.
    :return: The matching ZendoObject instance, or None if no match is found.
    """

    obj = [o for o in ZendoObject.instances if o.obj is obj]
    if len(obj) > 0:
        return obj[0]
    else:
        return None
