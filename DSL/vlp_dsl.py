import re
from type_system import *
from program import *

import torch
import math
import ast
from functools import reduce

t0 = PolymorphicType("t0")
t1 = PolymorphicType("t1")


def parse_number(response):
    # print(response)
    try:
        response = response.split("NUMBER")[-1]
        response = response.split(":")[-1]
        response = response.split(".")[0]
        # print(response)
        number = int(response)
    except:
        # raise ValueError()
        return 0

    return number


def parse_bool(response):
    response = response.strip().upper()
    if "YES" in response:
        return True
    elif "NO" in response:
        return False
    elif "ONCE" in response:
        return True
    else:
        print("!! Invalid response from model. Expected 'YES' or 'NO'.")
        return False


def parse_incomplete_list(raw):
    """Parses an incomplete list like
    objects = [['car', 'red'], ['tree', 'tall'], ['person', 'standing', 'small'],
    """
    lines = raw.strip().splitlines()
    parsed_objects = []

    for line in lines:
        line = line.strip()
        if line.startswith("objects"):
            continue  # skip the assignment line
        if line.startswith("["):
            try:
                # Try parsing line as a literal list
                parsed = ast.literal_eval(line.rstrip(","))
                if isinstance(parsed, list):
                    parsed_objects.append(parsed)
            except (SyntaxError, ValueError):
                # Reached the cutoff or malformed line
                break

    return parsed_objects


def parse_incomplete_flat_list(raw):
    """Parses an incomplete flat list like
    objects = ['car', 'red', 'tree', 'tall', 'person', 'standing', 'small']
    """
    list_pattern = re.compile(r"\[.*?\]")  # non-greedy match for each complete [...]
    matches = list_pattern.findall(raw)
    parsed_objects = []

    for match in matches:
        try:
            parsed = ast.literal_eval(match)
            if isinstance(parsed, list):
                parsed_objects.append(parsed)
        except (SyntaxError, ValueError):
            continue  # skip malformed sublists

    return parsed_objects


def get_dsl(prompter, variables, seed=42):

    objects = variables.get("objects", [])
    properties = variables.get("properties", [])
    actions = variables.get("actions", [])
    sceneries = variables.get("sceneries", [])

    with torch.no_grad():
        torch.cuda.empty_cache()

    def _flatten(l):
        return [x for xs in l for x in xs]

    def _eq(x):
        return lambda y: x == y

    def _gt(x):
        return lambda y: x > y

    def __unfold(p, f, n, x, recursion_limit=20):
        if recursion_limit <= 0:
            raise ValueError
        if p(x):
            return []
        return [f(x)] + __unfold(p, f, n, n(x), recursion_limit - 1)

    # ── Helper for substring matching ────────────────────────────────

    def _appears_in(predicate, strings):
        """Check if predicate appears as a substring in any of the strings."""
        return any(predicate in s for s in strings)

    # ── List-based count functions (operate on output of get_objects) ─

    def _count_object(img_representation):
        """Count entries that contain the given object string anywhere in the entry."""
        return lambda obj: __count_object(img_representation, obj)

    def __count_object(img_representation, obj):
        if isinstance(img_representation, list):
            return sum(
                1 for entry in img_representation
                if isinstance(entry, list) and _appears_in(obj, entry)
            )
        return 0

    def _count_property(img_representation):
        """Count entries that contain the given property in positions [1:]."""
        return lambda prop: __count_property(img_representation, prop)

    def __count_property(img_representation, prop):
        return sum(
            1 for entry in img_representation
            if isinstance(entry, list) and len(entry) > 1 and _appears_in(prop, entry[1:])
        )

    def _count_object_with_property(img_representation):
        """Count entries where entry[0] matches obj and prop appears in entry[1:]."""
        return lambda obj: lambda prop: __count_object_with_property(img_representation, obj, prop)

    def __count_object_with_property(img_representation, obj, prop):
        return sum(
            1 for entry in img_representation
            if isinstance(entry, list) and len(entry) > 1 and entry[0] == obj and prop in entry[1:]
        )

    def _count_properties(img_representation):
        """Count entries where both prop1 and prop2 appear together in entry[1:]."""
        return lambda prop1: lambda prop2: __count_properties(img_representation, prop1, prop2)

    def __count_properties(img_representation, prop1, prop2):
        return sum(
            1 for entry in img_representation
            if isinstance(entry, list) and len(entry) > 1
            and prop1 in entry[1:] and prop2 in entry[1:]
        )

    def _count_all_objects(img_representation):
        return len(img_representation) if isinstance(img_representation, list) else 0


    # ── Counting versions of action/interaction predicates ────────────

    def _count_property_with_action_with_property(img_representation):
        return lambda prop1: lambda action: lambda prop2: __count_property_with_action_with_object(img_representation, prop1, action, prop2)

    def _count_object_with_action_with_property(img_representation):
        return lambda obj: lambda action: lambda prop: __count_property_with_action_with_object(img_representation, obj, action, prop)

    def _count_object_with_action_with_object(img_representation):
        return lambda prop: lambda action: lambda obj: __count_property_with_action_with_object(img_representation, prop, action, obj)

    def _count_property_with_action_with_object(img_representation):
        return lambda obj1: lambda action: lambda obj2: __count_property_with_action_with_object(img_representation, obj1, action, obj2)

    def __count_property_with_action_with_object(img_representation, prop, action, obj):
        """Count objects with property prop that are performing action on objects of type obj."""
        if not prop or not action or not obj or prop == "None" or action == "None" or obj == "None":
            return 0
        count = 0
        for o in img_representation:
            if isinstance(o, list) and len(o) == 3:
                if prop == o[0] and action == o[1] and obj == o[2]:
                    count += 1
        return count

    # ─────────────────────────────────────────────────────────────────

    def _objects_from_img(img):
        """
        Obtain a list of objects and their properties from the image.
        Each inner list contains the object name followed by its properties as strings.
        """
        return __obtain_objects_from_img(img, objects, properties)

    def __obtain_objects_from_img(img, objects, properties):

        prompt = f"""
        ## Task
        Identify objects and their properties from the image using only the provided lists.

        **Objects:** {objects}
        **Properties:** {properties}

        ## Rules
        1. Only use objects/properties from the provided lists
        2. Return empty list if no valid objects found
        3. No explanations or additional text

        ## Output Format
        ```python
        objects = [
            ['object_name', 'property1', 'property2', ...],
            ['object_name', 'property1'],
            ...
        ]
        ```

        **If no valid objects:** `objects = [[]]`

        ## Examples

        **Example 1**
        - Objects: ["car", "person", "tree"]
        - Properties: ["red", "tall", "small", "standing"]
        - Image: Red car under tall tree with small standing person

        ```python
        objects = [
            ['car', 'red'],
            ['tree', 'tall'],
            ['person', 'standing', 'small']
        ]
        ```

        **Example 2**
        - Objects: ["dog", "ball", "book", "chair"]
        - Properties: ["blue", "sitting", "round"]
        - Image: Dog sitting by round ball and blue chair

        ```python
        objects = [
            ['dog', 'sitting'],
            ['ball', 'round'],
            ['chair', 'blue']
        ]
        ```

        **Example 3**
        - Objects: ["bicycle", "lamp", "table", "cup"]
        - Properties: ["green", "broken", "wooden", "white"]
        - Image: Table with laptop and cup

        ```python
        objects = [[]]
        ```
        *Note: Even though 'table' and 'cup' are in the objects list and visible in the image, neither has properties from the provided list, so no valid object-property combinations exist*

        **Analyze the image now:**
        """

        response = prompter.prompt_with_images(
            prompt_text=prompt, paths=[img], max_new_tokens=200, seed=seed
        )
        org_response = response

        # Parse the response to extract the objects and their properties
        try:
            # remove \n from response
            identifier = "objects =" if "objects =" in response else "object ="
            response = response.replace("\n", "")
            response = response.split(identifier)[-1]
            response = response.split("```")[0]
            object_list = ast.literal_eval(response)

        except Exception as e:
            # raise ValueError(f"Failed to parse the response: {e}")
            # print(f"Failed to parse the response: {org_response} - {e}")
            try:
                object_list_1 = parse_incomplete_list(response)
                object_list_2 = parse_incomplete_flat_list(response)

                # select the longest list
                if len(object_list_1) >= len(object_list_2):
                    # print("Successfully parsed object list 1.")
                    object_list = object_list_1
                else:
                    # print("Successfully parsed object list 2.")
                    object_list = object_list_2

                # print(object_list)

            except Exception as e2:
                try:
                    object_list = parse_incomplete_flat_list(response)
                except Exception as e3:
                    # If all parsing attempts fail, return an empty list
                    print(f"Failed to parse the response: {org_response} - {e3}")
                    object_list = [[]]

        if type(object_list) == list and len(object_list) > 0:
            if type(object_list[0]) == str:
                print(f"Turning action list {object_list} into a nested list.")
                object_list = [object_list]
                print(object_list)

        if type(object_list) != list:
            object_list = [[]]
        elif len(object_list) > 0 and type(object_list[0]) != list:
            object_list = [[]]

        return object_list

    def _actions_from_img(img):
        """
        Obtain a list of interaction triples from the image.
        Each inner list has the format [subject, action, object].
        """
        return __obtain_actions_from_img(img, objects, properties, actions)

    def __obtain_actions_from_img(img, objects, properties, actions):

        prompt = f"""
        ## Task
        Identify interaction triples in the image using only the provided lists.
        Each triple has the format [subject, action, object] where:
        - subject: the entity performing the action (from Objects or Properties)
        - action: what is happening between them (from Actions)
        - object: the entity the action is directed at (from Objects or Properties)

        **Objects:** {objects}
        **Properties:** {properties}
        **Actions:** {actions}

        ## Rules
        1. Only use values from the provided lists for each position
        2. Each triple must have exactly 3 elements: [subject, action, object]
        3. If the same interaction occurs multiple times in the image, include one entry per occurrence
        4. Return empty list if no valid triples found
        5. No explanations or additional text

        ## Output Format
        ```python
        actions = [
            ['subject1', 'action1', 'object1'],
            ['subject2', 'action2', 'object2'],
            ...
        ]
        ```

        **If no valid interactions:** `actions = [[]]`

        ## Examples

        **Example 1**
        - Objects: ["block", "pyramid", "wedge"]
        - Properties: ["blue", "red"]
        - Actions: ["touching", "grounded"]
        - Image: Two blue blocks each touching a pyramid

        ```python
        actions = [
            ['block', 'touching', 'pyramid'],
            ['block', 'touching', 'pyramid']
        ]
        ```
        *Note: The interaction occurs twice, so it appears twice in the list*

        **Example 2**
        - Objects: ["block", "wedge"]
        - Properties: ["blue", "red", "upright"]
        - Actions: ["touching", "supporting"]
        - Image: Blue block touching red wedge, wedge supporting a block

        ```python
        actions = [
            ['blue', 'touching', 'red'],
            ['wedge', 'supporting', 'block']
        ]
        ```

        **Analyze the image now:**
        """

        response = prompter.prompt_with_images(
            prompt_text=prompt, paths=[img], max_new_tokens=200, seed=seed
        )
        org_response = response
        # print(response)
        # Parse the response to extract the objects and their properties
        try:
            identifier = "actions =" if "actions =" in response else "action ="
            # remove \n from response
            response = response.replace("\n", "")
            response = response.split(identifier)[-1]
            response = response.split("```")[0]
            action_list = ast.literal_eval(response)

        except Exception as e:
            # raise ValueError(f"Failed to parse the response: {e}")
            print(f"Failed to parse the response: {org_response} - {e}")

            try:
                action_list_1 = parse_incomplete_list(response)
                action_list_2 = parse_incomplete_flat_list(response)

                # select the longest list
                if len(action_list_1) >= len(action_list_2):
                    print("Successfully parsed action list 1.")
                    action_list = action_list_1
                else:
                    print("Successfully parsed action list 2.")
                    action_list = action_list_2
            except Exception as e2:
                action_list = [[]]

        if type(action_list) == list and len(action_list) > 0:
            if type(action_list[0]) == str:
                print(f"Turning action list {action_list} into a nested list.")
                action_list = [action_list]
                print(action_list)

        return action_list

    #### ----------------------------------------------------------------- ####
    #### Done with the DSL functions, now we can create the DSL. ####

    semantics = {
        "get_objects": _objects_from_img,
        "get_actions": _actions_from_img,
        "count_object": _count_object,
        "count_property": _count_property,
        "count_object_with_property": _count_object_with_property,
        "count_properties": _count_properties,
        "count_all_objects": _count_all_objects,
        "count_property_with_action_with_property": _count_property_with_action_with_property,
        "count_object_with_action_with_property": _count_object_with_action_with_property,
        "count_property_with_action_with_object": _count_property_with_action_with_object,
        "count_object_with_action_with_object": _count_object_with_action_with_object,
        "gt?": _gt,
        "eq?": _eq,
        "odd?": lambda n: n % 2 == 1 and n != 0,
        "even?": lambda n: n % 2 == 0 and n != 0,
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "and": lambda bool1: lambda bool2: bool1 and bool2,
        "or": lambda bool1: lambda bool2: bool1 or bool2,
    }

    primitive_types = {
        "get_objects": Arrow(IMG, List(List(STRING))),
        "get_actions": Arrow(IMG, List(List(STRING))),
        "count_object": Arrow(List(List(STRING)), Arrow(OBJECT, INT)),
        "count_property": Arrow(List(List(STRING)), Arrow(PROPERTY, INT)),
        "count_object_with_property": Arrow(
            List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, INT))
        ),
        "count_properties": Arrow(
            List(List(STRING)), Arrow(PROPERTY, Arrow(PROPERTY, INT))
        ),
        "count_all_objects": Arrow(List(List(STRING)), INT),
        "count_property_with_action_with_property": Arrow(
            List(List(STRING)), Arrow(PROPERTY, Arrow(ACTION, Arrow(PROPERTY, INT)))
        ),
        "count_object_with_action_with_property": Arrow(
            List(List(STRING)), Arrow(OBJECT, Arrow(ACTION, Arrow(PROPERTY, INT)))
        ),
        "count_property_with_action_with_object": Arrow(
            List(List(STRING)), Arrow(PROPERTY, Arrow(ACTION, Arrow(OBJECT, INT)))
        ),
        "count_object_with_action_with_object": Arrow(
            List(List(STRING)), Arrow(OBJECT, Arrow(ACTION, Arrow(OBJECT, INT)))
        ),
        "gt?": Arrow(INT, Arrow(INT, BOOL)),
        "eq?": Arrow(INT, Arrow(INT, BOOL)),
        "odd?": Arrow(INT, BOOL),
        "even?": Arrow(INT, BOOL),
        "0": INT,
        "1": INT,
        "2": INT,
        "3": INT,
        "4": INT,
        "5": INT,
        "6": INT,
        "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
        "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
    }

    return semantics, primitive_types
