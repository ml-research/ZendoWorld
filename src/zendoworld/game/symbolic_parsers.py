"""Parse item(ID, color, shape, orientation, relation) scenes without model calls.

Accept a list/tuple of item strings, its Python string representation, or one
item string. VLP returns vocabulary-filtered lists; Scientist returns the
ZendoStructure/ZendoPiece objects consumed by its generated Python rules.
"""

import ast
import re

from zendoworld.game.conversions import ZendoPiece, ZendoStructure


_ITEM_RE = re.compile(
    r"^item\(\s*(\d+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(.+)\s*\)$"
)


def _scene_items(scene):
    """Return parsed ``(id, color, shape, orientation, relation)`` tuples."""
    if isinstance(scene, str):
        try:
            scene = ast.literal_eval(scene)
        except (SyntaxError, ValueError):
            scene = [scene]
    if not isinstance(scene, (list, tuple)):
        return []

    parsed = []
    for item in scene:
        if not isinstance(item, str):
            continue
        match = _ITEM_RE.match(item.strip())
        if match is None:
            continue
        item_id, color, shape, orientation, relation = match.groups()
        parsed.append((int(item_id), color.strip(), shape.strip(), orientation.strip(), relation.strip()))
    return sorted(parsed, key=lambda item: item[0])


def _allowed(value, objects, properties):
    if objects is None and properties is None:
        return True
    return value in (set(objects or ()) | set(properties or ()))


def _orientation_properties(orientation):
    """Keep the specific pose and add the orientation groups used by the GM."""
    if orientation in ("doorstop", "cheesecake"):
        return [orientation, "flat"]
    if orientation in ("upright", "upside_down"):
        return [orientation, "vertical"]
    return [orientation]


def parse_vlp_objects(scene, objects=None, properties=None):
    """Parse a scene into VLP's ``get_objects`` list representation."""
    result = []
    for _, color, shape, orientation, relation in _scene_items(scene):
        if objects is not None and shape not in set(objects):
            continue
        values = [shape, color, *_orientation_properties(orientation)]
        if relation == "grounded":
            values.append("grounded")
        if properties is not None:
            values = [values[0]] + [value for value in values[1:] if value in set(properties)]
        result.append(values)
    return result or [[]]


def parse_vlp_actions(scene, objects=None, properties=None, actions=None):
    """Parse pairwise scene relations into VLP's interaction triples."""
    parsed = _scene_items(scene)
    by_id = {item[0]: item for item in parsed}
    result = []

    for source_id, source_color, source_shape, source_orientation, relation in parsed:
        relation_match = re.match(
            r"^(touching|on_top_of|pointing)\(\s*(\d+)\s*\)$", relation
        )
        if relation_match is None:
            continue
        action, target_id = relation_match.groups()
        target = by_id.get(int(target_id))
        if target is None or (actions is not None and action not in set(actions)):
            continue
        _, target_color, target_shape, target_orientation, _ = target
        for subject in (source_shape, source_color, *_orientation_properties(source_orientation)):
            if not _allowed(subject, objects, properties):
                continue
            for target_value in (target_shape, target_color, *_orientation_properties(target_orientation)):
                if _allowed(target_value, objects, properties):
                    result.append([subject, action, target_value])
    return result or [[]]


def parse_scientist_structure(scene):
    """Resolve IDs to sorted piece indices; touching is symmetric, pointing is not.

    Stacking implies contact in both directions, as in the tensor representation.
    Wedge ``doorstop`` is represented as ``flat`` in Scientist's object API.
    """
    parsed = _scene_items(scene)
    index_by_id = {item[0]: index for index, item in enumerate(parsed)}
    pieces = []
    for item_id, color, shape, orientation, relation in parsed:
        touching = []
        on_top_of = None
        pointing = None
        relation_match = re.match(
            r"^(touching|on_top_of|pointing)\(\s*(\d+)\s*\)$", relation
        )
        if relation_match is not None:
            action, target_id = relation_match.groups()
            target_index = index_by_id.get(int(target_id))
            if target_index is not None:
                if action == "touching":
                    touching.append(target_index)
                elif action == "on_top_of":
                    on_top_of = target_index
                else:
                    pointing = target_index
        pieces.append(ZendoPiece(
            color=color,
            shape=shape,
            orientation="flat" if orientation == "doorstop" else orientation,
            touching=touching,
            on_top_of=on_top_of,
            pointing=pointing,
        ))
    for source_index, piece in enumerate(pieces):
        if piece.on_top_of is not None and piece.on_top_of not in piece.touching:
            piece.touching.append(piece.on_top_of)
        for target_index in list(piece.touching):
            if source_index not in pieces[target_index].touching:
                pieces[target_index].touching.append(source_index)
    for piece in pieces:
        piece.touching.sort()
    return ZendoStructure(pieces)
