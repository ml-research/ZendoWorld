"""Input modality selected by the public experiment player tag."""


def uses_image_paths(player_tag):
    """Symbolic runners use scene descriptions even when image files exist."""
    return not player_tag.endswith("_symbolic")
