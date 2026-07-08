# This file contains code derived from:
# - https://github.com/CapArrow/zendo_game_dataset_generator
import subprocess
import os
import time
import torch
import shutil

def render_scene(scene, path):
    scene_str = str(scene)
    config_file = "generation/configs/simple_config.yml"
    output_dir = "generation/output"

    try:
        env = os.environ.copy()

        env["PYTHONNOUSERSITE"] = "0"
        proc = subprocess.Popen(
            [
                "blender", "--background", "--python-use-system-env", "--python", "generation/render_single.py", "--",
                "--config-file", config_file,
                "--scene", scene_str,
                "--path", str(path),
            ],
            preexec_fn=os.setsid,
            env=env
        )
        proc.wait()

        time.sleep(1)
        image_tensor = torch.load(os.path.join(output_dir, (str(path) + ".pt")), weights_only=True)
        return image_tensor
    except Exception as e:
        print(f"Error rendering scene: {e}")
        return None
