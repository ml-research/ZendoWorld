from zendoworld.paths import PACKAGE_ROOT, RENDER_CONFIG, output_path
import subprocess
import os
import sys
import time
import torch
import shutil
from pathlib import Path
import zendoworld.experiments.compute_log as compute_log


def render_artifact_path(path, suffix=".png"):
    """Return the on-disk path for a run-namespaced render artifact."""
    relative_path = Path(str(path))
    namespace = os.environ.get("ZENDO_RUN_NAMESPACE", "").strip()
    if namespace:
        relative_path = Path(namespace) / relative_path
    return output_path("rendering") / (str(relative_path) + suffix)

def render_scene(scene, path):
    scene_str = str(scene)
    config_file = str(RENDER_CONFIG)
    namespaced_path = render_artifact_path(path, suffix="").relative_to(
        output_path("rendering")
    )
    pt_path = str(render_artifact_path(path, suffix=".pt"))


    try:
        # Delete any stale .pt file so a failed render can't silently reuse old data
        if os.path.exists(pt_path):
            os.remove(pt_path)

        blender_env = os.environ.copy()
        current_log = compute_log.current_log_path()
        if current_log:
            blender_env["ZENDO_COMPUTE_LOG"] = current_log
        # Use the site-packages of whichever Python is running this process,
        # falling back to BLENDER_PYTHONPATH env var for explicit overrides.
        site_packages = os.environ.get(
            "BLENDER_PYTHONPATH",
            next((p for p in sys.path if "site-packages" in p), ""),
        )
        blender_env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PACKAGE_ROOT.parent), site_packages, blender_env.get("PYTHONPATH", "")) if p)


        # Pin Blender to the last CUDA device visible to this process. With
        # device_map="auto", accelerate tends to put the embedding + vision
        # encoder + LM head on device 0, making it the heaviest; the last
        # device is the lightest gemma load, so Blender sharing it has the
        # least SM contention. Indices are container-local: if you ran docker
        # with `--gpus device=0,1,3`, "last device" is container index 2
        # (= host GPU 3). Override with ZENDO_BLENDER_CUDA_DEVICE.
        #
        # CUDA_VISIBLE_DEVICES is a *physical* device id, not relative to any
        # outer restriction -- so if this process already has its own
        # CUDA_VISIBLE_DEVICES narrowed (e.g. one worker among several, each
        # pinned to a single GPU), torch.cuda.device_count() is always 1 and
        # "last visible device" must be read from that inherited list, not
        # re-derived as a raw index (which would always resolve to "0" and
        # silently redirect every worker's Blender render onto physical GPU 0).
        inherited_visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
        if inherited_visible:
            default_blender_device = inherited_visible[-1]
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            default_blender_device = str(torch.cuda.device_count() - 1)
        else:
            default_blender_device = "0"
        blender_env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "ZENDO_BLENDER_CUDA_DEVICE", default_blender_device
        )

        proc = subprocess.Popen([
            "blender",
            "--background",
            "--python", str(Path(__file__).with_name("render_single.py")), "--",
            "--config-file", config_file,
            "--scene", scene_str,
            "--path", str(namespaced_path),
        ], env=blender_env)

        # Bound the wait so a hung Blender (e.g. unknown relation type spinning
        # the placement loop, or Cycles stalled on the GPU) can't block the
        # parent process forever. Override with ZENDO_BLENDER_TIMEOUT (seconds).
        timeout_s = float(os.environ.get("ZENDO_BLENDER_TIMEOUT", "600"))
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            print(f"Blender render timed out after {timeout_s:.0f}s for scene {scene_str!r}; killing subprocess.")
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            return None

        # Wait a moment for filesystem sync if needed
        time.sleep(1)
        # Read the results
        image_tensor = torch.load(pt_path, weights_only=True)
        return image_tensor
    except Exception as e:
        print(f"Error rendering scene: {e}")
        return None
