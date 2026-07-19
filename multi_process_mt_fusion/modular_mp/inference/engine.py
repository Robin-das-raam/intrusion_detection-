# inference/engine.py
# Inference engine controls
# Handles start, stop, status of inference loop

import threading
from inference.loop import run_inference_loop, inference_frames, inference_frames_lock

# ─────────────────────────────────────────
# Engine state
# ─────────────────────────────────────────
_running = False
_thread = None
_stop_event = threading.Event()


# ─────────────────────────────────────────
# Controls
# ─────────────────────────────────────────
def start_inference() -> dict:
    """
    Start inference loop in background thread.
    Returns status dict.
    """
    global _running, _thread, _stop_event

    if _running:
        return {"status": "already_running"}

    _stop_event.clear()
    _running = True
    _thread = threading.Thread(
        target=run_inference_loop,
        args=(_stop_event,),
        daemon=True
    )
    _thread.start()
    print("[Engine] Inference started.")
    return {"status": "started"}


def stop_inference() -> dict:
    """
    Stop inference loop.
    Returns status dict.
    """
    global _running, _stop_event

    if not _running:
        return {"status": "not_running"}

    _stop_event.set()
    _running = False
    print("[Engine] Inference stopped.")
    return {"status": "stopped"}


def is_running() -> bool:
    """Check if inference is currently running."""
    return _running


def get_inference_frame(cam_id: str):
    """
    Get latest annotated frame for a camera.

    Args:
        cam_id: camera id string

    Returns:
        np.ndarray | None: annotated frame or None
    """
    with inference_frames_lock:
        return inference_frames.get(cam_id)


def get_all_inference_cam_ids() -> list:
    """
    Get list of camera ids that have inference frames.

    Returns:
        list: list of cam_id strings
    """
    with inference_frames_lock:
        return list(inference_frames.keys())


def get_engine_status() -> dict:
    """
    Get full engine status.

    Returns:
        dict: status info
    """
    return {
        "running": _running,
        "active_cameras": get_all_inference_cam_ids()
    }