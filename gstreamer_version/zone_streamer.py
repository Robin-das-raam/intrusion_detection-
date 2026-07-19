# zones_streamer.py
import sys
import gi
import numpy as np
import cv2

# --- ADDED IMPORTS ---
# Import our configuration and the necessary utility functions
import configs_v1
from up_inference_core import (
    load_zones, 
    scale_zones, 
    build_zone_overlay
)

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# =============================================================
# 1. PRE-COMPUTE ZONE OVERLAYS (NEW SECTION)
# =============================================================
# For efficiency, we create the zone overlay images once at the start.
# Then, in real-time, we just blend these images onto the video frames.

print("[+] Pre-computing zone overlays...")
per_camera_overlays = []
target_shape = (configs_v1.RESIZED_HEIGHT, configs_v1.RESIZED_WIDTH)
target_dim_wh = (configs_v1.RESIZED_WIDTH, configs_v1.RESIZED_HEIGHT)

for i, zone_path in enumerate(configs_v1.ZONES_PATHS):
    raw_zones = load_zones(zone_path)
    if not raw_zones:
        # If a camera has no zones, add an empty placeholder
        per_camera_overlays.append(None)
        print(f"[Warning] No zones found for camera {i}, will not draw overlay.")
        continue
    
    # Scale zones from their original drawing dimensions to our target processing size
    scaled_zones = scale_zones(raw_zones, configs_v1.ZONES_SRC_DIM, target_dim_wh)
    
    # Create a pre-rendered image of the zone outlines and text
    overlay_image = build_zone_overlay(scaled_zones, target_shape)
    per_camera_overlays.append(overlay_image)
    print(f"[+] Zone overlay for camera {i} created.")

# =============================================================
# 2. PAD PROBE CALLBACK TO DRAW ZONES (NEW SECTION)
# =============================================================
def pad_probe_callback(pad, info, camera_id):
    """
    This function is called for every frame. It draws the pre-computed
    zone overlay onto the frame.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK

    # Map the buffer so we can write to it
    success, map_info = gst_buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
    if not success: return Gst.PadProbeReturn.OK

    # Create a NumPy array from the GStreamer buffer's memory (no copy)
    frame = np.ndarray(
        (configs_v1.RESIZED_HEIGHT, configs_v1.RESIZED_WIDTH, 3),
        buffer=map_info.data,
        dtype=np.uint8
    )

    # Get the correct pre-rendered overlay for this camera
    overlay = per_camera_overlays[camera_id]
    
    # If an overlay exists, blend it onto the frame
    if overlay is not None:
        # cv2.addWeighted makes the overlay slightly transparent (alpha=0.6)
        # The result is written directly back into 'frame', which is our GstBuffer.
        cv2.addWeighted(frame, 1.0, overlay, 0.6, 0, dst=frame)

    # Unmap the buffer to release it
    gst_buffer.unmap(map_info)
    return Gst.PadProbeReturn.OK

# =============================================================
# 3. MAIN FUNCTION (MODIFIED FROM YOUR grid_streamer.py)
# =============================================================
def on_pad_added(src, new_pad, target_element): # This function is unchanged
    print(f"Dynamic pad created, linking {src.get_name()} to {target_element.get_name()}")
    sink_pad = target_element.get_static_pad("sink")
    if sink_pad.is_linked(): return
    new_pad.link(sink_pad)

def main():
    Gst.init(sys.argv[1:] if len(sys.argv) > 1 else None)
    print("Creating Pipeline...")
    pipeline = Gst.Pipeline.new("multi-camera-grid-pipeline")

    # Common elements are unchanged
    compositor = Gst.ElementFactory.make("compositor", "compositor")
    final_convert = Gst.ElementFactory.make("videoconvert", "final-convert")
    sink = Gst.ElementFactory.make("autovideosink", "video-sink")

    compositor.set_property("background", 1)
    pipeline.add(compositor); pipeline.add(final_convert); pipeline.add(sink)

    # --- Create and Link a Branch for Each Camera ---
    for i, url in enumerate(configs_v1.CAMERA_URLS):
        print(f"Creating branch for camera {i+1}")

        # --- MODIFICATION: ADD A 'queue' ELEMENT ---
        # This is the key to guaranteeing a writable buffer for our probe.
        source = Gst.ElementFactory.make("rtspsrc", f"source-{i}")
        depay = Gst.ElementFactory.make("rtph264depay", f"depay-{i}")
        parse = Gst.ElementFactory.make("h264parse", f"parse-{i}")
        decode = Gst.ElementFactory.make("avdec_h264", f"decode-{i}")
        convert1 = Gst.ElementFactory.make("videoconvert", f"convert1-{i}")
        scale = Gst.ElementFactory.make("videoscale", f"scale-{i}")
        capsfilter = Gst.ElementFactory.make("capsfilter", f"capsfilter-{i}")
        queue = Gst.ElementFactory.make("queue", f"queue-{i}") # <-- ADDED ELEMENT
        convert2 = Gst.ElementFactory.make("videoconvert", f"convert2-{i}")

        if not all([source, depay, parse, decode, convert1, scale, capsfilter, queue, convert2]):
            print(f"Error: Not all elements for branch {i} could be created.")
            return -1

        source.set_property("location", url); source.set_property("latency", 200)
        caps = Gst.Caps.from_string(f"video/x-raw, width={configs_v1.RESIZED_WIDTH}, height={configs_v1.RESIZED_HEIGHT}, format=BGR")
        capsfilter.set_property("caps", caps)

        for elem in [source, depay, parse, decode, convert1, scale, capsfilter, queue, convert2]:
            pipeline.add(elem)

        # --- MODIFICATION: LINK THE 'queue' IN THE CHAIN ---
        source.connect("pad-added", on_pad_added, depay)
        depay.link(parse); parse.link(decode); decode.link(convert1)
        convert1.link(scale); scale.link(capsfilter)
        capsfilter.link(queue)      # <-- LINK TO QUEUE
        queue.link(convert2)        # <-- LINK FROM QUEUE
        
        # Link branch to compositor (unchanged)
        sink_pad = compositor.get_request_pad("sink_%u")
        sink_pad.set_property("xpos", i * configs_v1.RESIZED_WIDTH)
        sink_pad.set_property("ypos", 0)
        src_pad = convert2.get_static_pad("src")
        src_pad.link(sink_pad)
        
        # --- MODIFICATION: ATTACH THE PAD PROBE ---
        # We attach the probe to the source pad of our new 'queue' element.
        # This guarantees we get a writable buffer.
        queue_src_pad = queue.get_static_pad("src")
        queue_src_pad.add_probe(Gst.PadProbeType.BUFFER, pad_probe_callback, i) # Pass 'i' as the camera_id

    # --- Final linking and startup (unchanged) ---
    compositor.link(final_convert)
    final_convert.link(sink)
    
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    try: loop.run()
    except KeyboardInterrupt: print("Exiting...")
    finally: pipeline.set_state(Gst.State.NULL); loop.quit()

if __name__ == "__main__":
    sys.exit(main())