# grid_streamer.py
import sys
import gi

# Import our configuration
import configs

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def on_pad_added(src, new_pad, target_element):
    """
    Callback to link the 'sometimes' pad of rtspsrc to the depayloader.
    """
    print(f"Dynamic pad created, linking {src.get_name()} to {target_element.get_name()}")
    # Check if the target element's sink pad is already linked
    sink_pad = target_element.get_static_pad("sink")
    if sink_pad.is_linked():
        print("Sink pad already linked. Ignoring.")
        return

    # Link the new pad to the target element's sink pad
    pad_link_ret = new_pad.link(sink_pad)
    if pad_link_ret != Gst.PadLinkReturn.OK:
        print(f"Failed to link pads: {pad_link_ret}")


def main():
    # --- 1. GStreamer Initialization ---
    Gst.init(sys.argv[1:] if len(sys.argv) > 1 else None)

    print("Creating Pipeline...")
    pipeline = Gst.Pipeline.new("multi-camera-grid-pipeline")

    if not pipeline:
        print("Error: Could not create pipeline.")
        return -1

    # --- 2. Create GStreamer Elements ---
    # Common elements for the final output
    compositor = Gst.ElementFactory.make("compositor", "compositor")
    final_convert = Gst.ElementFactory.make("videoconvert", "final-convert")
    sink = Gst.ElementFactory.make("autovideosink", "video-sink")

    # Set compositor background to black
    compositor.set_property("background", 1)

    # Add common elements to the pipeline
    pipeline.add(compositor)
    pipeline.add(final_convert)
    pipeline.add(sink)

    # --- 3. Create and Link a Branch for Each Camera ---
    for i, url in enumerate(configs.CAMERA_URLS):
        # print(f"Creating branch for camera {i+1} at {url}")

        # Create elements for this branch
        source = Gst.ElementFactory.make("rtspsrc", f"source-{i}")
        depay = Gst.ElementFactory.make("rtph264depay", f"depay-{i}")
        parse = Gst.ElementFactory.make("h264parse", f"parse-{i}")
        decode = Gst.ElementFactory.make("avdec_h264", f"decode-{i}")
        convert1 = Gst.ElementFactory.make("videoconvert", f"convert1-{i}")
        scale = Gst.ElementFactory.make("videoscale", f"scale-{i}")
        capsfilter = Gst.ElementFactory.make("capsfilter", f"capsfilter-{i}")
        convert2 = Gst.ElementFactory.make("videoconvert", f"convert2-{i}")

        if not all([source, depay, parse, decode, convert1, scale, capsfilter, convert2]):
            print(f"Error: Not all elements for branch {i} could be created.")
            return -1

        # Configure elements
        source.set_property("location", url)
        source.set_property("latency", 200)

        # Set the desired output size for this stream
        caps = Gst.Caps.from_string(f"video/x-raw, width={configs.RESIZED_WIDTH}, height={configs.RESIZED_HEIGHT}")
        capsfilter.set_property("caps", caps)

        # Add elements to the pipeline
        pipeline.add(source)
        pipeline.add(depay)
        pipeline.add(parse)
        pipeline.add(decode)
        pipeline.add(convert1)
        pipeline.add(scale)
        pipeline.add(capsfilter)
        pipeline.add(convert2)

        # Link static elements
        depay.link(parse)
        parse.link(decode)
        decode.link(convert1)
        convert1.link(scale)
        scale.link(capsfilter)
        capsfilter.link(convert2)
        
        # Link the branch to the compositor
        # We need to request a new sink pad from the compositor
        sink_pad = compositor.get_request_pad("sink_%u")
        sink_pad.set_property("xpos", i * configs.RESIZED_WIDTH) # Position horizontally
        sink_pad.set_property("ypos", 0)
        
        # Get the source pad from the last element in the branch and link it
        src_pad = convert2.get_static_pad("src")
        src_pad.link(sink_pad)
        
        # Link the 'sometimes' pad of rtspsrc
        source.connect("pad-added", on_pad_added, depay)

    # --- 4. Link the Final Elements ---
    compositor.link(final_convert)
    final_convert.link(sink)
    
    # --- 5. Start the Pipeline ---
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    # --- 6. Run the Main Loop ---
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

if __name__ == "__main__":
    sys.exit(main())