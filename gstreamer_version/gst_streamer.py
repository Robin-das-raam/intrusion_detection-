import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import numpy as np
import threading
import queue
import logging

logger = logging.getLogger(__name__)
Gst.init(None)


class GstH264Streamer:
    def __init__(self, width: int, height: int, fps: int = 30, bitrate_kbps: int = 4000):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration_ns = Gst.util_uint64_scale_int(1, Gst.SECOND, fps)
        self._running = True
        self._pts = 0

        # Key fixes:
        # - h264parse config-interval=-1  -> injects SPS/PPS regularly
        # - mpegtsmux pat-interval=2s     -> resends PAT/PMT so mpegts.js never misses them
        # - appsink emit-signals=false    -> avoids deadlock with pull-sample thread
        # - appsink max-buffers=100       -> don't drop the very first PAT buffer
        pipe_str = (
            f"appsrc name=src format=time is-live=true do-timestamp=true ! "
            f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbps} "
            f"key-int-max={fps} threads=4 ! "
            f"video/x-h264,profile=baseline,stream-format=byte-stream ! "
            f"h264parse config-interval=-1 ! "
            f"mpegtsmux alignment=7 pat-interval=2000000000 ! "
            f"appsink name=sink max-buffers=100 drop=false emit-signals=false sync=false"
        )

        self.pipeline = Gst.parse_launch(pipe_str)
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsink = self.pipeline.get_by_name("sink")

        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"
        )
        self.appsrc.set_property("caps", caps)

        self.chunk_queue = queue.Queue(maxsize=300)
        self.pipeline.set_state(Gst.State.PLAYING)

        self._pull_thread = threading.Thread(target=self._pull_loop, daemon=True)
        self._pull_thread.start()

        logger.info("GstH264Streamer ready: %dx%d @ %dfps", width, height, fps)

    def _pull_loop(self):
        first = True
        while self._running:
            try:
                # emit-signals=false + pull-sample is the safe combo
                sample = self.appsink.emit("pull-sample")
                if sample is None:
                    if not self._running:
                        break
                    continue

                buffer = sample.get_buffer()
                ok, mapinfo = buffer.map(Gst.MapFlags.READ)
                if not ok:
                    continue
                chunk = bytes(mapinfo.data)
                buffer.unmap(mapinfo)

                if first and len(chunk) > 0:
                    logger.info("First encoded chunk received: %d bytes", len(chunk))
                    first = False

                try:
                    self.chunk_queue.put(chunk, block=False)
                except queue.Full:
                    try:
                        self.chunk_queue.get_nowait()
                        self.chunk_queue.put(chunk, block=False)
                    except queue.Empty:
                        pass
            except Exception as exc:
                logger.error("Pull loop error: %s", exc)
                time.sleep(0.1)

    def push_frame(self, frame: np.ndarray):
        if not self._running or frame is None:
            return
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            logger.warning("Frame size mismatch: expected %dx%d, got %dx%d",
                           self.width, self.height, frame.shape[1], frame.shape[0])
            return
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
        if buf is None:
            logger.error("Failed to allocate GstBuffer")
            return

        written = buf.fill(0, frame.tobytes())
        if written != frame.nbytes:
            logger.error("Buffer fill incomplete: %d / %d", written, frame.nbytes)
            return

        buf.pts = self._pts
        buf.duration = self.duration_ns
        self._pts += self.duration_ns

        ret = self.appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            logger.warning("push-buffer returned %s", ret)

    def get_chunks(self):
        while self._running:
            try:
                chunk = self.chunk_queue.get(timeout=1.0)
                yield chunk
            except queue.Empty:
                continue

    def stop(self):
        self._running = False
        self.appsrc.emit("end-of-stream")
        self.pipeline.set_state(Gst.State.NULL)
        self._pull_thread.join(timeout=2.0)
        logger.info("GstH264Streamer stopped.")