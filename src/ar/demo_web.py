"""Single-operator keyboard input and paced MJPEG streaming."""

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Full, Queue
from threading import Condition, Thread
import time


_PAGE = b"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Kandinsky WM AR</title>
<style>
body{margin:0;min-height:100dvh;display:grid;place-items:center;background:#141414;color:#eee;
font:16px system-ui,sans-serif}main{width:min(100%,1180px);text-align:center}
img{display:block;width:100%;max-height:86dvh;object-fit:contain;aspect-ratio:3/2}
p{margin:.8rem .5rem}kbd{font:inherit;color:#fff;font-weight:bold}
</style>
<main><img src="/stream" alt="World model driving stream">
<p><kbd>Space</kbd> start or pause &middot; <kbd>W</kbd> accelerate &middot;
<kbd>S</kbd> brake &middot; <kbd>A</kbd> <kbd>D</kbd> steer</p>
<p id="status" aria-live="polite">Paused</p></main>
<script>
const map = {KeyW:"forward", ArrowUp:"forward", KeyS:"brake", ArrowDown:"brake",
 KeyA:"left", ArrowLeft:"left", KeyD:"right", ArrowRight:"right"};
const keys = new Set();
let running = false, sending = false, dirty = false;
const statusLabel = document.getElementById("status");
async function send() {
 if (sending) { dirty = true; return; }
 sending = true;
 try {
  const res = await fetch("/keys", {method:"POST", headers:{"Content-Type":"application/json"},
   body:JSON.stringify({running,keys:[...keys].map(k=>map[k])})});
  if (!res.ok) throw Error();
  statusLabel.textContent = running ? "Driving" : "Paused";
 } catch (_) { statusLabel.textContent="Connection lost"; }
 finally {
  sending = false;
  if (dirty) { dirty = false; send(); }
 }
}
addEventListener("keydown", e => {
 if (e.code === "Space") {
  e.preventDefault(); if (!e.repeat) { running = !running; send(); } return;
 }
 if (map[e.code]) { e.preventDefault(); keys.add(e.code); if (running) send(); }
});
addEventListener("keyup", e => { if (keys.delete(e.code) && running) send(); });
addEventListener("blur", () => {
 keys.clear();
 if (running) { running = false; send(); }
});
setInterval(() => { if (running) send(); }, 120);
</script></html>"""
_KEYS = frozenset(("forward", "brake", "left", "right"))


class Browser:
    def __init__(self, host, port):
        self.condition = Condition()
        self.running = False
        self.received = 0.0
        self.keys = frozenset()
        self.number = 0
        self.frame = None
        self.closed = False
        self.chunks = Queue(maxsize=1)
        browser = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                if self.path != "/keys":
                    self.send_error(404)
                    return
                if self.headers.get("Sec-Fetch-Site") == "cross-site":
                    self.send_error(403)
                    return
                self.connection.settimeout(2.0)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 256:
                        raise ValueError()
                    data = json.loads(self.rfile.read(size))
                    keys, running = data["keys"], data["running"]
                    if not isinstance(keys, list) or type(running) is not bool:
                        raise ValueError()
                    browser.update(keys, running)
                except (ValueError, KeyError, TypeError, TimeoutError):
                    self.send_error(400)
                    return
                self.send_response(204)
                self.end_headers()

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(_PAGE)
                elif self.path == "/stream":
                    self.stream_frames()
                else:
                    self.send_error(404)

            def stream_frames(self):
                self.connection.settimeout(5.0)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = 0
                try:
                    while True:
                        with browser.condition:
                            browser.condition.wait_for(
                                lambda: browser.closed or seen != browser.number
                            )
                            if browser.closed and seen == browser.number:
                                return
                            seen, jpeg = browser.number, browser.frame
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg + b"\r\n"
                        )
                except (TimeoutError, ConnectionError):
                    return

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.player = Thread(target=self.play, daemon=True)

    def __enter__(self):
        self.thread.start()
        self.player.start()
        return self

    def __exit__(self, *_exc_info):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.player.join()

    def update(self, keys, running):
        keys = frozenset(keys)
        if not keys <= _KEYS:
            raise ValueError("Unknown keyboard action")
        with self.condition:
            self.keys = keys
            self.running = running
            self.received = time.monotonic()
            self.condition.notify_all()

    def next_keys(self):
        with self.condition:
            self.condition.wait_for(
                lambda: self.running and time.monotonic() - self.received < 0.5
            )
            return self.keys

    def publish(self, frame):
        with self.condition:
            self.number += 1
            self.frame = frame
            self.condition.notify_all()

    def enqueue(self, frames):
        while not self.closed:
            try:
                self.chunks.put(frames, timeout=0.2)
                return
            except Full:
                continue

    def drain(self):
        self.chunks.join()

    def play(self):
        next_frame = time.monotonic()
        while not self.closed:
            try:
                frames = self.chunks.get(timeout=0.2)
            except Empty:
                continue
            try:
                for frame in frames:
                    if self.closed:
                        return
                    now = time.monotonic()
                    time.sleep(max(0.0, next_frame - now))
                    self.publish(frame)
                    next_frame = max(now, next_frame) + 1 / 24
            finally:
                self.chunks.task_done()


def encode_jpeg(rgb, quality):
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="JPEG", quality=quality)
    return output.getvalue()
