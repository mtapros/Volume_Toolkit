# Android-focused Volume Toolkit (threaded decoder + background poller)
#
# Notes for this revision (v2.0.5):
# - Removed the "QR:" status label from the main screen (QR decode still updates "Last QR").
# - Thumbnails are rotated 90 degrees CCW to match live preview orientation.
# - Tapping a thumbnail "freezes" the preview by replacing the live texture with the thumbnail
#   stretched exactly the same as live preview; overlays remain on top (same PreviewOverlay).
# - While frozen, live view can keep running but new live textures are ignored.
# - After tap, the app fetches the full-resolution JPG in the background and replaces
#   the low-res thumbnail when ready (same orientation as live preview).
# - Tap the preview to unfreeze and return to live.
# - Capture type default is JPG; RAW remains selectable in the menu (no RAW download pipeline implemented).
#
import os
import json
import threading
import time
from datetime import datetime
from io import BytesIO
import csv
import queue

import requests
import urllib3

import kivy
kivy.require("2.0.0")

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.properties import NumericProperty, BooleanProperty, StringProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.utils import platform

from PIL import Image as PILImage

import cv2
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if os.environ.get("ANDROID_ARGUMENT"):
    private_dir = os.environ.get("ANDROID_PRIVATE")
    if private_dir:
        kivy_home = os.path.join(private_dir, ".kivy")
        os.makedirs(kivy_home, exist_ok=True)
        os.environ["KIVY_HOME"] = kivy_home


class PreviewOverlay(FloatLayout):
    show_border = BooleanProperty(True)
    show_grid = BooleanProperty(True)
    show_57 = BooleanProperty(True)
    show_810 = BooleanProperty(True)
    show_oval = BooleanProperty(True)

    grid_n = NumericProperty(3)

    oval_cx = NumericProperty(0.5)
    oval_cy = NumericProperty(0.6)
    oval_w = NumericProperty(0.333)
    oval_h = NumericProperty(0.333)

    # v2.0.4+: default rotation is 270 to rotate 180 from original behavior.
    preview_rotation = NumericProperty(270)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.img = Image(allow_stretch=True, keep_ratio=True)
        try:
            self.img.fit_mode = "contain"
        except Exception:
            pass
        self.add_widget(self.img)

        lw = 2

        with self.img.canvas.after:
            self._c_border = Color(0.2, 0.6, 1.0, 1.0)
            self._ln_border = Line(width=lw)

            self._c_grid = Color(1.0, 0.6, 0.0, 0.85)

            self._c_57 = Color(1.0, 0.2, 0.2, 0.95)
            self._ln_57 = Line(width=lw)

            self._c_810 = Color(1.0, 0.9, 0.2, 0.95)
            self._ln_810 = Line(width=lw)

            self._c_oval = Color(0.7, 0.2, 1.0, 0.95)
            self._ln_oval = Line(width=lw)

        self._ln_grid_list = []

        self.bind(pos=self._redraw, size=self._redraw)
        self.bind(
            show_border=self._redraw, show_grid=self._redraw, show_57=self._redraw,
            show_810=self._redraw, show_oval=self._redraw,
            grid_n=self._redraw,
            oval_cx=self._redraw, oval_cy=self._redraw, oval_w=self._redraw, oval_h=self._redraw
        )
        self.img.bind(pos=self._redraw, size=self._redraw, texture=self._redraw, texture_size=self._redraw)

        self._redraw()

    def set_texture(self, texture):
        self.img.texture = texture
        self._redraw()

    def _drawn_rect(self):
        wx, wy = self.img.pos
        ww, wh = self.img.size
        try:
            iw, ih = self.img.norm_image_size
        except Exception:
            return (wx, wy, ww, wh)
        dx = wx + (ww - iw) / 2.0
        dy = wy + (wh - ih) / 2.0
        return (dx, dy, iw, ih)

    @staticmethod
    def _center_crop_rect(frame_x, frame_y, frame_w, frame_h, aspect):
        frame_aspect = frame_w / frame_h
        if frame_aspect >= aspect:
            h = frame_h
            w = h * aspect
        else:
            w = frame_w
            h = w / aspect
        x = frame_x + (frame_w - w) / 2.0
        y = frame_y + (frame_h - h) / 2.0
        return (x, y, w, h)

    def _crop_aspect(self, a_w, a_h, fw, fh):
        if fw >= fh:
            return float(a_h) / float(a_w)
        return float(a_w) / float(a_h)

    def _clear_line_modes(self, ln: Line):
        try:
            ln.points = []
        except Exception:
            pass
        try:
            ln.rectangle = (0, 0, 0, 0)
        except Exception:
            pass

    def _redraw(self, *args):
        fx, fy, fw, fh = self._drawn_rect()

        self._ln_border.rectangle = (fx, fy, fw, fh) if self.show_border else (0, 0, 0, 0)

        if self.show_57:
            asp57 = self._crop_aspect(5.0, 7.0, fw, fh)
            self._ln_57.rectangle = self._center_crop_rect(fx, fy, fw, fh, asp57)
        else:
            self._ln_57.rectangle = (0, 0, 0, 0)

        if self.show_810:
            asp810 = self._crop_aspect(4.0, 5.0, fw, fh)
            self._ln_810.rectangle = self._center_crop_rect(fx, fy, fw, fh, asp810)
        else:
            self._ln_810.rectangle = (0, 0, 0, 0)

        n = int(self.grid_n)
        for line in list(self._ln_grid_list):
            try:
                self.img.canvas.after.remove(line)
            except Exception:
                pass
        self._ln_grid_list = []

        if self.show_grid and n >= 2:
            for i in range(1, n):
                x = fx + fw * (i / n)
                line = Line(points=[x, fy, x, fy + fh], width=2)
                self.img.canvas.after.add(line)
                self._ln_grid_list.append(line)
            for i in range(1, n):
                y = fy + fh * (i / n)
                line = Line(points=[fx, y, fx + fw, y], width=2)
                self.img.canvas.after.add(line)
                self._ln_grid_list.append(line)

        if self.show_oval:
            cx = fx + fw * float(self.oval_cx)
            cy = fy + fh * float(self.oval_cy)
            ow = fw * float(self.oval_w)
            oh = fh * float(self.oval_h)

            ow = max(0.05 * fw, min(ow, fw))
            oh = max(0.05 * fh, min(oh, fh))

            left = max(fx, min(cx - ow / 2.0, fx + fw - ow))
            bottom = max(fy, min(cy - oh / 2.0, fy + fh - oh))

            self._clear_line_modes(self._ln_oval)
            self._ln_oval.ellipse = (left, bottom, ow, oh)
        else:
            self._clear_line_modes(self._ln_oval)
            self._ln_oval.ellipse = (0, 0, 0, 0)


class CaptureType:
    JPG = "JPG"
    RAW = "RAW"


class VolumeToolkitApp(App):
    # v2.0.5: JPG default
    capture_type = StringProperty(CaptureType.JPG)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.connected = False
        self.camera_ip = "172.25.162.76"

        self.live_running = False
        self.session_started = False

        self._lock = threading.Lock()
        self._last_decoded_ts = 0.0

        self._fetch_thread = None
        self._display_event = None

        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        self._log_lines = []
        self._max_log_lines = 300

        self._frame_texture = None
        self._frame_size = None

        self.dropdown = None

        # QR decode
        self.qr_enabled = False
        self.qr_interval_s = 0.15
        self.qr_new_gate_s = 0.70
        self._qr_detector = cv2.QRCodeDetector()
        self._qr_thread = None
        self._latest_qr_text = None
        self._qr_seen = set()
        self._qr_last_add_time = 0.0

        # last decoded for QR
        self._latest_decoded_bgr = None
        self._latest_decoded_bgr_ts = 0.0

        # decoder thread
        self._decode_queue = queue.Queue(maxsize=2)
        self._decoder_thread = threading.Thread(target=self._decoder_loop, daemon=True)
        self._decoder_stop = threading.Event()
        self._decoder_thread.start()

        # author push (manual)
        self.author_max_chars = 60
        self._last_committed_author = None
        self._author_update_in_flight = False

        self._current_payload = ""

        # CSV / headers
        self.csv_headers = []
        self.csv_rows = []
        self.selected_headers = []
        self._headers_popup = None

        # thumbnails
        self._thumb_textures = []
        self._thumb_images = []
        self._thumb_paths = []  # ccapi paths

        self.thumb_dir = "thumbs"

        self._last_seen_image = None
        self._poll_thread = None
        self._poll_thread_stop = threading.Event()
        self.poll_interval_s = 2.0

        self._session = requests.Session()
        self._session.verify = False

        self._android_activity_bound = False
        self._csv_req_code = 4242

        # popups
        self._log_popup = None
        self._log_popup_label = None
        self._ip_popup = None
        self._fps_popup = None

        # v2.0.5: freeze (still) mode
        self._freeze_active = False
        self._freeze_ccapi_path = None
        self._freeze_request_id = 0  # increments on each freeze to ignore stale downloads

    def build(self):
        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        header = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        header.add_widget(Label(text="Volume Toolkit v2.0.5", font_size=sp(18)))
        self.menu_btn = Button(text="Menu", size_hint=(None, 1), width=dp(90), font_size=sp(16))
        header.add_widget(self.menu_btn)
        root.add_widget(header)

        row2 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.connect_btn = Button(text="Connect", font_size=sp(16), size_hint=(1, 1))
        self.start_btn = Button(text="Start", disabled=True, font_size=sp(16), size_hint=(1, 1))
        self.qr_btn = Button(text="QR Detect: OFF", disabled=True, font_size=sp(16), size_hint=(1, 1))
        row2.add_widget(self.connect_btn)
        row2.add_widget(self.start_btn)
        row2.add_widget(self.qr_btn)
        root.add_widget(row2)

        row_author = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(40))
        row_author.add_widget(Label(text="Author", size_hint=(None, 1), width=dp(90), font_size=sp(14)))
        self.author_label = Label(text="(not connected)", size_hint=(1, 1), font_size=sp(14),
                                  halign="left", valign="middle")
        self.author_label.bind(size=lambda *_: setattr(self.author_label, "text_size", (self.author_label.width, None)))
        row_author.add_widget(self.author_label)
        root.add_widget(row_author)

        row_payload = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(40))
        row_payload.add_widget(Label(text="Last QR", size_hint=(None, 1), width=dp(90), font_size=sp(14)))
        self.payload_label = Label(text="(none)", size_hint=(1, 1), font_size=sp(14),
                                   halign="left", valign="middle")
        self.payload_label.bind(size=lambda *_: setattr(self.payload_label, "text_size", (self.payload_label.width, None)))
        row_payload.add_widget(self.payload_label)
        root.add_widget(row_payload)

        self.metrics = Label(text="Delay: -- ms | Fetch: 0 | Decode: 0 | Display: 0",
                             size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.metrics)

        main_area = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, 0.6))

        preview_holder = AnchorLayout(anchor_x="center", anchor_y="center", size_hint=(0.75, 1))
        self.preview_scatter = Scatter(do_translation=True, do_scale=True, do_rotation=False,
                                       scale_min=0.5, scale_max=2.5)
        self.preview_scatter.size_hint = (None, None)

        self.preview = PreviewOverlay(size_hint=(None, None))
        # Tap preview to unfreeze
        self.preview.bind(on_touch_down=self._on_preview_touch)

        self.preview_scatter.add_widget(self.preview)
        preview_holder.add_widget(self.preview_scatter)
        main_area.add_widget(preview_holder)

        sidebar = BoxLayout(orientation="vertical", size_hint=(0.25, 1), spacing=dp(4))
        sidebar.add_widget(Label(text="Last 5", size_hint=(1, None), height=dp(20), font_size=sp(12)))
        for idx in range(5):
            img = Image(size_hint=(1, 0.18), allow_stretch=True, keep_ratio=True)
            img.thumb_index = idx
            img.bind(on_touch_down=self._on_thumb_touch)
            sidebar.add_widget(img)
            self._thumb_images.append(img)
        main_area.add_widget(sidebar)

        root.add_widget(main_area)

        self.push_btn = Button(text="Push Payload", size_hint=(1, None), height=dp(48),
                               font_size=sp(16), disabled=True)
        root.add_widget(self.push_btn)

        def fit_preview_to_holder(*_):
            w = max(dp(220), preview_holder.width * 0.98)
            h = max(dp(220), preview_holder.height * 0.98)
            self.preview_scatter.size = (w, h)
            self.preview.size = (w, h)
            self.preview_scatter.scale = 1.0
            self.preview_scatter.pos = (
                preview_holder.x + (preview_holder.width - w) / 2.0,
                preview_holder.y + (preview_holder.height - h) / 2.0
            )

        preview_holder.bind(pos=fit_preview_to_holder, size=fit_preview_to_holder)

        self.dropdown = self._build_dropdown(fit_preview_to_holder)
        self.menu_btn.bind(on_release=lambda *_: self.dropdown.open(self.menu_btn))

        self.connect_btn.bind(on_press=lambda *_: self.connect_camera())
        self.start_btn.bind(on_press=lambda *_: self.toggle_liveview())
        self.qr_btn.bind(on_press=lambda *_: self.toggle_qr_detect())
        self.push_btn.bind(on_press=lambda *_: self.push_payload())

        self._reschedule_display_loop(12)

        self._set_controls_idle()
        self.log("Android CCAPI GUI ready")
        return root

    # ---------- lifecycle ----------
    def on_start(self):
        if platform == "android":
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])
                self.log("Requested storage permissions (Android).")
            except Exception as e:
                self.log(f"Permission request failed: {e}")

    # ---------- logging ----------
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]
        if self._log_popup is not None and self._log_popup_label is not None:
            self._log_popup_label.text = "\n".join(self._log_lines)

    # ---------- HTTP ----------
    def _json_call(self, method, path, payload=None, timeout=8.0):
        url = f"https://{self.camera_ip}{path}"
        try:
            if method == "GET":
                resp = self._session.get(url, timeout=timeout)
            elif method == "POST":
                resp = self._session.post(url, json=payload, timeout=timeout)
            elif method == "PUT":
                resp = self._session.put(url, json=payload, timeout=timeout)
            elif method == "DELETE":
                resp = self._session.delete(url, timeout=timeout)
            else:
                raise ValueError("Unsupported method")

            status = f"{resp.status_code} {resp.reason}"
            data = None
            if resp.content:
                try:
                    data = resp.json()
                except Exception:
                    data = None
            return status, data
        except Exception as e:
            return f"ERR {e}", None

    def _get_bytes(self, url: str, timeout=12.0) -> bytes:
        resp = self._session.get(url, stream=True, timeout=timeout)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code} {resp.reason}")
        return resp.content

    # ---------- Author refresh ----------
    def refresh_author(self, note=None):
        if not self.connected:
            Clock.schedule_once(lambda *_: setattr(self.author_label, "text", "(not connected)"), 0)
            return

        def worker():
            st, data = self._json_call("GET", "/ccapi/ver100/functions/registeredname/author", None, timeout=8.0)
            if st.startswith("200") and isinstance(data, dict):
                author = (data.get("author") or "").strip()
            else:
                author = f"(read failed: {st})"

            Clock.schedule_once(lambda *_: setattr(self.author_label, "text", author if author else "(empty)"), 0)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- controls ----------
    def connect_camera(self):
        if self.live_running:
            self.log("Connect disabled while live view is running. Stop live view first.")
            return

        self.log(f"Connecting to {self.camera_ip}:443")
        st, data = self._json_call("GET", "/ccapi/ver100/deviceinformation", None, timeout=8.0)
        if st.startswith("200") and data:
            self.connected = True
            self.log("Connected OK")
            self.author_label.text = "(loading...)"
            self.refresh_author(note="connect")
        else:
            self.connected = False
            self.author_label.text = f"(connect failed: {st})"
            self.log(f"Connect failed: {st}")

        self._set_controls_idle()

    def _set_controls_idle(self):
        self.connect_btn.disabled = False
        self.start_btn.disabled = not self.connected
        self.start_btn.text = "Start"
        self.qr_btn.disabled = not self.connected
        self.push_btn.disabled = not self.connected

    def _set_controls_running(self):
        self.connect_btn.disabled = True
        self.start_btn.disabled = False
        self.start_btn.text = "Stop"
        self.qr_btn.disabled = False
        self.push_btn.disabled = False

    def toggle_liveview(self):
        if not self.connected:
            return
        if self.live_running:
            self.stop_liveview()
        else:
            self.start_liveview()

    def start_liveview(self):
        if not self.connected or self.live_running:
            return

        payload = {"liveviewsize": "small", "cameradisplay": "on"}
        self.log("Starting live view")
        st, _ = self._json_call("POST", "/ccapi/ver100/shooting/liveview", payload, timeout=10.0)
        if not st.startswith("200"):
            self.log(f"Live view start failed: {st}")
            return

        self.session_started = True
        self.live_running = True
        self._set_controls_running()

        with self._lock:
            self._latest_decoded_bgr = None
            self._latest_decoded_bgr_ts = 0.0

        self._last_decoded_ts = 0.0
        self._frame_texture = None
        self._frame_size = None

        self._qr_seen = set()
        self._qr_last_add_time = 0.0

        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        self._fetch_thread = threading.Thread(target=self._liveview_fetch_loop, daemon=True)
        self._fetch_thread.start()

        self._qr_thread = threading.Thread(target=self._qr_loop, daemon=True)
        self._qr_thread.start()

    def stop_liveview(self):
        if not self.live_running:
            self._set_controls_idle()
            return

        self.live_running = False
        if self.session_started:
            try:
                self._json_call("DELETE", "/ccapi/ver100/shooting/liveview", None, timeout=6.0)
            except Exception:
                pass
            self.session_started = False

        self.log("Live view stopped")
        self._set_controls_idle()

    # ---------- QR ----------
    def toggle_qr_detect(self):
        self.qr_enabled = not bool(self.qr_enabled)
        self.qr_btn.text = "QR Detect: ON" if self.qr_enabled else "QR Detect: OFF"
        self.log(f"QR detect {'enabled' if self.qr_enabled else 'disabled'}")

    def _set_payload(self, payload: str):
        payload = (payload or "").strip()
        self._current_payload = payload
        self.payload_label.text = payload if payload else "(none)"

    # ---------- push payload ----------
    def _author_value(self, payload: str) -> str:
        s = (payload or "").strip()
        if not s:
            return ""
        return s[: int(self.author_max_chars)]

    def push_payload(self):
        if not self.connected:
            self.log("Push payload skipped: not connected")
            return

        payload = self._author_value(self._current_payload)
        if not payload:
            self.log("Push payload skipped: no payload")
            return

        if self._author_update_in_flight:
            self.log("Push payload skipped: update in flight")
            return

        if self._last_committed_author == payload:
            self.log("Push payload skipped: already pushed")
            return

        self._author_update_in_flight = True
        self.log(f"Pushing payload to Author: '{payload}'")
        threading.Thread(target=self._commit_author_worker, args=(payload, "manual"), daemon=True).start()

    def _commit_author_worker(self, value: str, source: str):
        ok = False
        got = None
        err = None
        try:
            st_put, _ = self._json_call(
                "PUT",
                "/ccapi/ver100/functions/registeredname/author",
                {"author": value},
                timeout=8.0
            )
            if not st_put.startswith("200"):
                raise Exception(f"PUT failed: {st_put}")

            st_get, data = self._json_call(
                "GET",
                "/ccapi/ver100/functions/registeredname/author",
                None,
                timeout=8.0
            )
            if not st_get.startswith("200") or not isinstance(data, dict):
                raise Exception(f"GET failed: {st_get}")

            got = (data.get("author") or "").strip()
            ok = (got == value)

        except Exception as e:
            err = str(e)

        def _finish(_dt):
            self._author_update_in_flight = False
            if ok:
                self._last_committed_author = value
                self.log(f"Author updated+verified ({source}): '{value}'")
            else:
                self.log(f"Author verify failed ({source}). wrote='{value}' read='{got}' err='{err}'")
            self.refresh_author(note="after push")

        Clock.schedule_once(_finish, 0)

    # ---------- freeze/unfreeze ----------
    def _on_preview_touch(self, widget, touch):
        if not self._freeze_active:
            return False
        # only if tap is inside the preview widget
        if not self.preview.collide_point(*touch.pos):
            return False
        # unfreeze on tap
        self._unfreeze_preview()
        return True

    def _unfreeze_preview(self):
        self._freeze_active = False
        self._freeze_ccapi_path = None
        self._freeze_request_id += 1  # invalidate any in-flight still downloads
        self.log("Unfroze preview (returning to live)")

    def _freeze_with_texture(self, ccapi_path: str, tex: Texture):
        self._freeze_active = True
        self._freeze_ccapi_path = ccapi_path
        self._freeze_request_id += 1
        rid = self._freeze_request_id
        self.preview.set_texture(tex)
        self.log(f"Froze preview with still: {ccapi_path} (rid={rid})")
        return rid

    # ---------- live fetch/decoder/qr loops ----------
    def _reschedule_display_loop(self, fps):
        if self._display_event is not None:
            self._display_event.cancel()
        fps = max(1, int(fps))
        self._display_event = Clock.schedule_interval(self._display_tick, 1.0 / fps)

    def _display_tick(self, dt):
        self._display_count += 1
        self._update_metrics(self._last_decoded_ts)

    def _liveview_fetch_loop(self):
        url = f"https://{self.camera_ip}/ccapi/ver100/shooting/liveview/flip"
        while self.live_running:
            try:
                resp = self._session.get(url, timeout=5.0)
                if resp.status_code == 200 and resp.content:
                    jpeg = resp.content
                    ts = time.time()
                    self._fetch_count += 1
                    try:
                        self._decode_queue.put_nowait((jpeg, ts))
                    except queue.Full:
                        try:
                            _ = self._decode_queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self._decode_queue.put_nowait((jpeg, ts))
                        except Exception:
                            pass
                else:
                    time.sleep(0.03)
            except Exception as e:
                self.log(f"liveview fetch error: {e}")
                time.sleep(0.10)

    def _decoder_loop(self):
        while not self._decoder_stop.is_set():
            try:
                jpeg, ts = self._decode_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                rot = int(self.preview.preview_rotation) % 360
                if rot == 90:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                elif rot == 270:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

                with self._lock:
                    self._latest_decoded_bgr = bgr.copy()
                    self._latest_decoded_bgr_ts = ts

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                rgb_bytes = rgb.tobytes()

                def _update_texture_on_main(_dt, rgb_bytes=rgb_bytes, w=w, h=h, ts=ts):
                    # v2.0.5: if frozen, ignore live updates
                    if self._freeze_active:
                        return
                    try:
                        if self._frame_texture is None or self._frame_size != (w, h):
                            tex = Texture.create(size=(w, h), colorfmt="rgb")
                            tex.flip_vertical()
                            self._frame_texture = tex
                            self._frame_size = (w, h)
                            self.log(f"texture init size={w}x{h}")
                        self._frame_texture.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
                        self.preview.set_texture(self._frame_texture)
                        self._last_decoded_ts = ts
                    except Exception as e:
                        self.log(f"texture update err: {e}")

                Clock.schedule_once(_update_texture_on_main, 0)
                self._decode_count += 1

            except Exception:
                continue

    def _qr_loop(self):
        last_processed_ts = 0.0
        while self.live_running:
            if not self.qr_enabled:
                time.sleep(0.10)
                continue

            with self._lock:
                bgr = None
                ts = getattr(self, "_latest_decoded_bgr_ts", 0.0)
                if self._latest_decoded_bgr is not None:
                    bgr = self._latest_decoded_bgr.copy()

            if bgr is None or ts <= last_processed_ts:
                time.sleep(0.05)
                continue

            try:
                decoded, _points, _ = self._qr_detector.detectAndDecode(bgr)
                qr_text = decoded.strip() if isinstance(decoded, str) else ""
                if qr_text:
                    self._publish_qr(qr_text)
            except Exception:
                pass

            last_processed_ts = ts
            time.sleep(max(0.05, float(self.qr_interval_s)))

    def _publish_qr(self, text):
        now = time.time()
        if text:
            if (text not in self._qr_seen) and (now - self._qr_last_add_time >= self.qr_new_gate_s):
                self._qr_seen.add(text)
                self._qr_last_add_time = now
                self.log(f"QR: {text}")
        Clock.schedule_once(lambda *_: self._set_payload(text), 0)

    def _update_metrics(self, frame_ts):
        now = time.time()
        if now - self._stat_t0 >= 1.0:
            dt_s = now - self._stat_t0
            fetch_fps = self._fetch_count / dt_s
            dec_fps = self._decode_count / dt_s
            disp_fps = self._display_count / dt_s
            delay_ms = int((now - frame_ts) * 1000) if frame_ts else -1
            self.metrics.text = (
                f"Delay: {delay_ms if delay_ms >= 0 else '--'} ms | "
                f"Fetch: {fetch_fps:.1f} | Decode: {dec_fps:.1f} | Display: {disp_fps:.1f}"
            )
            self._fetch_count = 0
            self._decode_count = 0
            self._display_count = 0
            self._stat_t0 = now

    # ---------- still image fetch ----------
    def _contents_fullres_url(self, ccapi_path: str) -> str:
        """
        User-provided rule:
          full JPG = GET https://<camera-ip>/ccapi/ver100/contents/<path>
        where <path> is the path on the SD card.

        In our app, ccapi_path values coming from /ccapi/ver120/contents look like:
          /ccapi/ver120/contents/.../IMG_0001.JPG
        We'll strip the leading '/ccapi/ver120/contents/' and prepend '/ccapi/ver100/contents/'.
        """
        prefix = "/ccapi/ver120/contents/"
        if ccapi_path.startswith(prefix):
            sd_path = ccapi_path[len(prefix):]
        else:
            # fallback: if it's already an SD-card-like path, just use as-is
            sd_path = ccapi_path.lstrip("/")
        return f"https://{self.camera_ip}/ccapi/ver100/contents/{sd_path}"

    def _download_fullres_and_replace(self, ccapi_path: str, request_id: int):
        """
        Download full-res JPG for ccapi_path in background, rotate to preview orientation,
        and if still frozen with same request_id, replace the preview texture.
        """
        try:
            url = self._contents_fullres_url(ccapi_path)
            self.log(f"Full-res download (bg): {url}")
            jpg_bytes = self._get_bytes(url, timeout=20.0)
        except Exception as e:
            self.log(f"Full-res download error: {e}")
            return

        try:
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise Exception("cv2.imdecode failed for full-res")

            rot = int(self.preview.preview_rotation) % 360
            if rot == 90:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            elif rot == 270:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb_bytes = rgb.tobytes()
        except Exception as e:
            self.log(f"Full-res decode/rotate error: {e}")
            return

        def _apply(_dt, rgb_bytes=rgb_bytes, w=w, h=h):
            # only apply if still frozen and request matches
            if not self._freeze_active:
                return
            if request_id != self._freeze_request_id:
                return
            if self._freeze_ccapi_path != ccapi_path:
                return

            try:
                tex = Texture.create(size=(w, h), colorfmt="rgb")
                tex.flip_vertical()
                tex.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
                self.preview.set_texture(tex)
                self.log(f"Full-res applied to preview: {ccapi_path}")
            except Exception as e:
                self.log(f"Full-res texture apply error: {e}")

        Clock.schedule_once(_apply, 0)

    # ---------- thumbnails ----------
    def _download_thumb_for_path(self, ccapi_path: str):
        thumb_url = f"https://{self.camera_ip}{ccapi_path}?kind=thumbnail"
        self.log(f"Downloading thumbnail (bg): {thumb_url}")
        try:
            resp = self._session.get(thumb_url, stream=True, timeout=10.0)
            self.log(f"thumb status={resp.status_code} {resp.reason}")
            if resp.status_code != 200:
                return
            thumb_bytes = resp.content
        except Exception as e:
            self.log(f"Thumbnail download error: {e}")
            return

        # Save thumbnail for inspection
        try:
            os.makedirs(self.thumb_dir, exist_ok=True)
            name = os.path.basename(ccapi_path) or "image"
            if not name.lower().endswith((".jpg", ".jpeg")):
                name = name + ".jpg"
            out_path = os.path.join(self.thumb_dir, name)
            with open(out_path, "wb") as f:
                f.write(thumb_bytes)
        except Exception:
            pass

        # Decode using PIL and rotate CCW 90 to match live preview
        try:
            pil = PILImage.open(BytesIO(thumb_bytes)).convert("RGB")
            try:
                pil = pil.transpose(PILImage.Transpose.ROTATE_90)  # CCW
            except Exception:
                pil = pil.transpose(PILImage.ROTATE_90)
            pil.thumbnail((200, 200))
            w, h = pil.size
            rgb_bytes = pil.tobytes()
        except Exception as e:
            self.log(f"Thumbnail decode/rotate err (bg): {e}")
            return

        def _make_texture_and_update(_dt, rgb_bytes=rgb_bytes, w=w, h=h, ccapi_path=ccapi_path):
            try:
                tex = Texture.create(size=(w, h), colorfmt="rgb")
                tex.flip_vertical()
                tex.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
            except Exception as e:
                self.log(f"Texture create/blit err: {e}")
                return

            self._thumb_textures.insert(0, tex)
            self._thumb_paths.insert(0, ccapi_path)
            self._thumb_textures = self._thumb_textures[:5]
            self._thumb_paths = self._thumb_paths[:5]

            for idx, img in enumerate(self._thumb_images):
                if idx < len(self._thumb_textures):
                    img.texture = self._thumb_textures[idx]
                else:
                    img.texture = None

        Clock.schedule_once(_make_texture_and_update, 0)

    def _on_thumb_touch(self, image_widget, touch):
        if not image_widget.collide_point(*touch.pos):
            return False

        idx = getattr(image_widget, "thumb_index", None)
        if idx is None or idx >= len(self._thumb_paths):
            return False

        ccapi_path = self._thumb_paths[idx]
        tex = self._thumb_textures[idx]

        # Freeze preview immediately with thumbnail
        rid = self._freeze_with_texture(ccapi_path, tex)

        # Now fetch full-res JPG in background and replace when done
        if self.capture_type == CaptureType.JPG:
            threading.Thread(
                target=self._download_fullres_and_replace,
                args=(ccapi_path, rid),
                daemon=True
            ).start()
        else:
            self.log(f"Capture type is {self.capture_type}; full-res download not implemented for RAW.")

        return True

    # ---------- content listing + poller ----------
    def list_all_images(self):
        images = []
        st, root = self._json_call("GET", "/ccapi/ver120/contents", None, timeout=8.0)
        self.log(f"/ccapi/ver120/contents -> {st}")
        if not st.startswith("200") or not root or "path" not in root:
            return images

        for path in root["path"]:
            st_dir, dirs = self._json_call("GET", path, None, timeout=8.0)
            if not st_dir.startswith("200") or not dirs or "path" not in dirs:
                continue
            for d in dirs["path"]:
                st_num, num = self._json_call("GET", d + "?kind=number", None, timeout=8.0)
                if not st_num.startswith("200") or not num or "pagenumber" not in num:
                    continue
                pages = int(num["pagenumber"])
                for page in range(1, pages + 1):
                    st_files, f_data = self._json_call("GET", d + f"?page={page}", None, timeout=8.0)
                    if not st_files.startswith("200") or not f_data or "path" not in f_data:
                        continue
                    for f in f_data["path"]:
                        images.append(f)
        return images

    def download_and_thumbnail_latest(self):
        if not self.connected:
            self.log("Not connected; cannot fetch contents.")
            return

        images = self.list_all_images()
        if not images:
            self.log("No images found on camera.")
            return

        jpgs = [p for p in images if p.lower().endswith((".jpg", ".jpeg"))]
        if not jpgs:
            self.log("No JPG files found.")
            return

        latest = jpgs[-1]
        threading.Thread(target=self._download_thumb_for_path, args=(latest,), daemon=True).start()
        self._last_seen_image = latest

    def start_polling_new_images(self):
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self.log(f"Starting image poller every {self.poll_interval_s}s (background thread)")
        self._poll_thread_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)
        self._poll_thread.start()

    def stop_polling_new_images(self):
        if self._poll_thread is None:
            return
        self.log("Stopping image poller (background thread)")
        self._poll_thread_stop.set()
        self._poll_thread = None

    def _poll_worker(self):
        while not self._poll_thread_stop.is_set():
            try:
                images = self.list_all_images()
                if images:
                    jpgs = [p for p in images if p.lower().endswith((".jpg", ".jpeg"))]
                    if jpgs:
                        if self._last_seen_image is None:
                            self._last_seen_image = jpgs[-1]
                            self.log(f"Poll (bg): baseline set to {self._last_seen_image}")
                        else:
                            new_start_idx = None
                            for idx, path in enumerate(jpgs):
                                if path == self._last_seen_image:
                                    new_start_idx = idx + 1
                                    break
                            if new_start_idx is None:
                                self.log("Poll (bg): last_seen not found, resetting baseline")
                                self._last_seen_image = jpgs[-1]
                            else:
                                new_items = jpgs[new_start_idx:]
                                for path in new_items:
                                    self.log(f"Poll (bg): New image detected: {path}")
                                    threading.Thread(
                                        target=self._download_thumb_for_path,
                                        args=(path,),
                                        daemon=True
                                    ).start()
                                    self._last_seen_image = path
            except Exception as e:
                self.log(f"Poll worker error: {e}")

            self._poll_thread_stop.wait(self.poll_interval_s)

    def dump_ccapi(self):
        st, data = self._json_call("GET", "/ccapi", None, timeout=10.0)
        self.log(f"/ccapi status={st}")
        try:
            j = json.dumps(data, indent=2)
        except Exception:
            j = str(data)
        self.log("=== ccapi JSON START ===")
        for line in j.splitlines():
            self.log(line)
        self.log("=== ccapi JSON END ===")

    # ---------- menu / popups ----------
    def _style_menu_button(self, b):
        b.background_normal = ""
        b.background_down = ""
        b.background_color = (0.10, 0.10, 0.10, 0.80)
        b.color = (1, 1, 1, 1)
        return b

    def _build_dropdown(self, reset_callback):
        dd = DropDown(auto_dismiss=True)
        dd.auto_width = False
        dd.width = dp(380)
        dd.max_height = dp(600)

        with dd.canvas.before:
            Color(0.0, 0.0, 0.0, 0.80)
            panel = Rectangle(pos=dd.pos, size=dd.size)
        dd.bind(pos=lambda *_: setattr(panel, "pos", dd.pos), size=lambda *_: setattr(panel, "size", dd.size))

        def add_header(text):
            dd.add_widget(Label(text=text, size_hint_y=None, height=dp(26),
                                font_size=sp(15), color=(1, 1, 1, 1)))

        def add_button(text, on_press):
            b = Button(text=text, size_hint_y=None, height=dp(40), font_size=sp(13))
            self._style_menu_button(b)
            b.bind(on_release=lambda *_: on_press())
            dd.add_widget(b)

        def add_toggle(text, initial, on_change):
            row = BoxLayout(size_hint_y=None, height=dp(32), padding=[dp(6), 0, dp(6), 0])
            row.add_widget(Label(text=text, font_size=sp(13), color=(1, 1, 1, 1)))
            cb = CheckBox(active=initial, size_hint=(None, 1), width=dp(44))
            cb.bind(active=lambda inst, val: on_change(val))
            row.add_widget(cb)
            dd.add_widget(row)

        def add_capture_type_buttons():
            row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4), padding=[dp(4), 0, dp(4), 0])
            row.add_widget(Label(text="Capture:", size_hint=(None, 1), width=dp(70),
                                 font_size=sp(13), color=(1, 1, 1, 1)))

            def make_btn(label, ctype):
                b = Button(text=label, size_hint=(1, 1), font_size=sp(12))
                self._style_menu_button(b)

                def set_type():
                    self.capture_type = ctype
                    self.log(f"Capture type set to {ctype}")

                b.bind(on_release=lambda *_: set_type())
                return b

            row.add_widget(make_btn("JPG", CaptureType.JPG))
            row.add_widget(make_btn("RAW", CaptureType.RAW))
            dd.add_widget(row)

        add_header("Framing")
        add_button("Reset framing", reset_callback)

        add_header("Overlays")
        add_toggle("Border (blue)", True, lambda v: setattr(self.preview, "show_border", v))
        add_toggle("Grid (orange)", True, lambda v: setattr(self.preview, "show_grid", v))
        add_toggle("Crop 5:7 (red)", True, lambda v: setattr(self.preview, "show_57", v))
        add_toggle("Crop 8:10 (yellow)", True, lambda v: setattr(self.preview, "show_810", v))
        add_toggle("Oval (purple)", True, lambda v: setattr(self.preview, "show_oval", v))

        add_header("Network")
        add_button("Set camera IP…", lambda: self._open_ip_popup())

        add_header("Display")
        add_button("Set display FPS…", lambda: self._open_fps_popup())

        add_header("Author")
        add_button("Refresh Author now", lambda: self.refresh_author(note="manual"))
        add_button("Load CSV…", lambda: self._open_csv_filechooser())
        add_button("Select headers…", lambda: self._open_headers_popup())

        add_header("Capture")
        add_capture_type_buttons()
        add_button("Fetch latest image", lambda: self.download_and_thumbnail_latest())
        add_button("Start auto-fetch", lambda: self.start_polling_new_images())
        add_button("Stop auto-fetch", lambda: self.stop_polling_new_images())

        add_header("Debug")
        add_button("Dump /ccapi", lambda: self.dump_ccapi())
        add_button("Show log…", lambda: self._open_log_popup())

        return dd

    def _open_log_popup(self):
        if self._log_popup is not None:
            try:
                self._log_popup.dismiss()
            except Exception:
                pass
            self._log_popup = None

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        lbl = Label(text="\n".join(self._log_lines), size_hint_y=None, halign="left", valign="top",
                    font_size=sp(11))
        lbl.bind(width=lambda *_: setattr(lbl, "text_size", (lbl.width, None)))
        lbl.bind(texture_size=lambda *_: setattr(lbl, "height", lbl.texture_size[1]))
        sv.add_widget(lbl)
        root.add_widget(sv)

        btn = Button(text="Close", size_hint=(1, None), height=dp(44))
        root.add_widget(btn)

        popup = Popup(title="Log", content=root, size_hint=(0.95, 0.95))
        btn.bind(on_release=lambda *_: popup.dismiss())
        popup.bind(on_dismiss=lambda *_: self._clear_log_popup())
        popup.open()

        self._log_popup = popup
        self._log_popup_label = lbl

    def _clear_log_popup(self):
        self._log_popup = None
        self._log_popup_label = None

    def _open_ip_popup(self):
        if self._ip_popup is not None:
            try:
                self._ip_popup.dismiss()
            except Exception:
                pass
            self._ip_popup = None

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        root.add_widget(Label(text="Camera IP:", size_hint=(1, None), height=dp(28), font_size=sp(12)))
        ip_in = TextInput(text=self.camera_ip, multiline=False, font_size=sp(16),
                          size_hint=(1, None), height=dp(44))
        root.add_widget(ip_in)

        btns = BoxLayout(size_hint=(1, None), height=dp(44), spacing=dp(6))
        ok = Button(text="OK")
        cancel = Button(text="Cancel")
        btns.add_widget(ok)
        btns.add_widget(cancel)
        root.add_widget(btns)

        popup = Popup(title="Set Camera IP", content=root, size_hint=(0.9, 0.4))

        def do_ok(*_):
            new_ip = ip_in.text.strip()
            if new_ip:
                self.camera_ip = new_ip
                self.log(f"Camera IP set to {self.camera_ip}")
                if self.connected:
                    self.author_label.text = "(IP changed; reconnect)"
            popup.dismiss()

        ok.bind(on_release=do_ok)
        cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.bind(on_dismiss=lambda *_: setattr(self, "_ip_popup", None))
        popup.open()
        self._ip_popup = popup

    def _open_fps_popup(self):
        if self._fps_popup is not None:
            try:
                self._fps_popup.dismiss()
            except Exception:
                pass
            self._fps_popup = None

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        root.add_widget(Label(text="Display FPS (metrics tick)", size_hint=(1, None), height=dp(28),
                              font_size=sp(12)))

        slider = Slider(min=5, max=30, value=12, step=1)
        val_lbl = Label(text="12", size_hint=(1, None), height=dp(22), font_size=sp(12))
        root.add_widget(slider)
        root.add_widget(val_lbl)
        slider.bind(value=lambda inst, v: setattr(val_lbl, "text", str(int(v))))

        btns = BoxLayout(size_hint=(1, None), height=dp(44), spacing=dp(6))
        ok = Button(text="OK")
        cancel = Button(text="Cancel")
        btns.add_widget(ok)
        btns.add_widget(cancel)
        root.add_widget(btns)

        popup = Popup(title="Set Display FPS", content=root, size_hint=(0.9, 0.5))

        def do_ok(*_):
            fps = int(slider.value)
            self._reschedule_display_loop(fps)
            self.log(f"Display FPS set to {fps}")
            popup.dismiss()

        ok.bind(on_release=do_ok)
        cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.bind(on_dismiss=lambda *_: setattr(self, "_fps_popup", None))
        popup.open()
        self._fps_popup = popup

    # ---------- Android SAF CSV ----------
    def _bind_android_activity_once(self):
        if getattr(self, "_android_activity_bound", False):
            return
        try:
            from android import activity
            activity.bind(on_activity_result=self._on_android_activity_result)
            self._android_activity_bound = True
        except Exception as e:
            self.log(f"Android activity bind failed: {e}")

    def _open_csv_saf(self):
        self._bind_android_activity_once()
        self._csv_req_code = getattr(self, "_csv_req_code", 4242)

        try:
            from android import mActivity
            from jnius import autoclass

            Intent = autoclass("android.content.Intent")

            intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            intent.setType("*/*")

            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)

            self.log("Opening Android file picker…")
            mActivity.startActivityForResult(intent, self._csv_req_code)

        except Exception as e:
            self.log(f"Failed to open Android picker: {e}")

    def _on_android_activity_result(self, request_code, result_code, intent):
        if request_code != getattr(self, "_csv_req_code", 4242):
            return

        if result_code != -1 or intent is None:
            self.log("CSV picker canceled")
            return

        try:
            from android import mActivity
            from jnius import cast, autoclass

            Intent = autoclass("android.content.Intent")
            uri = cast("android.net.Uri", intent.getData())
            if uri is None:
                self.log("CSV picker returned no URI")
                return

            try:
                flags = intent.getFlags()
                take_flags = flags & (
                    Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                )
                mActivity.getContentResolver().takePersistableUriPermission(uri, take_flags)
            except Exception:
                pass

            data = self._read_android_uri_bytes(uri)
            self._parse_csv_bytes(data)
            self.log(f"CSV loaded from picker: {len(self.csv_rows)} rows")

        except Exception as e:
            self.log(f"CSV load failed (Android): {e}")

    def _read_android_uri_bytes(self, uri):
        from android import mActivity

        cr = mActivity.getContentResolver()
        stream = cr.openInputStream(uri)
        if stream is None:
            raise Exception("openInputStream() returned null")

        out = bytearray()
        buf = bytearray(64 * 1024)
        while True:
            n = stream.read(buf)
            if n == -1 or n == 0:
                break
            out.extend(buf[:n])
        stream.close()
        return bytes(out)

    def _open_csv_filechooser(self):
        if platform != "android":
            self.log("CSV load is Android-only (SAF). Please run on-device to load CSV.")
            return
        return self._open_csv_saf()

    def _parse_csv_bytes(self, b: bytes):
        self.log(f"CSV size: {len(b)} bytes")
        try:
            text = b.decode("utf-8-sig")
        except Exception:
            text = b.decode("latin-1", errors="replace")

        reader = csv.DictReader(text.splitlines())
        headers = reader.fieldnames or []
        self.csv_headers = headers
        rows = []
        for r in reader:
            rows.append({k: (r.get(k) or "").strip() for k in headers})
        self.csv_rows = rows
        self.log(f"CSV headers: {headers}")
        self.log(f"CSV rows: {len(rows)}")

        preferred = ["LAST_NAME", "FIRST_NAME", "GRADE", "TEACHER", "STUDENT_ID"]
        self.selected_headers = [h for h in preferred if h in headers]
        if not self.selected_headers and headers:
            self.selected_headers = headers[:3]

    def _open_headers_popup(self):
        if not self.csv_headers:
            self.log("No CSV loaded; cannot pick headers")
            return

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        root.add_widget(Label(text="Select columns to include in Author (joined with _):",
                              size_hint=(1, None), height=dp(40), font_size=sp(12)))

        sv = ScrollView(size_hint=(1, 1))
        inner = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        inner.bind(minimum_height=inner.setter("height"))
        sv.add_widget(inner)

        current_sel = set(self.selected_headers)

        for h in self.csv_headers:
            row = BoxLayout(size_hint_y=None, height=dp(28))
            lbl = Label(text=h, size_hint=(0.7, 1), font_size=sp(12), halign="left", valign="middle")
            lbl.bind(size=lbl.setter("text_size"))
            cb = CheckBox(active=(h in current_sel), size_hint=(0.3, 1))

            def toggle_cb(inst, val, header=h):
                if val:
                    if header not in self.selected_headers:
                        self.selected_headers.append(header)
                else:
                    if header in self.selected_headers:
                        self.selected_headers.remove(header)

            cb.bind(active=toggle_cb)
            row.add_widget(lbl)
            row.add_widget(cb)
            inner.add_widget(row)

        root.add_widget(sv)

        btns = BoxLayout(size_hint=(1, None), height=dp(36), spacing=dp(6))
        btn_ok = Button(text="OK")
        btn_cancel = Button(text="Cancel")
        btns.add_widget(btn_ok)
        btns.add_widget(btn_cancel)
        root.add_widget(btns)

        popup = Popup(title="Select CSV columns", content=root, size_hint=(0.9, 0.9))

        def do_ok(*_):
            self.log(f"Selected headers: {self.selected_headers}")
            popup.dismiss()

        btn_ok.bind(on_release=do_ok)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())

        popup.open()
        self._headers_popup = popup

    def on_stop(self):
        try:
            self.stop_liveview()
        except Exception:
            pass
        try:
            self.stop_polling_new_images()
        except Exception:
            pass
        try:
            self._decoder_stop.set()
        except Exception:
            pass


if __name__ == "__main__":
    VolumeToolkitApp().run()
