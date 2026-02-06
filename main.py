# Android-focused Volume Toolkit (threaded decoder + background poller)
#
# v2.1.8
#
# Base: v2.1.7
#
# Fixes:
# - Force RV row heights in data + view to prevent white rectangles.
# - Refresh RV layout + data on row size change and insert.
# - Middle band height computed from remaining space (no overlap with bottom buttons).
#
import os
import threading
import time
from datetime import datetime
from io import BytesIO
import csv
import queue
import hashlib
from urllib.parse import quote

import requests
import urllib3

import kivy
kivy.require("2.0.0")

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.properties import NumericProperty, BooleanProperty, StringProperty, ObjectProperty, ListProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.utils import platform

# RecycleView for scrollable unlimited thumb list
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import ButtonBehavior

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
        if frame_w <= 0 or frame_h <= 0:
            return (frame_x, frame_y, 0, 0)
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

        if self.show_grid and n >= 2 and fw > 0 and fh > 0:
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

        if self.show_oval and fw > 0 and fh > 0:
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


class ThumbRow(ButtonBehavior, Image):
    """
    RecycleView row: clickable thumbnail + border highlight.
    Data keys expected:
      - index: int
      - local_path: str
      - ccapi_path: str
      - active: bool
      - height: float
    """
    app = ObjectProperty(None)
    index = NumericProperty(-1)
    local_path = StringProperty("")
    ccapi_path = StringProperty("")
    active = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_stretch = True
        self.keep_ratio = True
        self.size_hint_y = None
        self.color = (1, 1, 1, 1)

        with self.canvas.after:
            self._c = Color(0.2, 1.0, 0.2, 1.0)
            self._ln = Line(rectangle=(0, 0, 0, 0), width=3)

        self.bind(pos=self._redraw_border, size=self._redraw_border, active=self._redraw_border)

    def refresh_view_attrs(self, rv, index, data):
        self.app = data.get("app")
        self.index = int(data.get("index", -1))
        self.local_path = data.get("local_path", "")
        self.ccapi_path = data.get("ccapi_path", "")
        self.active = bool(data.get("active", False))
        self.height = float(data.get("height", self.height or dp(90)))

        tex = None
        if self.app is not None and self.local_path:
            tex = self.app._load_thumb_texture_from_file(self.local_path)
        self.texture = tex

        self._redraw_border()
        return super().refresh_view_attrs(rv, index, data)

    def _redraw_border(self, *_):
        if self.active:
            x, y = self.pos
            w, h = self.size
            self._ln.rectangle = (x, y, w, h)
        else:
            self._ln.rectangle = (0, 0, 0, 0)

    def on_release(self):
        if self.app is not None:
            self.app.on_thumb_index_pressed(self.index)


class ThumbRV(RecycleView):
    pass


class VolumeToolkitApp(App):
    capture_type = StringProperty(CaptureType.JPG)
    STILL_TARGET_ASPECT = 2.0 / 3.0  # portrait

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.connected = False
        self.camera_ip = "172.25.162.76"

        self.live_running = False
        self.session_started = False

        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._last_decoded_ts = 0.0

        self._fetch_thread = None
        self._display_event = None

        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        self._log_lines = []
        self._max_log_lines = 500

        self._frame_texture = None
        self._frame_size = None

        # QR decode
        self.qr_enabled = False
        self.qr_interval_s = 0.15
        self.qr_new_gate_s = 0.70
        self._qr_detector = cv2.QRCodeDetector()
        self._qr_thread = None
        self._qr_seen = set()
        self._qr_last_add_time = 0.0

        self._latest_decoded_bgr = None
        self._latest_decoded_bgr_ts = 0.0

        # decoder thread
        self._decode_queue = queue.Queue(maxsize=2)
        self._decoder_thread = threading.Thread(target=self._decoder_loop, daemon=True)
        self._decoder_stop = threading.Event()
        self._decoder_thread.start()

        # author push
        self.author_max_chars = 60
        self._last_committed_author = None
        self._author_update_in_flight = False

        self._current_payload = ""   # QR payload
        self._csv_payload = ""       # CSV payload
        self._current_exif = ""      # EXIF author

        # CSV data
        self.csv_headers = []
        self.csv_rows = []
        self.selected_headers = []
        self._headers_popup = None
        self._subject_popup = None

        # thumbnails (history)
        self._thumb_items = []  # newest first; dict: {ccapi_path, local_path}
        self.thumb_dir = os.path.join(self.user_data_dir, "thumbs")
        self._active_thumb_index = None
        self._thumb_items_max = 2000
        self._thumb_row_height = dp(90)

        # thumb texture cache
        self._thumb_tex_cache = {}       # local_path -> Texture
        self._thumb_tex_cache_order = [] # oldest first
        self._thumb_tex_cache_max = 40

        # poller
        self._last_seen_image = None
        self._poll_thread = None
        self._poll_thread_stop = threading.Event()
        self.poll_interval_s = 2.0
        self.autofetch_enabled = False

        # HTTP session
        self._session = requests.Session()
        self._session.verify = False

        # Android SAF
        self._android_activity_bound = False
        self._csv_req_code = 4242

        # popups
        self.dropdown = None
        self._ip_popup = None
        self._fps_popup = None
        self._metrics_popup = None

        # freeze (still) mode
        self._freeze_active = False
        self._freeze_ccapi_path = None
        self._freeze_request_id = 0

        # Log overlay (bottom)
        self._log_overlay_visible = False
        self._log_overlay_label = None
        self._log_overlay_sv = None

        # RV
        self.thumb_rv = None

    # ---------- styling helpers ----------
    @staticmethod
    def _set_btn_style(btn: Button, bg_rgba, fg_rgba):
        btn.background_normal = ""
        btn.background_down = ""
        btn.background_disabled_normal = ""
        btn.background_color = bg_rgba
        btn.color = fg_rgba
        btn.disabled_color = fg_rgba

    def _apply_connect_btn_style(self):
        self._set_btn_style(self.connect_btn, (0.10, 0.35, 0.85, 1.0), (1, 1, 1, 1))
        if self.connected:
            self.connect_btn.text = "Connected"
            self.connect_btn.color = (1.0, 1.0, 0.0, 1.0)
        else:
            self.connect_btn.text = "Connect"
            self.connect_btn.color = (1, 1, 1, 1)

    def _apply_live_btn_style(self):
        if self.live_running:
            self.start_btn.text = "Live View On"
            self._set_btn_style(self.start_btn, (0.15, 0.65, 0.20, 1.0), (1.0, 1.0, 0.0, 1.0))
        else:
            self.start_btn.text = "Live View Off"
            self._set_btn_style(self.start_btn, (0.80, 0.15, 0.15, 1.0), (1, 1, 1, 1))

    def _apply_autofetch_btn_style(self):
        if self.autofetch_enabled:
            self.autofetch_btn.text = "Autofetch On"
            self._set_btn_style(self.autofetch_btn, (0.15, 0.65, 0.20, 1.0), (1.0, 1.0, 0.0, 1.0))
        else:
            self.autofetch_btn.text = "Autofetch Off"
            self._set_btn_style(self.autofetch_btn, (0.80, 0.15, 0.15, 1.0), (1, 1, 1, 1))

    def _apply_qr_btn_style(self):
        if self.qr_enabled:
            self.qr_btn.text = "QR Detect On"
            self._set_btn_style(self.qr_btn, (0.15, 0.65, 0.20, 1.0), (1.0, 1.0, 0.0, 1.0))
        else:
            self.qr_btn.text = "QR Detect Off"
            self._set_btn_style(self.qr_btn, (0.80, 0.15, 0.15, 1.0), (1, 1, 1, 1))

    # ---------- lifecycle ----------
    def on_start(self):
        if platform == "android":
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])
                self.log("Requested storage permissions (Android).")
            except Exception as e:
                self.log(f"Permission request failed: {e}")

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

    # ---------- app exit ----------
    def exit_app(self):
        try:
            self.stop_liveview()
        except Exception:
            pass
        try:
            self.stop_polling_new_images()
        except Exception:
            pass
        self.stop()

    # ---------- log overlay ----------
    def _set_log_overlay_visible(self, visible: bool):
        self._log_overlay_visible = bool(visible)
        if not hasattr(self, "log_overlay"):
            return
        self.log_overlay.opacity = 1.0 if self._log_overlay_visible else 0.0
        self.log_overlay.disabled = not self._log_overlay_visible
        self.log_overlay.height = dp(220) if self._log_overlay_visible else 0

    def _append_log_overlay(self):
        if self._log_overlay_label is None:
            return
        self._log_overlay_label.text = "\n".join(self._log_lines)
        if self._log_overlay_sv is not None:
            Clock.schedule_once(lambda *_: setattr(self._log_overlay_sv, "scroll_y", 0.0), 0)

    # ---------- logging ----------
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with self._log_lock:
            self._log_lines.append(line)
            if len(self._log_lines) > self._max_log_lines:
                self._log_lines = self._log_lines[-self._max_log_lines:]
        Clock.schedule_once(lambda *_: self._append_log_overlay(), 0)

    def _clear_log(self):
        with self._log_lock:
            self._log_lines = []
        Clock.schedule_once(lambda *_: self._append_log_overlay(), 0)

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

    # ---------- matching colors ----------
    def _match_trimmed(self, s: str) -> str:
        return (s or "").strip()

    def _apply_match_colors(self):
        E = self._match_trimmed(self._current_exif)
        Q = self._match_trimmed(self._current_payload)
        C = self._match_trimmed(self._csv_payload)

        GREEN = (0.2, 1.0, 0.2, 1.0)
        RED = (1.0, 0.2, 0.2, 1.0)
        WHITE = (1.0, 1.0, 1.0, 1.0)

        e_green = False
        q_green = False
        c_green = False

        if E and C and E == C:
            e_green = True
            c_green = True
        if E and Q and E == Q:
            e_green = True
            q_green = True

        if e_green:
            self.exif_label.color = GREEN
        else:
            if E and ((C and E != C) or (Q and E != Q)):
                self.exif_label.color = RED
            else:
                self.exif_label.color = WHITE

        if c_green:
            self.csv_payload_label.color = GREEN
        else:
            if E and C and (E != C) and (E == Q) and Q:
                self.csv_payload_label.color = RED
            else:
                self.csv_payload_label.color = WHITE

        if q_green:
            self.payload_label.color = GREEN
        else:
            if E and Q and (E != Q) and (E == C) and C:
                self.payload_label.color = RED
            else:
                self.payload_label.color = WHITE

    # ---------- EXIF refresh ----------
    def refresh_exif(self):
        if not self.connected:
            Clock.schedule_once(lambda *_: self._set_exif_text("(not connected)"), 0)
            return

        def worker():
            st, data = self._json_call("GET", "/ccapi/ver100/functions/registeredname/author", None, timeout=8.0)
            if st.startswith("200") and isinstance(data, dict):
                exif = (data.get("author") or "").strip()
            else:
                exif = f"(read failed: {st})"
            Clock.schedule_once(lambda *_: self._set_exif_text(exif if exif else "(empty)"), 0)

        threading.Thread(target=worker, daemon=True).start()

    def _set_exif_text(self, exif: str):
        self._current_exif = exif or ""
        self.exif_label.text = self._current_exif
        self._apply_match_colors()

    # ---------- payload setters ----------
    def _set_payload(self, payload: str):
        payload = (payload or "").strip()
        self._current_payload = payload
        self.payload_label.text = payload if payload else "(none)"
        self._apply_match_colors()

    def _set_csv_payload(self, payload: str):
        payload = (payload or "").strip()
        self._csv_payload = payload
        self.csv_payload_label.text = payload if payload else "(none)"
        self._apply_match_colors()

    # ---------- connect / controls ----------
    def _set_controls_idle(self):
        self.connect_btn.disabled = False
        self.start_btn.disabled = not self.connected
        self.autofetch_btn.disabled = not self.connected
        self.qr_btn.disabled = not self.connected
        self.push_btn.disabled = not self.connected
        self.subject_btn.disabled = not (self.connected and bool(self.csv_rows))

        self._apply_connect_btn_style()
        self._apply_live_btn_style()
        self._apply_autofetch_btn_style()
        self._apply_qr_btn_style()

    def _set_controls_running(self):
        self.connect_btn.disabled = True
        self.start_btn.disabled = False
        self.autofetch_btn.disabled = False
        self.qr_btn.disabled = False
        self.push_btn.disabled = False
        self.subject_btn.disabled = not bool(self.csv_rows)

        self._apply_connect_btn_style()
        self._apply_live_btn_style()
        self._apply_autofetch_btn_style()
        self._apply_qr_btn_style()

    def connect_camera(self):
        if self.live_running:
            self.log("Connect disabled while live view is running. Stop live view first.")
            return

        self.log(f"Connecting to {self.camera_ip}:443")
        st, data = self._json_call("GET", "/ccapi/ver100/deviceinformation", None, timeout=8.0)
        if st.startswith("200") and data:
            self.connected = True
            self.log("Connected OK")
            self._set_exif_text("(loading...)")
            self.refresh_exif()
        else:
            self.connected = False
            self._set_exif_text(f"(connect failed: {st})")
            self.log(f"Connect failed: {st}")

        self._set_controls_idle()

    # ---------- live view ----------
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

    # ---------- AutoFetch ----------
    def toggle_autofetch(self):
        if not self.connected:
            return

        self.autofetch_enabled = not bool(self.autofetch_enabled)
        self._apply_autofetch_btn_style()

        if self.autofetch_enabled:
            self._last_seen_image = None
            self.log("Autofetch enabled: setting baseline to latest (no backlog)")
            threading.Thread(target=self._autofetch_set_baseline_worker, daemon=True).start()
            self.start_polling_new_images()
        else:
            self.log("Autofetch disabled")
            self.stop_polling_new_images()

    def _autofetch_set_baseline_worker(self):
        try:
            images = self.list_all_images()
            jpgs = [p for p in images if p.lower().endswith((".jpg", ".jpeg"))]
            if jpgs:
                self._last_seen_image = jpgs[-1]
                self.log(f"Autofetch baseline set to {self._last_seen_image}")
        except Exception as e:
            self.log(f"Autofetch baseline error: {e}")

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
                if not self.autofetch_enabled:
                    self._poll_thread_stop.wait(self.poll_interval_s)
                    continue

                images = self.list_all_images()
                jpgs = [p for p in images if p.lower().endswith((".jpg", ".jpeg"))]
                if not jpgs:
                    self._poll_thread_stop.wait(self.poll_interval_s)
                    continue

                if self._last_seen_image is None:
                    self._last_seen_image = jpgs[-1]
                    self.log(f"Poll (bg): baseline set to {self._last_seen_image}")
                    self._poll_thread_stop.wait(self.poll_interval_s)
                    continue

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
                        threading.Thread(target=self._download_thumb_for_path, args=(path,), daemon=True).start()
                        self._last_seen_image = path

            except Exception as e:
                self.log(f"Poll worker error: {e}")

            self._poll_thread_stop.wait(self.poll_interval_s)

    # ---------- QR ----------
    def toggle_qr_detect(self):
        self.qr_enabled = not bool(self.qr_enabled)
        self._apply_qr_btn_style()
        self.log(f"QR detect {'enabled' if self.qr_enabled else 'disabled'}")

    # ---------- push payload chooser ----------
    def _author_value(self, payload: str) -> str:
        s = (payload or "").strip()
        if not s:
            return ""
        return s[: int(self.author_max_chars)]

    def push_payload(self):
        if not self.connected:
            self.log("Push payload skipped: not connected")
            return
        if self._author_update_in_flight:
            self.log("Push payload skipped: update in flight")
            return

        qr = (self._current_payload or "").strip()
        csvp = (self._csv_payload or "").strip()

        options = []
        if qr:
            options.append(("QR", qr))
        if csvp:
            options.append(("CSV", csvp))

        if not options:
            self.log("Push payload skipped: no QR or CSV payload available")
            return

        if len(options) == 1:
            label, value = options[0]
            self._push_author_value(value, source=label)
            return

        root = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))
        root.add_widget(Label(text="Choose payload to push:", size_hint=(1, None), height=dp(30), font_size=sp(14)))

        popup = Popup(title="Push Payload", content=root, size_hint=(0.95, 0.6))

        for label, value in options:
            preview = value[:120] + ("…" if len(value) > 120 else "")
            btn = Button(text=f"{label}: {preview}", size_hint=(1, None), height=dp(48))

            def _make_onpress(lbl=label, val=value):
                return lambda *_: (popup.dismiss(), self._push_author_value(val, source=lbl))

            btn.bind(on_release=_make_onpress())
            root.add_widget(btn)

        cancel = Button(text="Cancel", size_hint=(1, None), height=dp(44))
        cancel.bind(on_release=lambda *_: popup.dismiss())
        root.add_widget(cancel)

        popup.open()

    def _push_author_value(self, raw_value: str, source="manual"):
        value = self._author_value(raw_value)
        if not value:
            self.log(f"Push payload skipped ({source}): empty after trim")
            return
        if self._last_committed_author == value:
            self.log(f"Push payload skipped ({source}): already pushed")
            return

        self._author_update_in_flight = True
        self.log(f"Pushing payload ({source}) to Author: '{value}'")
        threading.Thread(target=self._commit_author_worker, args=(value, source), daemon=True).start()

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
            self.refresh_exif()

        Clock.schedule_once(_finish, 0)

    # ---------- metrics loop ----------
    def _reschedule_display_loop(self, fps):
        if self._display_event is not None:
            self._display_event.cancel()
        fps = max(1, int(fps))
        self._display_event = Clock.schedule_interval(self._display_tick, 1.0 / fps)

    def _display_tick(self, dt):
        self._display_count += 1
        self._update_metrics(self._last_decoded_ts)

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

    def _open_metrics_popup(self):
        if self._metrics_popup is not None:
            try:
                self._metrics_popup.dismiss()
            except Exception:
                pass
            self._metrics_popup = None

        root = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))
        lbl = Label(text=self.metrics.text, font_size=sp(12))
        root.add_widget(lbl)

        def _update_lbl(_dt):
            lbl.text = self.metrics.text
        ev = Clock.schedule_interval(_update_lbl, 0.25)

        close = Button(text="Close", size_hint=(1, None), height=dp(44))
        root.add_widget(close)

        popup = Popup(title="Live Metrics", content=root, size_hint=(0.9, 0.35))

        def _close(*_):
            try:
                ev.cancel()
            except Exception:
                pass
            popup.dismiss()

        close.bind(on_release=_close)
        popup.bind(on_dismiss=lambda *_: setattr(self, "_metrics_popup", None))
        popup.open()
        self._metrics_popup = popup

    # ---------- live fetch/decoder/qr loops ----------
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

    def _rotate_bgr(self, bgr):
        rot = int(self.preview.preview_rotation) % 360
        if rot == 90:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if rot == 180:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        if rot == 270:
            return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return bgr

    @staticmethod
    def _center_crop_bgr_to_aspect(bgr, target_aspect_w_over_h: float):
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        if h <= 0 or w <= 0:
            return bgr
        src_aspect = float(w) / float(h)
        tgt = float(target_aspect_w_over_h)
        if abs(src_aspect - tgt) < 1e-3:
            return bgr

        if src_aspect > tgt:
            new_w = int(h * tgt)
            x0 = max(0, (w - new_w) // 2)
            return bgr[:, x0:x0 + new_w]
        else:
            new_h = int(w / tgt)
            y0 = max(0, (h - new_h) // 2)
            return bgr[y0:y0 + new_h, :]

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

                bgr = self._rotate_bgr(bgr)

                with self._lock:
                    self._latest_decoded_bgr = bgr.copy()
                    self._latest_decoded_bgr_ts = ts

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                rgb_bytes = rgb.tobytes()

                def _update_texture_on_main(_dt, rgb_bytes=rgb_bytes, w=w, h=h, ts=ts):
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

    # ---------- thumb file naming + cache ----------
    def _thumb_local_path_for_ccapi_path(self, ccapi_path: str) -> str:
        base = os.path.basename(ccapi_path) or "image"
        if not base.lower().endswith((".jpg", ".jpeg")):
            base = base + ".jpg"
        safe = quote(ccapi_path, safe="")
        h = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:16]
        name = f"{h}_{base}"
        return os.path.join(self.thumb_dir, name)

    def _thumb_cache_put(self, local_path: str, tex: Texture):
        self._thumb_tex_cache[local_path] = tex
        if local_path in self._thumb_tex_cache_order:
            self._thumb_tex_cache_order.remove(local_path)
        self._thumb_tex_cache_order.append(local_path)

        while len(self._thumb_tex_cache_order) > int(self._thumb_tex_cache_max):
            old = self._thumb_tex_cache_order.pop(0)
            try:
                del self._thumb_tex_cache[old]
            except Exception:
                pass

    def _thumb_cache_get(self, local_path: str):
        tex = self._thumb_tex_cache.get(local_path)
        if tex is not None:
            if local_path in self._thumb_tex_cache_order:
                self._thumb_tex_cache_order.remove(local_path)
            self._thumb_tex_cache_order.append(local_path)
        return tex

    def _load_thumb_texture_from_file(self, local_path: str):
        tex = self._thumb_cache_get(local_path)
        if tex is not None:
            return tex
        if not local_path or not os.path.exists(local_path):
            return None
        try:
            pil = PILImage.open(local_path).convert("RGB")
            pil.thumbnail((240, 240))
            w, h = pil.size
            rgb_bytes = pil.tobytes()
            tex = Texture.create(size=(w, h), colorfmt="rgb")
            tex.flip_vertical()
            tex.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
            self._thumb_cache_put(local_path, tex)
            return tex
        except Exception as e:
            self.log(f"thumb file decode error: {e}")
            return None

    def _rv_rebuild_data(self):
        if self.thumb_rv is None:
            return
        data = []
        with self._state_lock:
            active_idx = self._active_thumb_index
            freeze_active = self._freeze_active
            items = list(self._thumb_items)
            row_h = float(self._thumb_row_height)
        for i, item in enumerate(items):
            data.append({
                "app": self,
                "index": i,
                "local_path": item.get("local_path", ""),
                "ccapi_path": item.get("ccapi_path", ""),
                "active": (active_idx == i and freeze_active),
                "height": row_h,
            })
        self.thumb_rv.data = data
        self.thumb_rv.refresh_from_data()

    def _rv_rebuild_data_main(self):
        Clock.schedule_once(lambda *_: self._rv_rebuild_data(), 0)

    # ---------- review toggle ----------
    def _close_review(self):
        with self._state_lock:
            self._freeze_active = False
            self._freeze_ccapi_path = None
            self._freeze_request_id += 1
            self._active_thumb_index = None
        self._rv_rebuild_data_main()
        self.log("Review closed (thumb toggle)")

    def on_thumb_index_pressed(self, idx: int):
        with self._state_lock:
            if idx < 0 or idx >= len(self._thumb_items):
                return
            if self._freeze_active and self._active_thumb_index == idx:
                do_close = True
            else:
                do_close = False

        if do_close:
            self._close_review()
            return

        with self._state_lock:
            item = self._thumb_items[idx]
            ccapi_path = item.get("ccapi_path", "")
            local_path = item.get("local_path", "")

            rid = self._freeze_request_id + 1
            self._freeze_active = True
            self._freeze_ccapi_path = ccapi_path
            self._freeze_request_id = rid
            self._active_thumb_index = idx

        self._rv_rebuild_data_main()
        self.log(f"REVIEW START idx={idx} rid={rid} path={ccapi_path}")

        tex = self._load_thumb_texture_from_file(local_path)
        if tex is not None:
            self.preview.set_texture(tex)

        threading.Thread(target=self._freeze_pipeline_for_thumb, args=(ccapi_path, rid), daemon=True).start()

    # ---------- endpoints ----------
    def _contents_fullres_url(self, ccapi_path: str) -> str:
        prefix = "/ccapi/ver120/contents/"
        if ccapi_path.startswith(prefix):
            sd_path = ccapi_path[len(prefix):]
        else:
            sd_path = ccapi_path.lstrip("/")
        sd_path_enc = quote(sd_path, safe="/")
        return f"https://{self.camera_ip}/ccapi/ver100/contents/{sd_path_enc}"

    def _download_fullres_and_replace(self, ccapi_path: str, request_id: int):
        urls = [
            ("PRIMARY", f"https://{self.camera_ip}{ccapi_path}"),
            ("FALLBACK", self._contents_fullres_url(ccapi_path)),
        ]

        jpg_bytes = None
        used = None

        for name, url in urls:
            try:
                self.log(f"FULLRES {name} START url={url}")
                resp = self._session.get(url, timeout=25.0, stream=True)
                if resp.status_code != 200 or not resp.content:
                    self.log(f"FULLRES {name} FAIL status={resp.status_code} bytes={len(resp.content) if resp.content else 0}")
                    continue
                jpg_bytes = resp.content
                used = name
                self.log(f"FULLRES {name} OK bytes={len(jpg_bytes)}")
                break
            except Exception as e:
                self.log(f"FULLRES {name} ERROR {e}")

        if jpg_bytes is None:
            self.log("FULLRES GIVEUP (no endpoint succeeded)")
            return

        try:
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise Exception("cv2.imdecode failed for full-res")

            bgr = self._rotate_bgr(bgr)
            bgr = self._center_crop_bgr_to_aspect(bgr, self.STILL_TARGET_ASPECT)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb_bytes = rgb.tobytes()
        except Exception as e:
            self.log(f"FULLRES {used} DECODE/ROTATE/CROP ERROR: {e}")
            return

        def _apply(_dt, rgb_bytes=rgb_bytes, w=w, h=h):
            with self._state_lock:
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
                self.log(f"FULLRES {used} APPLY w={w} h={h}")
            except Exception as e:
                self.log(f"FULLRES {used} APPLY ERROR: {e}")

        Clock.schedule_once(_apply, 0)

    # ---------- thumbnail download (sidebar) ----------
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

        try:
            os.makedirs(self.thumb_dir, exist_ok=True)
            local_path = self._thumb_local_path_for_ccapi_path(ccapi_path)
            with open(local_path, "wb") as f:
                f.write(thumb_bytes)
        except Exception as e:
            self.log(f"Thumbnail save error: {e}")
            return

        def _add_item(_dt):
            with self._state_lock:
                self._thumb_items.insert(0, {"ccapi_path": ccapi_path, "local_path": local_path})
                if self._active_thumb_index is not None:
                    self._active_thumb_index += 1
                if len(self._thumb_items) > self._thumb_items_max:
                    self._thumb_items = self._thumb_items[:self._thumb_items_max]
            self._rv_rebuild_data()

        Clock.schedule_once(_add_item, 0)

    def _freeze_pipeline_for_thumb(self, ccapi_path: str, request_id: int):
        thumb_url = f"https://{self.camera_ip}{ccapi_path}?kind=thumbnail"
        try:
            self.log(f"REVIEW THUMB START url={thumb_url}")
            b = self._get_bytes(thumb_url, timeout=12.0)
            self.log(f"REVIEW THUMB OK bytes={len(b)}")
        except Exception as e:
            self.log(f"REVIEW THUMB ERROR {e}")
            return

        try:
            arr = np.frombuffer(b, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise Exception("imdecode thumb failed")
            bgr = self._rotate_bgr(bgr)
            bgr = self._center_crop_bgr_to_aspect(bgr, self.STILL_TARGET_ASPECT)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb_bytes = rgb.tobytes()
        except Exception as e:
            self.log(f"REVIEW THUMB DECODE/ROTATE/CROP ERROR {e}")
            return

        def apply_thumb(_dt):
            with self._state_lock:
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
                self.log(f"REVIEW THUMB APPLY w={w} h={h}")
            except Exception as e:
                self.log(f"REVIEW THUMB APPLY ERROR {e}")

        Clock.schedule_once(apply_thumb, 0)

        if self.capture_type == CaptureType.JPG:
            self._download_fullres_and_replace(ccapi_path, request_id)
        else:
            self.log("RAW selected; full-res RAW fetch not implemented.")

    # ---------- contents listing ----------
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

    # ---------- menu / dropdown ----------
    def _style_menu_button(self, b):
        b.background_normal = ""
        b.background_down = ""
        b.background_color = (0.10, 0.10, 0.10, 0.80)
        b.color = (1, 1, 1, 1)
        return b

    def _build_dropdown(self):
        dd = DropDown(auto_dismiss=True)
        dd.auto_width = False
        dd.width = dp(380)
        dd.max_height = dp(650)

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
        add_button("Show metrics…", lambda: self._open_metrics_popup())
        add_toggle("Log overlay", False, lambda v: self._set_log_overlay_visible(v))

        add_header("EXIF / Author")
        add_button("Refresh Current EXIF now", lambda: self.refresh_exif())

        add_header("CSV")
        add_button("Load CSV…", lambda: self._open_csv_menu_popup())
        add_button("Subject List", lambda: self.open_subject_list())

        add_header("Capture")
        add_capture_type_buttons()

        add_header("Debug")
        add_button("Clear log", lambda: self._clear_log())

        return dd

    # ---------- popups ----------
    def _open_ip_popup(self):
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
                    self._set_exif_text("(IP changed; reconnect)")
            popup.dismiss()

        ok.bind(on_release=do_ok)
        cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def _open_fps_popup(self):
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
        popup.open()

    # ---------- CSV menu + SAF + Subject List ----------
    def _open_csv_menu_popup(self):
        root = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))
        b1 = Button(text="Load CSV file", size_hint=(1, None), height=dp(48))
        b2 = Button(text="Select headers", size_hint=(1, None), height=dp(48))
        b3 = Button(text="Close", size_hint=(1, None), height=dp(44))
        root.add_widget(b1)
        root.add_widget(b2)
        root.add_widget(b3)

        popup = Popup(title="Load CSV", content=root, size_hint=(0.9, 0.45))
        b1.bind(on_release=lambda *_: (popup.dismiss(), self._open_csv_filechooser()))
        b2.bind(on_release=lambda *_: (popup.dismiss(), self._open_headers_popup()))
        b3.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

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
            Clock.schedule_once(lambda *_: self._set_controls_idle(), 0)

        except Exception as e:
            self.log(f"CSV load failed (Android): {e}")

    def _read_android_uri_bytes(self, uri):
        from android import mActivity
        cr = mActivity.getContentResolver()
        stream = cr.openInputStream(uri)
        if stream is None:
            raise Exception("openInputStream() returned null")
        try:
            out = bytearray()
            buf = bytearray(64 * 1024)
            while True:
                n = stream.read(buf)
                if n == -1 or n == 0:
                    break
                out.extend(buf[:n])
            return bytes(out)
        finally:
            stream.close()

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
        root.add_widget(Label(text="Select columns to include in CSV Payload (joined with _):",
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

    def _build_csv_payload_from_row(self, row: dict) -> str:
        headers = self.selected_headers if self.selected_headers else (self.csv_headers[:3] if self.csv_headers else [])
        parts = []
        for h in headers:
            parts.append((row.get(h) or "").strip())
        parts = [p for p in parts if p]
        return "_".join(parts)

    def open_subject_list(self):
        if not self.csv_rows:
            self.log("No CSV loaded; Subject List unavailable")
            return

        if self._subject_popup is not None:
            try:
                self._subject_popup.dismiss()
            except Exception:
                pass
            self._subject_popup = None

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))

        row_search = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        row_search.add_widget(Label(text="Search", size_hint=(None, 1), width=dp(70), font_size=sp(12)))
        ti_search = TextInput(text="", multiline=False, font_size=sp(14))
        row_search.add_widget(ti_search)
        root.add_widget(row_search)

        row_sort = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        row_sort.add_widget(Label(text="Sort by", size_hint=(None, 1), width=dp(70), font_size=sp(12)))
        default_sort = self.selected_headers[0] if self.selected_headers else (self.csv_headers[0] if self.csv_headers else "")
        ti_sort = TextInput(text=default_sort, multiline=False, font_size=sp(14), hint_text="Header name")
        row_sort.add_widget(ti_sort)
        cb_desc = CheckBox(active=False, size_hint=(None, 1), width=dp(44))
        row_sort.add_widget(Label(text="Desc", size_hint=(None, 1), width=dp(50), font_size=sp(12)))
        row_sort.add_widget(cb_desc)
        root.add_widget(row_sort)

        root.add_widget(Label(text="Filters (contains):", size_hint=(1, None), height=dp(22), font_size=sp(12)))

        sv_filters = ScrollView(size_hint=(1, None), height=dp(160), do_scroll_x=False)
        filters_inner = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        filters_inner.bind(minimum_height=filters_inner.setter("height"))
        sv_filters.add_widget(filters_inner)

        filter_inputs = {}
        for h in self.csv_headers:
            r = BoxLayout(size_hint_y=None, height=dp(32), spacing=dp(6))
            lbl = Label(text=h, size_hint=(0.4, 1), font_size=sp(11), halign="left", valign="middle")
            lbl.bind(size=lbl.setter("text_size"))
            ti = TextInput(text="", multiline=False, font_size=sp(12), size_hint=(0.6, 1))
            r.add_widget(lbl)
            r.add_widget(ti)
            filters_inner.add_widget(r)
            filter_inputs[h] = ti

        root.add_widget(sv_filters)

        root.add_widget(Label(text="Results:", size_hint=(1, None), height=dp(22), font_size=sp(12)))
        sv_results = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        results_inner = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        results_inner.bind(minimum_height=results_inner.setter("height"))
        sv_results.add_widget(results_inner)
        root.add_widget(sv_results)

        btn_bar = BoxLayout(size_hint=(1, None), height=dp(44), spacing=dp(6))
        btn_refresh = Button(text="Refresh")
        btn_close = Button(text="Close")
        btn_bar.add_widget(btn_refresh)
        btn_bar.add_widget(btn_close)
        root.add_widget(btn_bar)

        popup = Popup(title="Subject List", content=root, size_hint=(0.98, 0.98))

        def compute_filtered_rows():
            search = (ti_search.text or "").strip().lower()
            sort_h = (ti_sort.text or "").strip()
            sort_desc = bool(cb_desc.active)

            filters = {}
            for h, ti in filter_inputs.items():
                v = (ti.text or "").strip().lower()
                if v:
                    filters[h] = v

            rows = self.csv_rows

            if filters:
                def row_ok(r):
                    for h, v in filters.items():
                        if v not in ((r.get(h) or "").lower()):
                            return False
                    return True
                rows = [r for r in rows if row_ok(r)]

            if search:
                headers = self.selected_headers if self.selected_headers else self.csv_headers
                def hit(r):
                    for h in headers:
                        if search in ((r.get(h) or "").lower()):
                            return True
                    return False
                rows = [r for r in rows if hit(r)]

            if sort_h:
                def key_fn(r):
                    return (r.get(sort_h) or "").lower()
                try:
                    rows = sorted(rows, key=key_fn, reverse=sort_desc)
                except Exception:
                    pass

            return rows

        def render_results():
            results_inner.clear_widgets()
            rows = compute_filtered_rows()
            max_show = 200
            show_rows = rows[:max_show]

            if not show_rows:
                results_inner.add_widget(Label(text="(no results)", size_hint_y=None, height=dp(24), font_size=sp(12)))
                return

            headers = self.selected_headers if self.selected_headers else (self.csv_headers[:3] if self.csv_headers else [])

            for r in show_rows:
                payload = self._build_csv_payload_from_row(r)
                display = payload if payload else " / ".join([(r.get(h) or "") for h in headers])
                btn = Button(text=display[:120], size_hint_y=None, height=dp(44), font_size=sp(12))

                def _make_pick(row=r, payload=payload):
                    def _pick(*_):
                        p = payload or self._build_csv_payload_from_row(row)
                        self._set_csv_payload(p)
                        self.log(f"CSV payload selected: {p}")
                        popup.dismiss()
                    return _pick

                btn.bind(on_release=_make_pick())
                results_inner.add_widget(btn)

            if len(rows) > max_show:
                results_inner.add_widget(Label(
                    text=f"(showing first {max_show} of {len(rows)} matches)",
                    size_hint_y=None, height=dp(24), font_size=sp(11)
                ))

        btn_refresh.bind(on_release=lambda *_: render_results())
        btn_close.bind(on_release=lambda *_: popup.dismiss())
        ti_search.bind(text=lambda *_: render_results())
        cb_desc.bind(active=lambda *_: render_results())
        ti_sort.bind(text=lambda *_: render_results())
        for ti in filter_inputs.values():
            ti.bind(text=lambda *_: render_results())

        popup.bind(on_dismiss=lambda *_: setattr(self, "_subject_popup", None))
        popup.open()
        self._subject_popup = popup
        render_results()

    # ---------- build ----------
    def build(self):
        outer = FloatLayout()
        main = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6), size_hint=(1, 1))
        outer.add_widget(main)

        header = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        self.exit_btn = Button(text="Exit", size_hint=(None, 1), width=dp(90), font_size=sp(14))
        header.add_widget(self.exit_btn)
        header.add_widget(Label(text="Volume Toolkit v2.1.8", font_size=sp(18)))
        self.menu_btn = Button(text="Menu", size_hint=(None, 1), width=dp(90), font_size=sp(16))
        header.add_widget(self.menu_btn)
        main.add_widget(header)

        top_font = sp(12)
        row2 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.connect_btn = Button(text="Connect", font_size=top_font, size_hint=(1, 1))
        self.start_btn = Button(text="Live View Off", disabled=True, font_size=top_font, size_hint=(1, 1))
        self.autofetch_btn = Button(text="Autofetch Off", disabled=True, font_size=top_font, size_hint=(1, 1))
        self.qr_btn = Button(text="QR Detect Off", disabled=True, font_size=top_font, size_hint=(1, 1))
        row2.add_widget(self.connect_btn)
        row2.add_widget(self.start_btn)
        row2.add_widget(self.autofetch_btn)
        row2.add_widget(self.qr_btn)
        main.add_widget(row2)

        def info_row(title, initial, title_w=dp(110), h=dp(28), f=sp(13)):
            row = BoxLayout(spacing=dp(6), size_hint=(1, None), height=h)
            t = Label(text=title, size_hint=(None, 1), width=title_w, font_size=f, halign="left", valign="middle")
            t.bind(size=lambda *_: setattr(t, "text_size", (t.width, None)))
            v = Label(text=initial, size_hint=(1, 1), font_size=f, halign="left", valign="middle")
            v.bind(size=lambda *_: setattr(v, "text_size", (v.width, None)))
            row.add_widget(t)
            row.add_widget(v)
            return row, v

        row_exif, self.exif_label = info_row("Current EXIF", "(not connected)", h=dp(28), f=sp(13))
        row_qr, self.payload_label = info_row("QR Payload", "(none)", h=dp(28), f=sp(13))
        row_csv, self.csv_payload_label = info_row("CSV Payload", "(none)", h=dp(24), f=sp(12))
        main.add_widget(row_exif)
        main.add_widget(row_qr)
        main.add_widget(row_csv)

        self.metrics = Label(text="Delay: -- ms | Fetch: 0 | Decode: 0 | Display: 0")

        # Middle band (explicit height)
        middle = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, None))
        main.add_widget(middle)

        preview_holder = AnchorLayout(anchor_x="center", anchor_y="center", size_hint=(0.80, 1))
        self.preview = PreviewOverlay(size_hint=(None, None))
        preview_holder.add_widget(self.preview)
        middle.add_widget(preview_holder)

        sidebar = BoxLayout(orientation="vertical", size_hint=(0.20, 1), spacing=dp(4))
        sidebar.add_widget(Label(text="Most Recent", size_hint=(1, None), height=dp(20), font_size=sp(12)))

        self.thumb_rv = ThumbRV(size_hint=(1, 1))
        rbl = RecycleBoxLayout(orientation="vertical", default_size=(None, dp(90)),
                               default_size_hint=(1, None), size_hint=(1, None))
        rbl.bind(minimum_height=rbl.setter("height"))
        self.thumb_rv.layout_manager = rbl
        self.thumb_rv.viewclass = ThumbRow
        self.thumb_rv.add_widget(rbl)
        sidebar.add_widget(self.thumb_rv)
        middle.add_widget(sidebar)

        def fit_preview_to_holder(*_):
            w = preview_holder.width * 0.98
            h = preview_holder.height * 0.98
            self.preview.size = (w, h)

        preview_holder.bind(pos=fit_preview_to_holder, size=fit_preview_to_holder)

        # keep thumb list ~4 visible + enforce row height
        def size_thumb_rows(*_):
            avail = max(dp(240), sidebar.height - dp(20) - dp(4))
            row_h = max(dp(56), avail / 4.0)
            self._thumb_row_height = row_h
            self.thumb_rv.layout_manager.default_size = (None, row_h)
            self.thumb_rv.refresh_from_layout()
            self._rv_rebuild_data_main()

        sidebar.bind(size=size_thumb_rows)

        bottom = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, None), height=dp(52))
        self.push_btn = Button(text="Push Payload", font_size=sp(16), disabled=True)
        self.subject_btn = Button(text="Subject List", font_size=sp(16), disabled=True)
        bottom.add_widget(self.push_btn)
        bottom.add_widget(self.subject_btn)
        main.add_widget(bottom)

        def recalc_middle_height(*_):
            spacing = main.spacing
            gaps = 6  # header,row2,row_exif,row_qr,row_csv,middle,bottom
            used = header.height + row2.height + row_exif.height + row_qr.height + row_csv.height + bottom.height
            used += spacing * gaps
            middle.height = max(dp(200), main.height - used)

        main.bind(size=recalc_middle_height)
        Clock.schedule_once(recalc_middle_height, 0)

        # Bottom log overlay
        self.log_overlay = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            height=0,
            pos_hint={"x": 0, "y": 0},
            spacing=dp(4),
            padding=[dp(6), dp(6), dp(6), dp(6)]
        )
        with self.log_overlay.canvas.before:
            Color(0.0, 0.0, 0.0, 0.75)
            self._log_overlay_bg = Rectangle(pos=self.log_overlay.pos, size=self.log_overlay.size)
        self.log_overlay.bind(
            pos=lambda *_: setattr(self._log_overlay_bg, "pos", self.log_overlay.pos),
            size=lambda *_: setattr(self._log_overlay_bg, "size", self.log_overlay.size),
        )

        log_top = BoxLayout(size_hint=(1, None), height=dp(32), spacing=dp(6))
        log_top.add_widget(Label(text="Log", size_hint=(1, 1), font_size=sp(12), color=(1, 1, 1, 1)))
        btn_clear = Button(text="Clear", size_hint=(None, 1), width=dp(70), font_size=sp(12))
        btn_close = Button(text="X", size_hint=(None, 1), width=dp(44), font_size=sp(12))
        log_top.add_widget(btn_clear)
        log_top.add_widget(btn_close)
        self.log_overlay.add_widget(log_top)

        self._log_overlay_sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self._log_overlay_label = Label(text="", size_hint_y=None, halign="left", valign="top",
                                        font_size=sp(10), color=(1, 1, 1, 1))
        self._log_overlay_label.bind(width=lambda *_: setattr(self._log_overlay_label, "text_size", (self._log_overlay_label.width, None)))
        self._log_overlay_label.bind(texture_size=lambda *_: setattr(self._log_overlay_label, "height", self._log_overlay_label.texture_size[1]))
        self._log_overlay_sv.add_widget(self._log_overlay_label)
        self.log_overlay.add_widget(self._log_overlay_sv)

        outer.add_widget(self.log_overlay)
        self._set_log_overlay_visible(False)

        btn_close.bind(on_release=lambda *_: self._set_log_overlay_visible(False))
        btn_clear.bind(on_release=lambda *_: self._clear_log())

        # Menu + events
        self.dropdown = self._build_dropdown()
        self.menu_btn.bind(on_release=lambda *_: self.dropdown.open(self.menu_btn))

        self.exit_btn.bind(on_press=lambda *_: self.exit_app())
        self.connect_btn.bind(on_press=lambda *_: self.connect_camera())
        self.start_btn.bind(on_press=lambda *_: self.toggle_liveview())
        self.autofetch_btn.bind(on_press=lambda *_: self.toggle_autofetch())
        self.qr_btn.bind(on_press=lambda *_: self.toggle_qr_detect())
        self.push_btn.bind(on_press=lambda *_: self.push_payload())
        self.subject_btn.bind(on_press=lambda *_: self.open_subject_list())

        self._reschedule_display_loop(12)

        self._apply_connect_btn_style()
        self._apply_live_btn_style()
        self._apply_autofetch_btn_style()
        self._apply_qr_btn_style()
        self._set_controls_idle()

        # init rv data
        self._rv_rebuild_data()

        self.log("Android CCAPI GUI ready")
        return outer


if __name__ == "__main__":
    VolumeToolkitApp().run()
