# Android-focused Volume Toolkit (threaded decoder + background poller)
# v12-based corrected main.py:
# - Restored _json_call
# - Fixed HTML-escaped characters / syntax issues
# - Grid drawn in orange
# - Thumbnails rotated to match preview
# - Thumbnails only by default; tapping a thumb overlays a low-res stretched image,
#   background-downloads full-res JPG and replaces it when ready
# - QR pulse scanning: 1s max, highlight box shown up to 3s (green)
# - Decoder thread started after build() to avoid race conditions
# - Connect runs in background thread
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
from kivy.core.window import Window
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

# Suppress self-signed HTTPS warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Android: keep Kivy writable files out of the extracted app directory.
if os.environ.get("ANDROID_ARGUMENT"):
    private_dir = os.environ.get("ANDROID_PRIVATE")
    if private_dir:
        kivy_home = os.path.join(private_dir, ".kivy")
        os.makedirs(kivy_home, exist_ok=True)
        os.environ["KIVY_HOME"] = kivy_home


def pil_rotate_90s(img, ang):
    ang = int(ang) % 360
    if ang == 0:
        return img
    try:
        T = PILImage.Transpose
        if ang == 90:
            return img.transpose(T.ROTATE_90)
        if ang == 180:
            return img.transpose(T.ROTATE_180)
        if ang == 270:
            return img.transpose(T.ROTATE_270)
        return img
    except Exception:
        # Pillow older variants
        if ang == 90:
            return img.transpose(PILImage.ROTATE_90)
        if ang == 180:
            return img.transpose(PILImage.ROTATE_180)
        if ang == 270:
            return img.transpose(PILImage.ROTATE_270)
        return img


class PreviewOverlay(FloatLayout):
    show_border = BooleanProperty(True)
    show_grid = BooleanProperty(True)
    show_57 = BooleanProperty(True)
    show_810 = BooleanProperty(True)
    show_oval = BooleanProperty(True)
    show_qr = BooleanProperty(True)

    grid_n = NumericProperty(3)

    oval_cx = NumericProperty(0.5)
    oval_cy = NumericProperty(0.6)
    oval_w = NumericProperty(0.333)
    oval_h = NumericProperty(0.333)

    # default preview rotation (user requested preview rotated 180° from prior default)
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
        lw_qr = 6

        with self.img.canvas.after:
            # border
            self._c_border = Color(0.2, 0.6, 1.0, 1.0)
            self._ln_border = Line(width=lw)

            # placeholders (grid is drawn explicitly orange later)
            self._c_grid = Color(1.0, 0.6, 0.0, 0.85)

            self._c_57 = Color(1.0, 0.2, 0.2, 0.95)
            self._ln_57 = Line(width=lw)

            self._c_810 = Color(1.0, 0.9, 0.2, 0.95)
            self._ln_810 = Line(width=lw)

            self._c_oval = Color(0.7, 0.2, 1.0, 0.95)
            self._ln_oval = Line(width=lw)

            # QR highlight (green)
            self._c_qr = Color(0.0, 1.0, 0.0, 0.95)
            self._ln_qr = Line(width=lw_qr, close=True)

        # store (ColorObj, LineObj) pairs for grid lines to remove them cleanly
        self._ln_grid_list = []

        self.bind(pos=self._redraw, size=self._redraw)
        self.bind(
            show_border=self._redraw, show_grid=self._redraw, show_57=self._redraw,
            show_810=self._redraw, show_oval=self._redraw, show_qr=self._redraw,
            grid_n=self._redraw,
            oval_cx=self._redraw, oval_cy=self._redraw, oval_w=self._redraw, oval_h=self._redraw
        )
        self.img.bind(pos=self._redraw, size=self._redraw, texture=self._redraw, texture_size=self._redraw)

        self._qr_points_px = None
        self._redraw()

    def set_texture(self, texture):
        self.img.texture = texture
        self._redraw()

    def set_qr(self, points_px):
        self._qr_points_px = points_px
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

        # remove old grid color/line objects
        for col_obj, line_obj in list(self._ln_grid_list):
            try:
                self.img.canvas.after.remove(col_obj)
            except Exception:
                pass
            try:
                self.img.canvas.after.remove(line_obj)
            except Exception:
                pass
        self._ln_grid_list = []

        # Draw grid lines explicitly in orange so they don't pick up other Color state
        n = int(self.grid_n)
        if self.show_grid and n >= 2:
            for i in range(1, n):
                x = fx + fw * (i / n)
                col = Color(1.0, 0.6, 0.0, 0.85)  # orange
                ln = Line(points=[x, fy, x, fy + fh], width=2)
                self.img.canvas.after.add(col)
                self.img.canvas.after.add(ln)
                self._ln_grid_list.append((col, ln))
            for i in range(1, n):
                y = fy + fh * (i / n)
                col = Color(1.0, 0.6, 0.0, 0.85)
                ln = Line(points=[fx, y, fx + fw, y], width=2)
                self.img.canvas.after.add(col)
                self.img.canvas.after.add(ln)
                self._ln_grid_list.append((col, ln))

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

        if self.show_qr and self._qr_points_px and self.img.texture and self.img.texture.size[0] > 0:
            iw, ih = self.img.texture.size
            dx, dy, dw, dh = fx, fy, fw, fh

            line_pts = []
            for (x, y) in self._qr_points_px:
                u = float(x) / float(iw)
                v = float(y) / float(ih)
                sx = dx + u * dw
                sy = dy + v * dh
                line_pts += [sx, sy]

            self._ln_qr.points = line_pts
        else:
            self._ln_qr.points = []


class CaptureType:
    JPG = "JPG"
    RAW = "RAW"
    BOTH = "Both"


class VolumeToolkitApp(App):
    capture_type = StringProperty(CaptureType.JPG)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default IP moved to settings popup
        self.camera_ip = "192.168.34.29"

        self.connected = False
        self.live_running = False
        self.session_started = False

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_jpeg_ts = 0.0
        self._last_decoded_ts = 0.0

        self._fetch_thread = None
        self._display_event = None

        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        # internal log lines (hidden by default)
        self._log_lines = []
        self._max_log_lines = 300
        self.show_log = False

        self._frame_texture = None
        self._frame_size = None

        self.dropdown = None

        # QR: default OFF; temporary pulse on preview tap
        self.qr_enabled = False
        self._qr_temp_active = False
        self._qr_pulse_event = None
        self.qr_interval_s = 0.15
        self.qr_new_gate_s = 0.70
        self._qr_detector = cv2.QRCodeDetector()
        self._qr_thread = None
        self._latest_qr_text = ""
        self._latest_qr_points = None
        self._qr_seen = set()
        self._qr_last_add_time = 0.0

        # For sharing latest decoded BGR with QR thread
        self._latest_decoded_bgr = None
        self._latest_decoded_bgr_ts = 0.0

        # Decoder queue; thread will be created and started in build() to avoid race
        self._decode_queue = queue.Queue(maxsize=2)
        self._decoder_thread = None
        self._decoder_stop = threading.Event()

        # Overlay mode (set when a thumbnail is selected)
        self._overlay_active = False
        self._overlay_thumb_index = None

        # For QR highlight timeout (3s)
        self._qr_highlight_event = None

        # Author
        self.author_max_chars = 60
        self._last_committed_author = None
        self._author_update_in_flight = False
        self.manual_payload = ""

        # CSV / headers
        self.csv_headers = []
        self.csv_rows = []
        self.selected_headers = []
        self._headers_popup = None

        # Thumbnails
        self._thumb_textures = []
        self._thumb_images = []
        self._thumb_paths = []

        # Storage
        self.download_dir = "downloads"
        self.thumb_dir = "thumbs"

        self._last_seen_image = None
        self._poll_thread = None
        self._poll_thread_stop = threading.Event()
        self.poll_interval_s = 2.0

        self.save_full_size = False

        # HTTPS session
        self._session = requests.Session()
        self._session.verify = False

        # Android SAF picker
        self._android_activity_bound = False
        self._csv_req_code = 4242

        # Layout references (populated in build)
        self.header = None
        self.preview_holder = None

    # ---------- network helper (restored) ----------
    def _json_call(self, method, path, payload=None, timeout=8.0):
        """
        Centralized HTTP + JSON helper. Returns (status_string, parsed_json_or_None).
        Call from a background thread on Android (this function blocks).
        """
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

    def build(self):
        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        # Header and menu
        self.header = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        self.header_title = Label(text="Volume Toolkit v1.0.6", font_size=sp(18))
        self.header.add_widget(self.header_title)
        self.menu_btn = Button(text="Menu", size_hint=(None, 1), width=dp(90), font_size=sp(16))
        self.header.add_widget(self.menu_btn)
        root.add_widget(self.header)

        # Control row: Connect + Start only (IP in menu)
        row2 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.connect_btn = Button(text="Connect", font_size=sp(16), size_hint=(None, 1), width=dp(120))
        self._style_connect_button(initial=True)
        self.start_btn = Button(text="Start", disabled=True, font_size=sp(16), size_hint=(None, 1), width=dp(140))
        self._style_start_button(stopped=True)
        row2.add_widget(self.connect_btn)
        row2.add_widget(self.start_btn)
        root.add_widget(row2)

        # Last QR string shown where metrics label was
        self.qr_last_label = Label(text="QR: none", size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.qr_last_label)

        # Compact status line
        self.status = Label(text="Status: not connected", size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.status)

        # Small QR status
        self.qr_status = Label(text="", size_hint=(1, None), height=dp(18), font_size=sp(11))
        root.add_widget(self.qr_status)

        # Main area: preview 80%, thumbs 20%
        main_area = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, 0.6))

        self.preview_holder = AnchorLayout(anchor_x="center", anchor_y="center", size_hint=(0.80, 1))
        self.preview_scatter = Scatter(do_translation=False, do_scale=False, do_rotation=False, size_hint=(None, None))
        self.preview = PreviewOverlay(size_hint=(None, None))
        # make sure the handler exists (it's defined later in class)
        self.preview.bind(on_touch_down=self._on_preview_touch)
        self.preview_scatter.add_widget(self.preview)
        self.preview_holder.add_widget(self.preview_scatter)
        main_area.add_widget(self.preview_holder)

        sidebar = BoxLayout(orientation="vertical", size_hint=(0.20, 1), spacing=dp(4))
        sidebar.add_widget(Label(text="Last 5", size_hint=(1, None), height=dp(20), font_size=sp(12)))
        for idx in range(5):
            img = Image(size_hint=(1, None), height=dp(100), allow_stretch=True, keep_ratio=True)
            img.thumb_index = idx
            img.bind(on_touch_down=self._on_thumb_touch)
            sidebar.add_widget(img)
            self._thumb_images.append(img)
        main_area.add_widget(sidebar)

        root.add_widget(main_area)

        # Fit preview to the holder with fixed sizes
        def fit_preview_to_holder(*_):
            w = max(dp(220), self.preview_holder.width * 0.98)
            h = max(dp(220), self.preview_holder.height * 0.98)
            self.preview_scatter.size = (w, h)
            self.preview.size = (w, h)
            self.preview_scatter.pos = (
                self.preview_holder.x + (self.preview_holder.width - w) / 2.0,
                self.preview_holder.y + (self.preview_holder.height - h) / 2.0
            )

        self._fit_preview_to_holder = fit_preview_to_holder
        self.preview_holder.bind(pos=fit_preview_to_holder, size=fit_preview_to_holder)

        # Log area (hidden by default)
        self.log_holder = BoxLayout(orientation="vertical", size_hint=(1, None), height=0)
        log_sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.log_label = Label(text="", size_hint_y=None, halign="left", valign="top", font_size=sp(11))
        self.log_label.bind(width=lambda *_: setattr(self.log_label, "text_size", (self.log_label.width, None)))
        self.log_label.bind(texture_size=lambda *_: setattr(self.log_label, "height", self.log_label.texture_size[1]))
        log_sv.add_widget(self.log_label)
        self.log_holder.add_widget(log_sv)
        root.add_widget(self.log_holder)

        # Menu
        self.dropdown = self._build_dropdown()
        self.menu_btn.bind(on_release=lambda *_: self.dropdown.open(self.menu_btn))

        # Button bindings
        self.connect_btn.bind(on_press=lambda *_: self.connect_camera())
        self.start_btn.bind(on_press=lambda *_: self._on_start_pressed())

        # Start the decoder thread now that preview exists (avoid race)
        if self._decoder_thread is None:
            self._decoder_thread = threading.Thread(target=self._decoder_loop, daemon=True)
            self._decoder_thread.start()

        # Display ticker
        self._reschedule_display_loop(12)

        # React to rotation/resize
        Window.bind(on_resize=self._on_window_resize)

        self._set_controls_idle()
        self._log_internal("UI ready")
        return root

    # ---------- window/rotation handling ----------
    def _on_window_resize(self, instance, width, height):
        try:
            if width > height:
                self.header.height = dp(40)
                self.header_title.font_size = sp(18)
                self.menu_btn.width = dp(90)
            else:
                self.header.height = dp(44)
                self.header_title.font_size = sp(16)
                self.menu_btn.width = dp(80)
        except Exception:
            pass
        try:
            self._fit_preview_to_holder()
        except Exception:
            pass

    # ---------- styling helpers ----------
    def _style_connect_button(self, initial=False):
        self.connect_btn.background_normal = ""
        self.connect_btn.background_down = ""
        self.connect_btn.background_color = (0.06, 0.45, 0.75, 1.0)  # blue
        if initial or not self.connected:
            self.connect_btn.color = (1, 1, 1, 1)
        else:
            self.connect_btn.color = (1, 1, 0, 1)

    def _style_start_button(self, stopped=True):
        self.start_btn.background_normal = ""
        self.start_btn.background_down = ""
        if stopped:
            self.start_btn.background_color = (0.0, 0.6, 0.0, 1.0)
            self.start_btn.color = (1, 1, 1, 1)
            self.start_btn.text = "Start"
        else:
            self.start_btn.background_color = (0.8, 0.0, 0.0, 1.0)
            self.start_btn.color = (1, 1, 1, 1)
            self.start_btn.text = "Stop"

    # ---------- logging (internal) ----------
    def _log_internal(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]
        if getattr(self, "show_log", False):
            self._refresh_log_view()

    def _refresh_log_view(self):
        metrics_line = self._get_metrics_text()
        self.log_label.text = metrics_line + "\n\n" + "\n".join(self._log_lines)

    def _get_metrics_text(self):
        return f"Delay: -- ms | Fetch: {self._fetch_count} | Decode: {self._decode_count} | Display: {self._display_count}"

    def _set_log_visible(self, visible: bool):
        self.show_log = bool(visible)
        if self.show_log:
            self.log_holder.height = dp(150)
            self.log_holder.opacity = 1
            self.log_holder.disabled = False
            self._refresh_log_view()
        else:
            self.log_holder.height = 0
            self.log_holder.opacity = 0
            self.log_holder.disabled = True

    # ---------- menu / UI ----------
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
        dd.max_height = dp(600)

        with dd.canvas.before:
            Color(0.0, 0.0, 0.0, 0.80)
            panel = Rectangle(pos=dd.pos, size=dd.size)
        dd.bind(pos=lambda *_: setattr(panel, "pos", dd.pos), size=lambda *_: setattr(panel, "size", dd.size))

        def add_header(text):
            dd.add_widget(Label(text=text, size_hint_y=None, height=dp(26), font_size=sp(15), color=(1, 1, 1, 1)))

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

        add_header("Framing")
        add_button("Reset framing", lambda: self._fit_preview_to_holder())

        add_header("Overlays")
        add_toggle("Border (blue)", True, lambda v: setattr(self.preview, "show_border", v))
        add_toggle("Grid (orange)", True, lambda v: setattr(self.preview, "show_grid", v))
        add_toggle("Crop 5:7 (red)", True, lambda v: setattr(self.preview, "show_57", v))
        add_toggle("Crop 8:10 (yellow)", True, lambda v: setattr(self.preview, "show_810", v))
        add_toggle("Oval (purple)", True, lambda v: setattr(self.preview, "show_oval", v))
        add_toggle("QR overlay", True, lambda v: setattr(self.preview, "show_qr", v))

        add_header("QR & Author")
        add_toggle("QR detect (OpenCV) (permanent)", False, lambda v: self._set_qr_enabled(bool(v)))
        add_button("Load CSV…", lambda: self._open_csv_filechooser())
        add_button("Select headers…", lambda: self._open_headers_popup())
        add_button("Push payload (Author)", lambda: self._maybe_commit_author(self.manual_payload, source="manual"))

        add_header("Capture")
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4), padding=[dp(4), 0, dp(4), 0])
        row.add_widget(Label(text="Capture:", size_hint=(None, 1), width=dp(70), font_size=sp(13), color=(1, 1, 1, 1)))
        def mk_btn(label, ctype):
            b = Button(text=label, size_hint=(1, 1), font_size=sp(12))
            self._style_menu_button(b)
            b.bind(on_release=lambda *_: (setattr(self, "capture_type", ctype), self._log_internal(f"Capture type set to {ctype}")))
            return b
        row.add_widget(mk_btn("JPG", CaptureType.JPG))
        row.add_widget(mk_btn("RAW", CaptureType.RAW))
        row.add_widget(mk_btn("Both", CaptureType.BOTH))
        dd.add_widget(row)

        add_button("Fetch latest image", lambda: threading.Thread(target=self._background_download_latest, daemon=True).start())
        add_button("Start auto-fetch", lambda: self.start_polling_new_images())
        add_button("Stop auto-fetch", lambda: self.stop_polling_new_images())

        add_header("Settings")
        add_button("IP settings…", lambda: self._open_ip_popup())
        add_button("Display FPS…", lambda: self._open_fps_popup())
        add_toggle("Show log", False, lambda v: self._set_log_visible(v))

        add_header("Debug")
        add_button("Dump /ccapi", lambda: self.dump_ccapi())

        return dd

    # ---------- FPS / IP popups ----------
    def _open_fps_popup(self):
        content = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(6))
        sv = Slider(min=5, max=30, value=12, step=1)
        lbl = Label(text=str(int(sv.value)), size_hint=(1, None), height=dp(28))
        sv.bind(value=lambda _, v: setattr(lbl, "text", str(int(v))))
        content.add_widget(Label(text="Display FPS", size_hint=(1, None), height=dp(28)))
        content.add_widget(sv)
        btns = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        ok = Button(text="OK")
        cancel = Button(text="Cancel")
        btns.add_widget(ok)
        btns.add_widget(cancel)
        content.add_widget(btns)
        popup = Popup(title="Display FPS", content=content, size_hint=(0.8, 0.35))

        def do_ok(*_):
            self._reschedule_display_loop(int(sv.value))
            popup.dismiss()

        ok.bind(on_release=do_ok)
        cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def _open_ip_popup(self):
        content = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(6))
        ti = TextInput(text=self.camera_ip, multiline=False, font_size=sp(16))
        content.add_widget(Label(text="Camera IP (no port):", size_hint=(1, None), height=dp(28)))
        content.add_widget(ti)
        btns = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        btn_ok = Button(text="Save")
        btn_cancel = Button(text="Cancel")
        btns.add_widget(btn_ok)
        btns.add_widget(btn_cancel)
        content.add_widget(btns)
        popup = Popup(title="IP settings", content=content, size_hint=(0.85, 0.35))

        def do_save(*_):
            val = ti.text.strip()
            if val:
                self.camera_ip = val
                self._log_internal(f"Camera IP set to {val}")
            popup.dismiss()

        btn_ok.bind(on_release=do_save)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    # ---------- connect / author ----------
    def connect_camera(self):
        if self.live_running:
            self._log_internal("Connect disabled while live view is running. Stop first.")
            return

        if not self.camera_ip:
            self.status.text = "Status: enter an IP (use Settings->IP)"
            return

        self.connect_btn.disabled = True
        self.status.text = f"Status: connecting to {self.camera_ip}:443..."
        self._log_internal(f"Connecting to {self.camera_ip}:443")

        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        try:
            status, data = self._json_call("GET", '/ccapi/ver100/deviceinformation', None, timeout=8.0)
        except Exception as e:
            status, data = f"ERR {e}", None

        def _finish(dt):
            try:
                if status and str(status).startswith("200") and data:
                    self.connected = True
                    self.status.text = f"Status: connected ({data.get('productname', 'camera')})"
                    self._log_internal("Connected OK")
                    try:
                        self.connect_btn.color = (1, 1, 0, 1)
                    except Exception:
                        pass
                    self.start_btn.disabled = False
                else:
                    self.connected = False
                    self.status.text = f"Status: connect failed ({status})"
                    self._log_internal(f"Connect failed: {status}")
                    try:
                        self.connect_btn.color = (1, 1, 1, 1)
                    except Exception:
                        pass
                    self.start_btn.disabled = True
            finally:
                try:
                    self.connect_btn.disabled = False
                except Exception:
                    pass

        Clock.schedule_once(_finish, 0)

    def _author_value(self, payload):
        s = (payload or "").strip()
        if not s:
            return ""
        return s[: int(self.author_max_chars)]

    def _maybe_commit_author(self, payload: str, source="qr"):
        value = self._author_value(payload)
        if not value:
            return
        if not self.connected:
            self._log_internal(f"Author update skipped ({source}): not connected")
            return
        if self._last_committed_author == value:
            return
        if self._author_update_in_flight:
            return

        self._author_update_in_flight = True
        Clock.schedule_once(lambda *_: setattr(self.qr_status, "text", f"Author updating… ({source})"), 0)
        threading.Thread(target=self._commit_author_worker, args=(value, source), daemon=True).start()

    def _commit_author_worker(self, value: str, source: str):
        ok = False
        got = None
        err = None
        try:
            st_put, _ = self._json_call(
                "PUT",
                '/ccapi/ver100/functions/registeredname/author',
                {"author": value},
                timeout=8.0
            )
            if not st_put.startswith("200"):
                raise Exception(f"PUT failed: {st_put}")

            st_get, data = self._json_call(
                "GET",
                '/ccapi/ver100/functions/registeredname/author',
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
                self._log_internal(f"Author updated+verified ({source}): '{value}'")
                self.qr_status.text = "Author updated ✓"
            else:
                self._log_internal(f"Author verify failed ({source}). wrote='{value}' read='{got}' err='{err}'")
                self.qr_status.text = "Author verify failed ✗"

        Clock.schedule_once(_finish, 0)

    # ---------- liveview + QR + decoder ----------
    def _set_qr_enabled(self, enabled: bool):
        self.qr_enabled = bool(enabled)
        if not self.qr_enabled and not self._qr_temp_active:
            self._set_qr_ui(None, None, note="QR: off")
        elif self.qr_enabled:
            self._set_qr_ui(None, None, note="QR: on")

    def _reschedule_display_loop(self, fps):
        if self._display_event is not None:
            try:
                self._display_event.cancel()
            except Exception:
                pass
        fps = max(1, int(fps))
        self._display_event = Clock.schedule_interval(self._ui_noop_display_tick, 1.0 / fps)

    def _ui_noop_display_tick(self, dt):
        self._display_count += 1
        self._update_metrics(self._last_decoded_ts)

    def _set_controls_idle(self):
        self.connect_btn.disabled = False
        self._style_connect_button()
        self._style_start_button(stopped=True)

    def _set_controls_running(self):
        self.connect_btn.disabled = True
        self._style_connect_button()
        self._style_start_button(stopped=False)

    def _on_start_pressed(self):
        if not self.live_running:
            if not self.connected:
                self._log_internal("Cannot start live: not connected")
                return
            self.start_liveview()
            self._set_controls_running()
        else:
            self.stop_liveview()
            self._set_controls_idle()

    def start_liveview(self):
        if not self.connected or self.live_running:
            return

        payload = {"liveviewsize": "small", "cameradisplay": "on"}
        self._log_internal("Starting live view size=small, cameradisplay=on")

        status, _ = self._json_call("POST", '/ccapi/ver100/shooting/liveview', payload, timeout=10.0)
        if not status.startswith("200"):
            self.status.text = f"Status: live view start failed ({status})"
            self._log_internal(f"Live view start failed: {status}")
            return

        self.session_started = True
        self.live_running = True
        self._set_controls_running()
        self.status.text = "Status: live"

        with self._lock:
            self._latest_jpeg = None
            self._latest_jpeg_ts = 0.0
        self._last_decoded_ts = 0.0
        self._frame_texture = None
        self._frame_size = None

        self._latest_qr_text = ""
        self._latest_qr_points = None
        self._qr_seen = set()
        self._qr_last_add_time = 0.0
        self._set_qr_ui(None, None, note="QR: off" if not self.qr_enabled else "QR: on")

        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        # start fetch + qr threads
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
                self._json_call("DELETE", '/ccapi/ver100/shooting/liveview', None, timeout=6.0)
            except Exception:
                pass
            self.session_started = False

        self.status.text = "Status: connected (live stopped)" if self.connected else "Status: not connected"
        self._log_internal("Live view stopped")
        self._set_controls_idle()

    def _liveview_fetch_loop(self):
        url = f"https://{self.camera_ip}/ccapi/ver100/shooting/liveview/flip"
        while self.live_running:
            try:
                resp = self._session.get(url, timeout=5.0)
                if resp.status_code == 200 and resp.content:
                    jpeg = resp.content
                    ts = time.time()
                    with self._lock:
                        self._latest_jpeg = jpeg
                        self._latest_jpeg_ts = ts
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
                self._log_internal(f"liveview fetch error: {e}")
                time.sleep(0.10)

    def _decoder_loop(self):
        while not self._decoder_stop.is_set():
            try:
                jpeg, ts = self._decode_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # defensive: ensure preview exists before using preview.preview_rotation
            if not hasattr(self, "preview") or self.preview is None:
                time.sleep(0.05)
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
                    try:
                        # if user selected overlay, don't replace preview
                        if getattr(self, "_overlay_active", False):
                            return
                        if self._frame_texture is None or self._frame_size != (w, h):
                            tex = Texture.create(size=(w, h), colorfmt="rgb")
                            tex.flip_vertical()
                            self._frame_texture = tex
                            self._frame_size = (w, h)
                            self._log_internal(f"texture init size={w}x{h}")
                        self._frame_texture.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
                        self.preview.set_texture(self._frame_texture)
                        self._last_decoded_ts = ts
                    except Exception as e:
                        self._log_internal(f"texture update err: {e}")

                Clock.schedule_once(_update_texture_on_main, 0)
                self._decode_count += 1

            except Exception:
                continue

    def _qr_loop(self):
        last_processed_ts = 0.0
        while self.live_running:
            if not (self.qr_enabled or self._qr_temp_active):
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
                decoded, points, _ = self._qr_detector.detectAndDecode(bgr)
                qr_text = decoded.strip() if isinstance(decoded, str) else ""
                qr_points = None
                if points is not None:
                    try:
                        pts = points.astype(int).reshape(-1, 2)
                        if len(pts) >= 4:
                            qr_points = [(int(pts[i][0]), int(pts[i][1])) for i in range(4)]
                    except Exception:
                        qr_points = None

                if qr_text or qr_points:
                    self._publish_qr(qr_text if qr_text else None, qr_points)
                    if self._qr_temp_active:
                        Clock.schedule_once(lambda *_: self._end_pulse_qr(), 0)
            except Exception:
                pass

            last_processed_ts = ts
            time.sleep(max(0.05, float(self.qr_interval_s)))

    def _publish_qr(self, text, points):
        now = time.time()

        if text:
            if (text not in self._qr_seen) and (now - self._qr_last_add_time >= self.qr_new_gate_s):
                self._qr_seen.add(text)
                self._qr_last_add_time = now
                self._log_internal(f"QR: {text}")
            self._maybe_commit_author(text, source="qr")

        if text:
            self._latest_qr_text = text
            Clock.schedule_once(lambda *_: setattr(self.qr_last_label, "text", f"QR: {text}"), 0)
        elif self._latest_qr_text:
            Clock.schedule_once(lambda *_: setattr(self.qr_last_label, "text", f"QR: {self._latest_qr_text}"), 0)

        if points:
            if getattr(self, "_qr_highlight_event", None):
                try:
                    self._qr_highlight_event.cancel()
                except Exception:
                    pass
                self._qr_highlight_event = None

            Clock.schedule_once(lambda *_: self.preview.set_qr(points), 0)
            try:
                self._qr_highlight_event = Clock.schedule_once(lambda *_: self._clear_qr_overlay(), 3.0)
            except Exception:
                def bg_clear():
                    time.sleep(3.0)
                    Clock.schedule_once(lambda *_: self._clear_qr_overlay(), 0)
                threading.Thread(target=bg_clear, daemon=True).start()

        note = f"QR: {text[:80]}" if text else ("QR: detected" if points else ("QR: on" if self.qr_enabled else "QR: off"))
        Clock.schedule_once(lambda *_: setattr(self.qr_status, "text", note), 0)

    def _clear_qr_overlay(self):
        try:
            self.preview.set_qr(None)
        except Exception:
            pass
        self._latest_qr_points = None
        if getattr(self, "_qr_highlight_event", None):
            try:
                self._qr_highlight_event.cancel()
            except Exception:
                pass
            self._qr_highlight_event = None

    def _set_qr_ui(self, text, points, note="QR: none"):
        if text:
            self._latest_qr_text = text
            self.qr_last_label.text = f"QR: {text}"
        if points:
            self._latest_qr_points = points
            self.preview.set_qr(points)
            if getattr(self, "_qr_highlight_event", None):
                try:
                    self._qr_highlight_event.cancel()
                except Exception:
                    pass
            self._qr_highlight_event = Clock.schedule_once(lambda *_: self._clear_qr_overlay(), 3.0)
        self.qr_status.text = note

    # ---------- preview tap handling for QR pulse ----------
    def _on_preview_touch(self, instance, touch):
        # only react to touches inside the preview widget
        if not instance.collide_point(*touch.pos):
            return False
        # Start a QR pulse scan for 1 second (or until detection)
        self._start_pulse_qr()
        # Return True to indicate touch handled
        return True

    def _start_pulse_qr(self):
        # Cancel any existing scheduled end-of-pulse
        if getattr(self, "_qr_pulse_event", None):
            try:
                self._qr_pulse_event.cancel()
            except Exception:
                pass
            self._qr_pulse_event = None

        self._qr_temp_active = True
        # Update UI immediately
        self._set_qr_ui(None, None, note="QR: scanning…")
        # Schedule end of pulse after 1 second (store event so we can cancel it early)
        try:
            self._qr_pulse_event = Clock.schedule_once(lambda *_: self._end_pulse_qr(), 1.0)
        except Exception:
            def _bg_end():
                time.sleep(1.0)
                Clock.schedule_once(lambda *_: self._end_pulse_qr(), 0)
            threading.Thread(target=_bg_end, daemon=True).start()

    def _end_pulse_qr(self, *args):
        # Cancel the scheduled Clock event if it still exists
        if getattr(self, "_qr_pulse_event", None):
            try:
                self._qr_pulse_event.cancel()
            except Exception:
                pass
            self._qr_pulse_event = None

        # Turn off the temporary pulse flag
        self._qr_temp_active = False

        # Restore UI state depending on permanent qr_enabled flag
        if not getattr(self, "qr_enabled", False):
            self._set_qr_ui(None, None, note="QR: off")
        else:
            self._set_qr_ui(None, None, note="QR: on")

    # ---------- thumbnail selection / overlay behavior ----------
    def _on_thumb_touch(self, image_widget, touch):
        if not image_widget.collide_point(*touch.pos):
            return False
        idx = getattr(image_widget, "thumb_index", None)
        if idx is None:
            return False
        if idx >= len(self._thumb_paths):
            return False

        ccapi_path = self._thumb_paths[idx]
        # Display low-res thumbnail stretched over preview immediately
        if idx < len(self._thumb_textures):
            thumb_tex = self._thumb_textures[idx]
            self._select_image_overlay(ccapi_path, thumb_tex, idx)
        else:
            # no texture yet — fetch thumbnail then overlay
            threading.Thread(target=self._download_thumb_and_overlay, args=(ccapi_path, idx), daemon=True).start()
        return True

    def _download_thumb_and_overlay(self, ccapi_path, idx):
        # helper used when a thumb hasn't been loaded but user tapped it
        self._download_thumb_for_path(ccapi_path)
        # attempt to get texture for idx
        if idx < len(self._thumb_textures):
            tex = self._thumb_textures[idx]
            Clock.schedule_once(lambda *_: self._select_image_overlay(ccapi_path, tex, idx), 0)

    def _select_image_overlay(self, ccapi_path: str, thumb_texture: Texture, thumb_index: int):
        """
        Put the low-res thumb onto the preview (stretched), retain overlays.
        Start background fetch of full-res jpg and replace when ready.
        """
        # Enter overlay mode so liveview frames do not replace the selected image
        self._overlay_active = True
        self._overlay_thumb_index = thumb_index

        # Apply thumb texture immediately on main thread
        Clock.schedule_once(lambda *_: self.preview.set_texture(thumb_texture), 0)

        # Background fetch full-res JPG
        threading.Thread(target=self._fetch_full_and_replace, args=(ccapi_path, thumb_index), daemon=True).start()

    def _fetch_full_and_replace(self, ccapi_path: str, thumb_index: int):
        """
        Download full-res JPG (background), create a texture, then replace the preview
        and the thumb in the strip with the high-res texture.
        """
        full_url = f"https://{self.camera_ip}{ccapi_path}"
        try:
            resp = self._session.get(full_url, timeout=20.0, stream=True)
            if resp.status_code != 200 or not resp.content:
                self._log_internal(f"Full image download failed: {resp.status_code}")
                return
            data = resp.content
        except Exception as e:
            self._log_internal(f"Full image download err: {e}")
            return

        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self._log_internal("cv2.imdecode returned None for full image")
                return

            # rotate to match preview orientation
            rot = getattr(self.preview, "preview_rotation", 0) % 360 if hasattr(self, "preview") else 0
            if rot == 90:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            elif rot == 270:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb_bytes = rgb.tobytes()

            # create Kivy texture on main thread and apply
            def _apply_full(_dt):
                try:
                    tex = Texture.create(size=(w, h), colorfmt="rgb")
                    tex.flip_vertical()
                    tex.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
                except Exception as e:
                    self._log_internal(f"Full texture create err: {e}")
                    return

                # put it on preview (replacing low-res thumb)
                try:
                    self.preview.set_texture(tex)
                except Exception:
                    pass

                # replace thumbnail texture in strip as feedback (scaled down by Kivy)
                try:
                    if thumb_index < len(self._thumb_textures):
                        self._thumb_textures[thumb_index] = tex
                        if thumb_index < len(self._thumb_images):
                            self._thumb_images[thumb_index].texture = tex
                except Exception:
                    pass

                # exit overlay mode so liveview resumes updating preview afterwards
                self._overlay_active = False

            Clock.schedule_once(_apply_full, 0)

        except Exception as e:
            self._log_internal(f"Full image decode err: {e}")
            return

    # ---------- remaining methods (thumbnail download, poller, etc.) ----------
    def list_all_images(self):
        images = []
        status, root = self._json_call("GET", '/ccapi/ver120/contents', None, timeout=8.0)
        self._log_internal(f"/ccapi/ver120/contents -> {status}")
        if not status.startswith("200") or not root or "path" not in root:
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

    def _download_thumb_for_path(self, ccapi_path: str):
        """
        Downloads thumbnail bytes and schedules a main-thread task to create a Kivy texture
        and update UI. Rotates thumbnail to match preview orientation.
        """
        thumb_url = f"https://{self.camera_ip}{ccapi_path}?kind=thumbnail"
        self._log_internal(f"Downloading thumbnail (bg): {thumb_url}")
        try:
            resp = self._session.get(thumb_url, stream=True, timeout=10.0)
            self._log_internal(f"thumb status={resp.status_code} {resp.reason}")
            if resp.status_code != 200:
                return
            thumb_bytes = resp.content
        except Exception as e:
            self._log_internal(f"Thumbnail download error: {e}")
            return

        # Save to disk (optional)
        try:
            os.makedirs(self.thumb_dir, exist_ok=True)
            name = os.path.basename(ccapi_path) or "image"
            if not name.lower().endswith(('.jpg', '.jpeg')):
                name = name + '.jpg'
            out_path = os.path.join(self.thumb_dir, name)
            with open(out_path, "wb") as f:
                f.write(thumb_bytes)
            self._log_internal(f"Saved thumbnail {out_path}")
        except Exception as e:
            self._log_internal(f"Saving thumbnail err: {e}")

        # Decode into RGB bytes using PIL on background thread, rotate to match preview, then schedule UI update
        try:
            pil = PILImage.open(BytesIO(thumb_bytes)).convert("RGB")
            # rotate to match preview orientation if preview exists
            rot = getattr(self.preview, "preview_rotation", 0) if hasattr(self, "preview") else 0
            if rot:
                pil = pil_rotate_90s(pil, rot)
            pil.thumbnail((200, 200))
            w, h = pil.size
            rgb_bytes = pil.tobytes()
        except Exception as e:
            self._log_internal(f"Thumbnail decode err (bg): {e}")
            return

        def _make_texture_and_update(_dt, rgb_bytes=rgb_bytes, w=w, h=h, ccapi_path=ccapi_path):
            try:
                tex = Texture.create(size=(w, h), colorfmt="rgb")
                tex.flip_vertical()
                tex.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
            except Exception as e:
                self._log_internal(f"Texture create/blit err: {e}")
                return

            # Insert at front (thumb0 is newest)
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

    def _background_download_latest(self):
        self.download_and_thumbnail_latest()

    def download_and_thumbnail_latest(self):
        if not self.connected:
            self._log_internal("Not connected; cannot fetch contents.")
            return

        images = self.list_all_images()
        self._log_internal(f"contents: {len(images)} total entries")
        if not images:
            self._log_internal("No images found on camera.")
            return

        jpgs = [p for p in images if p.lower().endswith(('.jpg', '.jpeg'))]
        if not jpgs:
            self._log_internal("No JPG files found.")
            return

        latest = jpgs[-1]
        threading.Thread(target=self._download_thumb_for_path, args=(latest,), daemon=True).start()
        self._last_seen_image = latest

    def start_polling_new_images(self):
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._log_internal(f"Starting image poller every {self.poll_interval_s}s (background thread)")
        self._poll_thread_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)
        self._poll_thread.start()

    def stop_polling_new_images(self):
        if self._poll_thread is None:
            return
        self._log_internal("Stopping image poller (background thread)")
        self._poll_thread_stop.set()
        self._poll_thread = None

    def _poll_worker(self):
        while not self._poll_thread_stop.is_set():
            try:
                images = self.list_all_images()
                if images:
                    jpgs = [p for p in images if p.lower().endswith(('.jpg', '.jpeg'))]
                    if jpgs:
                        if self._last_seen_image is None:
                            self._last_seen_image = jpgs[-1]
                            self._log_internal(f"Poll (bg): baseline set to {self._last_seen_image}")
                        else:
                            new_start_idx = None
                            for idx, path in enumerate(jpgs):
                                if path == self._last_seen_image:
                                    new_start_idx = idx + 1
                                    break
                            if new_start_idx is None:
                                self._log_internal("Poll (bg): last_seen not found, resetting baseline")
                                self._last_seen_image = jpgs[-1]
                            else:
                                new_items = jpgs[new_start_idx:]
                                for path in new_items:
                                    self._log_internal(f"Poll (bg): New image detected: {path}")
                                    # only download thumbnails by default
                                    threading.Thread(target=self._download_thumb_for_path, args=(path,), daemon=True).start()
                                    self._last_seen_image = path
            except Exception as e:
                self._log_internal(f"Poll worker error: {e}")

            stop_event = self._poll_thread_stop
            stop_event.wait(self.poll_interval_s)

    def dump_ccapi(self):
        status, data = self._json_call("GET", '/ccapi', None, timeout=10.0)
        self._log_internal(f"/ccapi status={status}")
        try:
            j = json.dumps(data, indent=2)
        except Exception:
            j = str(data)
        self._log_internal("=== ccapi JSON START ===")
        for line in j.splitlines():
            self._log_internal(line)
        self._log_internal("=== ccapi JSON END ===")

    def _open_csv_filechooser(self):
        if platform != 'android':
            self._log_internal("CSV load is Android-only (SAF). Please run on-device to load CSV.")
            return
        return self._open_csv_saf()

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
