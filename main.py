# Volume Toolkit V1.0.5
#
# Android Kivy Canon CCAPI tool.
#
# V1.0.1 highlights:
# - Preview starts rotated 90° CCW.
# - Collapsible metrics drawer (collapsed by default).
# - Start button toggles Start/Stop (Stop button removed).
# - Thumbnails reflow:
#     Portrait  -> right rail (20% width), vertical scroll
#     Landscape -> bottom strip (20% height), horizontal scroll
# - Grid overlay diagonal bug fixed (draw independent segments); grid fixed 3x3.
# - Connection setup popup + persistent nickname profiles (JsonStore), IP only (assume https://:443).
# - Reduced QR load: slower interval + skip duplicate frames by JPEG timestamp.
# - CSV headers popup includes Sort/Filter options; Student picker shows same options.

import os
import csv
import json
import threading
import time
from datetime import datetime
from io import BytesIO

import urllib3
import requests

import kivy
kivy.require("2.0.0")

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, InstructionGroup, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.graphics.transformation import Matrix
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.storage.jsonstore import JsonStore
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.modalview import ModalView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserListView
from kivy.utils import platform

from PIL import Image as PILImage
import cv2
import numpy as np



APP_VERSION = '1.1.0'

# Android: keep Kivy writable files out of the extracted app directory (avoid permission errors).
if os.environ.get("ANDROID_ARGUMENT"):
    private_dir = os.environ.get("ANDROID_PRIVATE")
    if private_dir:
        kivy_home = os.path.join(private_dir, ".kivy")
        os.makedirs(kivy_home, exist_ok=True)
        os.environ["KIVY_HOME"] = kivy_home

# Suppress self-signed HTTPS warning from camera.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
        # Fallback for older Pillow
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

    grid_n = NumericProperty(3)  # fixed 3x3 for now

    # Purple oval defaults: w/h=1/3, centered horizontally, vertical center at 2/3.
    oval_cx = NumericProperty(0.5)
    oval_cy = NumericProperty(2.0 / 3.0)
    oval_w = NumericProperty(1.0 / 3.0)
    oval_h = NumericProperty(1.0 / 3.0)

    # Start rotated 90° CCW
    preview_rotation = NumericProperty(90)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.img = Image(allow_stretch=True, keep_ratio=True)
        try:
            self.img.fit_mode = "contain"
        except Exception:
            pass
        self.add_widget(self.img)

        self._lw = 2
        self._lw_qr = 6

        with self.img.canvas.after:
            self._c_border = Color(0.2, 0.6, 1.0, 1.0)
            self._ln_border = Line(width=self._lw)

            self._c_grid = Color(1.0, 0.6, 0.0, 0.85)

        self._grid_group = InstructionGroup()
        self.img.canvas.after.add(self._grid_group)

        with self.img.canvas.after:
            self._c_57 = Color(1.0, 0.2, 0.2, 0.95)
            self._ln_57 = Line(width=self._lw)

            self._c_810 = Color(1.0, 0.9, 0.2, 0.95)
            self._ln_810 = Line(width=self._lw)

            self._c_oval = Color(0.7, 0.2, 1.0, 0.95)
            self._ln_oval = Line(width=self._lw)

            self._c_qr = Color(0.2, 1.0, 0.2, 0.95)
            self._ln_qr = Line(width=self._lw_qr, close=True)

        self._qr_points_px = None

        self.bind(pos=self._redraw, size=self._redraw)
        self.bind(
            show_border=self._redraw,
            show_grid=self._redraw,
            show_57=self._redraw,
            show_810=self._redraw,
            show_oval=self._redraw,
            show_qr=self._redraw,
            grid_n=self._redraw,
            oval_cx=self._redraw,
            oval_cy=self._redraw,
            oval_w=self._redraw,
            oval_h=self._redraw,
        )
        self.img.bind(pos=self._redraw, size=self._redraw, texture=self._redraw, texture_size=self._redraw)

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

    @staticmethod
    def _clear_line_modes(ln: Line):
        try:
            ln.points = []
        except Exception:
            pass
        try:
            ln.rectangle = (0, 0, 0, 0)
        except Exception:
            pass

    def _redraw(self, *_args):
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

        # Grid: independent segments (fix diagonal artifacts)
        self._grid_group.clear()
        n = int(self.grid_n)
        if self.show_grid and n >= 2:
            for i in range(1, n):
                x = fx + fw * (i / n)
                self._grid_group.add(Line(points=[x, fy, x, fy + fh], width=self._lw))
            for i in range(1, n):
                y = fy + fh * (i / n)
                self._grid_group.add(Line(points=[fx, y, fx + fw, y], width=self._lw))

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

        # Connection
        self.connected = False
        self.camera_ip = "192.168.34.29"
        self.active_profile = ""

        # Live view
        self.live_running = False
        self.session_started = False
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_jpeg_ts = 0.0
        self._last_decoded_ts = 0.0

        self._fetch_thread = None
        self._qr_thread = None
        self._display_event = None

        # Stats
        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        # Log
        self._log_lines = []
        self._max_log_lines = 300
        self.show_log = True

        # Rendering
        self._frame_texture = None
        self._frame_size = None

        # Drawer
        self.drawer_open = False

        # QR
        self.qr_enabled = True
        self.qr_interval_s = 0.40
        self.qr_new_gate_s = 0.70
        self._qr_detector = cv2.QRCodeDetector()
        self._latest_qr_text = None
        self._latest_qr_points = None
        self._qr_seen = set()
        self._qr_last_add_time = 0.0
        self._last_qr_processed_ts = 0.0

        # Author
        self.author_max_chars = 60
        self._last_committed_author = None
        self._author_update_in_flight = False
        self.manual_payload = ""

        # CSV
        self.csv_headers = []
        self.csv_rows = []
        self.selected_headers = []

        # Student list sort/filter
        self.student_list_mode = "sort"   # "sort" or "filter"
        self.student_list_key = ""
        self.student_filter_text = ""

        # Thumbnails
        self._thumb_textures = []
        self._thumb_images = []
        self._thumb_paths = []

        # Storage / polling
        self.thumb_dir = "thumbs"
        self._last_seen_image = None
        self._poll_event = None
        self.poll_interval_s = 2.0

        # HTTPS session
        self._session = requests.Session()
        self._session.verify = False

        # Android SAF
        self._android_activity_bound = False
        self._csv_req_code = 4242

        # Persistent settings
        self._store = None

    # ---------------- Settings (profiles) ----------------

    def _settings_path(self):
        os.makedirs(self.user_data_dir, exist_ok=True)
        return os.path.join(self.user_data_dir, "settings.json")

    def _ensure_store(self):
        if self._store is not None:
            return
        try:
            self._store = JsonStore(self._settings_path())
        except Exception as e:
            self._store = None
            self.log(f"Settings init failed: {e}")

    def _get_profiles(self):
        self._ensure_store()
        if not self._store:
            return {}
        if self._store.exists("profiles"):
            try:
                return dict(self._store.get("profiles").get("items") or {})
            except Exception:
                return {}
        return {}

    def _set_profiles(self, profiles: dict):
        self._ensure_store()
        if self._store:
            self._store.put("profiles", items=profiles)

    def _get_active_profile_name(self):
        self._ensure_store()
        if not self._store:
            return ""
        if self._store.exists("active_profile"):
            try:
                return str(self._store.get("active_profile").get("name") or "")
            except Exception:
                return ""
        return ""

    def _set_active_profile_name(self, name: str):
        self._ensure_store()
        if self._store:
            self._store.put("active_profile", name=name)

    def _bootstrap_profiles(self):
        profiles = self._get_profiles()
        active = self._get_active_profile_name()

        if not profiles:
            profiles = {"Default": {"ip": self.camera_ip}}
            active = "Default"
            self._set_profiles(profiles)
            self._set_active_profile_name(active)

        if active and active in profiles:
            self.active_profile = active
            self.camera_ip = (profiles[active].get("ip") or self.camera_ip).strip()
        else:
            name = sorted(profiles.keys())[0]
            self.active_profile = name
            self.camera_ip = (profiles[name].get("ip") or self.camera_ip).strip()
            self._set_active_profile_name(name)

    # ---------------- UI ----------------

    def build(self):
        self._bootstrap_profiles()

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        header = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        header.add_widget(Label(text=f"Volume Toolkit {APP_VERSION}", font_size=sp(18)))
        root.add_widget(header)

        # Menu button in its own row (better touch detection on Android)
        self.menu_btn = Button(
            text="OPEN MENU",
            size_hint=(1, None),
            height=dp(50),
            font_size=sp(18),
            background_color=(0.2, 0.6, 0.9, 1)
        )
        root.add_widget(self.menu_btn)

        # Metrics drawer (collapsed by default)
        self.metrics_drawer = BoxLayout(orientation="vertical", size_hint=(1, None), height=0, spacing=dp(6))
        self.metrics_drawer.opacity = 0
        self.metrics_drawer.disabled = True

        self.metrics_inner = BoxLayout(orientation="vertical", size_hint=(1, None), spacing=dp(6))
        self.metrics_inner.bind(minimum_height=self.metrics_inner.setter("height"))
        self.metrics_drawer.add_widget(self.metrics_inner)

        # Connection row
        conn_row = BoxLayout(size_hint=(1, None), height=dp(38), spacing=dp(6))
        self.conn_label = Label(text="", font_size=sp(12), halign="left", valign="middle")
        self.conn_label.bind(size=self.conn_label.setter("text_size"))
        self.conn_setup_btn = Button(text="Setup…", size_hint=(None, 1), width=dp(110), font_size=sp(14))
        conn_row.add_widget(self.conn_label)
        conn_row.add_widget(self.conn_setup_btn)
        self.metrics_inner.add_widget(conn_row)

        # IP input in drawer
        self.ip_input = TextInput(text=self.camera_ip, multiline=False, font_size=sp(16), padding=[dp(10)] * 4)
        self.ip_input.size_hint = (1, None)
        self.ip_input.height = dp(44)
        self.metrics_inner.add_widget(self.ip_input)

        # Connect + Start toggle
        row2 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.connect_btn = Button(text="Connect", font_size=sp(16))
        self.start_btn = Button(text="Start", disabled=True, font_size=sp(16))
        row2.add_widget(self.connect_btn)
        row2.add_widget(self.start_btn)
        self.metrics_inner.add_widget(row2)

        # FPS
        row3 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(40))
        row3.add_widget(Label(text="Display FPS", size_hint=(None, 1), width=dp(110), font_size=sp(14)))
        self.fps_slider = Slider(min=5, max=30, value=12, step=1)
        self.fps_label = Label(text="12", size_hint=(None, 1), width=dp(50), font_size=sp(14))
        row3.add_widget(self.fps_slider)
        row3.add_widget(self.fps_label)
        self.metrics_inner.add_widget(row3)

        # Status / metrics / QR
        self.status = Label(text="Status: not connected", size_hint=(1, None), height=dp(22), font_size=sp(13))
        self.metrics = Label(
            text="Delay: -- ms | Fetch: 0 | Decode: 0 | Display: 0",
            size_hint=(1, None),
            height=dp(22),
            font_size=sp(13),
        )
        self.qr_status = Label(text="QR: none", size_hint=(1, None), height=dp(22), font_size=sp(13))
        self.metrics_inner.add_widget(self.status)
        self.metrics_inner.add_widget(self.metrics)
        self.metrics_inner.add_widget(self.qr_status)

        root.add_widget(self.metrics_drawer)

        # Main area
        self.main_area = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, 0.68))

        self.preview_holder = AnchorLayout(anchor_x="center", anchor_y="center")
        self.preview_holder.size_hint = (0.80, 1)

        self.preview_scatter = Scatter(do_translation=True, do_scale=True, do_rotation=False, scale_min=0.5, scale_max=2.5)
        self.preview_scatter.size_hint = (None, None)

        self.preview = PreviewOverlay(size_hint=(None, None))
        self.preview_scatter.add_widget(self.preview)
        self.preview_holder.add_widget(self.preview_scatter)
        self.main_area.add_widget(self.preview_holder)

        # Thumbs area: ScrollView + strip
        self.sidebar = BoxLayout(orientation="vertical", spacing=dp(4))
        self.sidebar.size_hint = (0.20, 1)

        self.thumb_scroll = ScrollView(size_hint=(1, 1))
        self.thumb_strip = BoxLayout(orientation="vertical", spacing=dp(4), size_hint=(1, None))
        self.thumb_strip.bind(minimum_height=self.thumb_strip.setter("height"))
        self.thumb_strip.bind(minimum_width=self.thumb_strip.setter("width"))
        self.thumb_scroll.add_widget(self.thumb_strip)

        for idx in range(5):
            img = Image(allow_stretch=True, keep_ratio=True)
            img.thumb_index = idx
            img.bind(on_touch_down=self._on_thumb_touch)
            self.thumb_strip.add_widget(img)
            self._thumb_images.append(img)

        self.sidebar.add_widget(self.thumb_scroll)
        self.main_area.add_widget(self.sidebar)
        root.add_widget(self.main_area)

        # Fit preview to holder + reset scatter transform
        def fit_preview_to_holder(*_):
            w = max(dp(220), self.preview_holder.width * 0.98)
            h = max(dp(220), self.preview_holder.height * 0.98)
            self.preview_scatter.size = (w, h)
            self.preview.size = (w, h)

            self.preview_scatter.transform = Matrix().identity()
            self.preview_scatter.scale = 1.0
            self.preview_scatter.pos = (
                self.preview_holder.x + (self.preview_holder.width - w) / 2.0,
                self.preview_holder.y + (self.preview_holder.height - h) / 2.0,
            )

        self.preview_holder.bind(pos=fit_preview_to_holder, size=fit_preview_to_holder)

        # Log panel
        self.log_holder = BoxLayout(orientation="vertical", size_hint=(1, None), height=dp(150))
        log_sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.log_label = Label(text="", size_hint_y=None, halign="left", valign="top", font_size=sp(11))
        self.log_label.bind(width=lambda *_: setattr(self.log_label, "text_size", (self.log_label.width, None)))
        self.log_label.bind(texture_size=lambda *_: setattr(self.log_label, "height", self.log_label.texture_size[1]))
        log_sv.add_widget(self.log_label)
        self.log_holder.add_widget(log_sv)
        root.add_widget(self.log_holder)

        # Menu
        self.menu_modal = self._build_menu_modal(fit_preview_to_holder)

        def open_menu(*_):
            self.log("[MENU] Button pressed")
            self.menu_modal.open()

        self.menu_btn.bind(on_press=open_menu)

        # Bindings
        self.conn_setup_btn.bind(on_release=lambda *_: self._open_connection_setup())
        self.connect_btn.bind(on_press=lambda *_: self.connect_camera())
        self.start_btn.bind(on_press=lambda *_: self._toggle_live())
        self.fps_slider.bind(value=self._on_fps_change)
        self._reschedule_display_loop(int(self.fps_slider.value))

        # Responsive reflow
        Window.bind(on_resize=self._on_window_resize)
        Clock.schedule_once(lambda *_: self._apply_ui_orientation(Window.width, Window.height), 0)

        # Drawer collapsed by default
        Clock.schedule_once(lambda *_: self._set_drawer_open(False, instant=True), 0)

        self._set_controls_idle()
        self._update_conn_label()
        self.log(f"Volume Toolkit v{APP_VERSION} ready")
        return root

    # ---------- Responsive layout ----------

    def _on_window_resize(self, _win, w, h):
        self._apply_ui_orientation(w, h)

    def _apply_ui_orientation(self, w, h):
        landscape = w > h

        if landscape:
            # Preview top, thumbs bottom (20% height)
            self.main_area.orientation = "vertical"
            self.preview_holder.size_hint = (1, 0.80)
            self.sidebar.size_hint = (1, 0.20)

            self.thumb_scroll.do_scroll_x = True
            self.thumb_scroll.do_scroll_y = False

            self.thumb_strip.orientation = "horizontal"
            self.thumb_strip.size_hint = (None, 1)

            thumb_w = max(dp(96), w * 0.20)
            for img in self._thumb_images:
                img.size_hint = (None, 1)
                img.width = thumb_w
        else:
            # Preview left, thumbs right (20% width)
            self.main_area.orientation = "horizontal"
            self.preview_holder.size_hint = (0.80, 1)
            self.sidebar.size_hint = (0.20, 1)

            self.thumb_scroll.do_scroll_x = False
            self.thumb_scroll.do_scroll_y = True

            self.thumb_strip.orientation = "vertical"
            self.thumb_strip.size_hint = (1, None)

            thumb_h = max(dp(80), h * 0.12)
            for img in self._thumb_images:
                img.size_hint = (1, None)
                img.height = thumb_h

    # ---------- Drawer ----------

    def _drawer_target_height(self):
        # metrics_inner.height tracks minimum_height due to binding
        return self.metrics_inner.height + dp(8)

    def _set_drawer_open(self, open_: bool, instant: bool = False):
        open_ = bool(open_)
        self.drawer_open = open_

        if open_:
            self.metrics_drawer.disabled = False
            self.metrics_drawer.opacity = 1
            target = self._drawer_target_height()
            if instant:
                self.metrics_drawer.height = target
                return
            Animation.cancel_all(self.metrics_drawer, "height")
            Animation(height=target, d=0.18).start(self.metrics_drawer)
        else:
            if instant:
                self.metrics_drawer.height = 0
                self.metrics_drawer.opacity = 0
                self.metrics_drawer.disabled = True
                return

            Animation.cancel_all(self.metrics_drawer, "height")

            def _finish(*_):
                self.metrics_drawer.opacity = 0
                self.metrics_drawer.disabled = True

            anim = Animation(height=0, d=0.18)
            anim.bind(on_complete=lambda *_: _finish())
            anim.start(self.metrics_drawer)

    def _toggle_drawer(self):
        self._set_drawer_open(not self.drawer_open, instant=False)

    # ---------- Logging / HTTPS ----------

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines :]
        if hasattr(self, "log_label"):
            self.log_label.text = "\n".join(self._log_lines)

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

    # ---------- Menu UI ----------

    def _set_log_visible(self, visible: bool):
        self.show_log = bool(visible)
        self.log_holder.height = dp(150) if self.show_log else 0
        self.log_holder.opacity = 1 if self.show_log else 0
        self.log_holder.disabled = not self.show_log

    def _style_menu_button(self, b: Button):
        b.background_normal = ""
        b.background_down = ""
        b.background_color = (0.10, 0.10, 0.10, 0.80)
        b.color = (1, 1, 1, 1)
        return b

    def _build_menu_modal(self, reset_callback):
        # Create the menu content using DropDown structure
        dd = DropDown(auto_dismiss=True)
        dd.auto_width = False
        dd.width = min(dp(380), Window.width * 0.92)
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
            cb.bind(active=lambda _inst, val: on_change(val))
            row.add_widget(cb)
            dd.add_widget(row)

        def add_capture_type_buttons():
            row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4), padding=[dp(4), 0, dp(4), 0])
            row.add_widget(Label(text="Capture:", size_hint=(None, 1), width=dp(70), font_size=sp(13), color=(1, 1, 1, 1)))

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
            row.add_widget(make_btn("Both", CaptureType.BOTH))
            dd.add_widget(row)

        add_header("Metrics")
        add_button("Toggle metrics drawer", self._toggle_drawer)
        add_button("Connection setup…", self._open_connection_setup)

        add_header("Framing")
        add_button("Reset framing", reset_callback)

        add_header("Overlays")
        add_toggle("Border (blue)", True, lambda v: setattr(self.preview, "show_border", v))
        add_toggle("Grid (orange)", True, lambda v: setattr(self.preview, "show_grid", v))
        add_toggle("Crop 5:7 (red)", True, lambda v: setattr(self.preview, "show_57", v))
        add_toggle("Crop 8:10 (yellow)", True, lambda v: setattr(self.preview, "show_810", v))
        add_toggle("Oval (purple)", True, lambda v: setattr(self.preview, "show_oval", v))
        add_toggle("QR overlay", True, lambda v: setattr(self.preview, "show_qr", v))

        add_header("QR & Author")
        add_toggle("QR detect (OpenCV)", True, lambda v: self._set_qr_enabled(v))
        add_button("Load CSV…", self._open_csv_filechooser)
        add_button("Select headers…", self._open_headers_popup)
        add_button("Student picker…", self._open_student_picker)
        add_button("Push payload (Author)", lambda: self._maybe_commit_author(self.manual_payload, source="manual"))

        add_header("Capture")
        add_capture_type_buttons()
        add_button("Fetch latest image", self.download_and_thumbnail_latest)
        add_button("Start auto-fetch", self.start_polling_new_images)
        add_button("Stop auto-fetch", self.stop_polling_new_images)

        add_header("UI")
        add_toggle("Show log", True, lambda v: self._set_log_visible(v))

        # Wrap dropdown in ModalView (this is the fix!)
        modal = ModalView(size_hint=(0.95, 0.95), auto_dismiss=True)
        modal.add_widget(dd)
        return modal


    def _update_conn_label(self):
        if not hasattr(self, "conn_label"):
            return
        nick = self.active_profile.strip() if self.active_profile else "No profile"
        ip = self.camera_ip.strip() if self.camera_ip else "No IP"
        self.conn_label.text = f"Camera: {nick}  |  {ip}:443"

    def _open_connection_setup(self):
        profiles = self._get_profiles()

        root = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))
        root.add_widget(Label(text="Connection setup (HTTPS :443 assumed)", size_hint=(1, None), height=dp(24), font_size=sp(13)))

        nick_row = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        nick_row.add_widget(Label(text="Nickname", size_hint=(None, 1), width=dp(90), font_size=sp(12)))
        nick_input = TextInput(text=self.active_profile or "", multiline=False, font_size=sp(14))
        pick_btn = Button(text="Pick…", size_hint=(None, 1), width=dp(90), font_size=sp(12))
        nick_row.add_widget(nick_input)
        nick_row.add_widget(pick_btn)
        root.add_widget(nick_row)

        ip_row = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        ip_row.add_widget(Label(text="Camera IP", size_hint=(None, 1), width=dp(90), font_size=sp(12)))
        ip_input = TextInput(text=self.camera_ip or "", multiline=False, font_size=sp(14))
        ip_row.add_widget(ip_input)
        root.add_widget(ip_row)

        msg = Label(text="", size_hint=(1, None), height=dp(24), font_size=sp(12))
        root.add_widget(msg)

        btns = BoxLayout(size_hint=(1, None), height=dp(44), spacing=dp(6))
        btn_use = Button(text="Use", font_size=sp(14))
        btn_save = Button(text="Save", font_size=sp(14))
        btn_cancel = Button(text="Cancel", font_size=sp(14))
        btns.add_widget(btn_use)
        btns.add_widget(btn_save)
        btns.add_widget(btn_cancel)
        root.add_widget(btns)

        dd = DropDown(auto_dismiss=True)
        dd.auto_width = False
        dd.width = min(dp(320), Window.width * 0.88)

        def rebuild_picklist():
            dd.clear_widgets()
            keys = sorted((profiles or {}).keys())
            if not keys:
                b = Button(text="(no saved profiles)", size_hint_y=None, height=dp(36), font_size=sp(12))
                self._style_menu_button(b)
                dd.add_widget(b)
                return
            for name in keys:
                b = Button(text=name, size_hint_y=None, height=dp(40), font_size=sp(13))
                self._style_menu_button(b)

                def _select(n=name):
                    nick_input.text = n
                    ip_input.text = (profiles.get(n, {}).get("ip") or "").strip()
                    dd.dismiss()

                b.bind(on_release=lambda *_i, n=name: _select(n))
                dd.add_widget(b)

        rebuild_picklist()
        pick_btn.bind(on_release=lambda *_: dd.open(pick_btn))

        popup = Popup(title="Connection setup", content=root, size_hint=(0.92, 0.62))

        def apply_use(save_active: bool):
            nick = (nick_input.text or "").strip()
            ip = (ip_input.text or "").strip()
            if not ip:
                msg.text = "IP is required."
                return False

            self.camera_ip = ip
            self.ip_input.text = ip

            if save_active and nick and nick in profiles:
                self.active_profile = nick
                self._set_active_profile_name(nick)

            self._update_conn_label()
            return True

        def do_use(*_):
            nick = (nick_input.text or "").strip()
            ok = apply_use(save_active=bool(nick and nick in profiles))
            if ok:
                popup.dismiss()

        def do_save(*_):
            nick = (nick_input.text or "").strip()
            ip = (ip_input.text or "").strip()
            if not nick:
                msg.text = "Nickname is required to Save."
                return
            if not ip:
                msg.text = "IP is required."
                return

            profiles2 = self._get_profiles()
            profiles2[nick] = {"ip": ip}
            self._set_profiles(profiles2)

            self.active_profile = nick
            self._set_active_profile_name(nick)

            # Apply the saved values too
            self.camera_ip = ip
            self.ip_input.text = ip
            self._update_conn_label()

            msg.text = f"Saved profile '{nick}'."
            popup.dismiss()

        btn_use.bind(on_release=do_use)
        btn_save.bind(on_release=do_save)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    # ---------- Controls ----------

    def _set_controls_idle(self):
        self.ip_input.disabled = False
        self.connect_btn.disabled = False
        self.start_btn.disabled = not self.connected
        self.start_btn.text = "Start"
        self.conn_setup_btn.disabled = False

    def _set_controls_running(self):
        self.ip_input.disabled = True
        self.connect_btn.disabled = True
        self.start_btn.disabled = False
        self.start_btn.text = "Stop"
        self.conn_setup_btn.disabled = True

    def _toggle_live(self):
        if self.live_running:
            self.stop_live_view()
        else:
            self.start_live_view()

    def _on_fps_change(self, *_):
        fps = int(self.fps_slider.value)
        self.fps_label.text = str(fps)
        self._reschedule_display_loop(fps)

    def _reschedule_display_loop(self, fps):
        if self._display_event is not None:
            self._display_event.cancel()
        fps = max(1, int(fps))
        self._display_event = Clock.schedule_interval(self._ui_decode_and_display, 1.0 / fps)

    # ---------- Connect ----------

    def connect_camera(self):
        if self.live_running:
            self.log("Connect disabled while live view is running. Stop first.")
            return

        self.camera_ip = (self.ip_input.text or "").strip()
        if not self.camera_ip:
            self.status.text = "Status: enter an IP"
            return

        self._update_conn_label()
        self.status.text = f"Status: connecting to {self.camera_ip}:443…"
        self.log(f"Connecting to {self.camera_ip}:443")

        status, data = self._json_call("GET", "/ccapi/ver100/deviceinformation", None, timeout=8.0)
        if status.startswith("200") and isinstance(data, dict):
            self.connected = True
            prod = (data.get("productname") or "camera").strip()
            self.status.text = f"Status: connected ({prod})"
            self.log("Connected OK")
        else:
            self.connected = False
            self.status.text = f"Status: connect failed ({status})"
            self.log(f"Connect failed: {status}")

        self._set_controls_idle()

    # ---------- Live view ----------

    def start_live_view(self):
        if not self.connected or self.live_running:
            return

        payload = {"liveviewsize": "small", "cameradisplay": "on"}
        self.log("Starting live view (size=small, cameradisplay=on)")
        status, _ = self._json_call("POST", "/ccapi/ver100/shooting/liveview", payload, timeout=10.0)
        if not status.startswith("200"):
            self.status.text = f"Status: live view start failed ({status})"
            self.log(f"Live view start failed: {status}")
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

        # QR state
        self._latest_qr_text = None
        self._latest_qr_points = None
        self._qr_seen = set()
        self._qr_last_add_time = 0.0
        self._last_qr_processed_ts = 0.0
        self._set_qr_ui(None, None, note="QR: none")

        # stats
        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        self._fetch_thread = threading.Thread(target=self._live_view_fetch_loop, daemon=True)
        self._fetch_thread.start()

        self._qr_thread = threading.Thread(target=self._qr_loop, daemon=True)
        self._qr_thread.start()

    def stop_live_view(self):
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

        self.status.text = "Status: connected (live stopped)" if self.connected else "Status: not connected"
        self.log("Live view stopped")
        self._set_controls_idle()

    def _live_view_fetch_loop(self):
        url = f"https://{self.camera_ip}/ccapi/ver100/shooting/liveview/flip"
        while self.live_running:
            try:
                resp = self._session.get(url, timeout=5.0)
                if resp.status_code == 200 and resp.content:
                    with self._lock:
                        self._latest_jpeg = resp.content
                        self._latest_jpeg_ts = time.time()
                    self._fetch_count += 1
                else:
                    time.sleep(0.03)
            except Exception as e:
                self.log(f"Liveview fetch error: {e}")
                time.sleep(0.10)

    # ---------- QR ----------

    def _set_qr_enabled(self, enabled: bool):
        self.qr_enabled = bool(enabled)
        if not self.qr_enabled:
            self._set_qr_ui(None, None, note="QR: off")

    def _qr_loop(self):
        while self.live_running:
            if not self.qr_enabled:
                time.sleep(0.15)
                continue

            with self._lock:
                jpeg = self._latest_jpeg
                jpeg_ts = self._latest_jpeg_ts

            # Skip if no frame or already processed this frame timestamp
            if not jpeg or jpeg_ts <= self._last_qr_processed_ts:
                time.sleep(0.05)
                continue

            self._last_qr_processed_ts = jpeg_ts

            try:
                pil = PILImage.open(BytesIO(jpeg)).convert("RGB")
                pil = pil_rotate_90s(pil, self.preview.preview_rotation)
                rgb = np.array(pil)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                decoded, points, _ = self._qr_detector.detectAndDecode(bgr)
                text = decoded.strip() if isinstance(decoded, str) else ""
                qrpoints = None

                if points is not None:
                    try:
                        pts = points.astype(int).reshape(-1, 2)
                        if len(pts) == 4:
                            qrpoints = [(int(pts[i][0]), int(pts[i][1])) for i in range(4)]
                    except Exception:
                        qrpoints = None

                if text or qrpoints:
                    self._publish_qr(text if text else None, qrpoints)
            except Exception:
                pass

            time.sleep(max(0.05, float(self.qr_interval_s)))

    def _publish_qr(self, text, points):
        now = time.time()

        if text:
            if text not in self._qr_seen and (now - self._qr_last_add_time) >= self.qr_new_gate_s:
                self._qr_seen.add(text)
                self._qr_last_add_time = now
                self.log(f"QR: {text}")
                self._maybe_commit_author(text, source="qr")

        if not self.qr_enabled:
            note = "QR: off"
        elif text:
            note = f"QR: {text[:80]}"
        elif points:
            note = "QR: detected (undecoded)"
        else:
            note = "QR: none"

        Clock.schedule_once(lambda *_: self._set_qr_ui(text, points, note=note), 0)

    def _set_qr_ui(self, text, points, note="QR: none"):
        if text:
            self._latest_qr_text = text
        if points:
            self._latest_qr_points = points
        self.preview.set_qr(points)
        self.qr_status.text = note

    # ---------- UI decode/display loop ----------

    def _ui_decode_and_display(self, _dt):
        if not self.live_running:
            return

        with self._lock:
            jpeg = self._latest_jpeg
            jpeg_ts = self._latest_jpeg_ts

        self._display_count += 1

        if (not jpeg) or (jpeg_ts <= self._last_decoded_ts):
            self._update_metrics(jpeg_ts)
            return

        try:
            pil = PILImage.open(BytesIO(jpeg)).convert("RGB")
            pil = pil_rotate_90s(pil, self.preview.preview_rotation)

            w, h = pil.size
            rgb_bytes = pil.tobytes()

            if self._frame_texture is None or self._frame_size != (w, h):
                tex = Texture.create(size=(w, h), colorfmt="rgb")
                tex.flip_vertical()
                self._frame_texture = tex
                self._frame_size = (w, h)
                self.log(f"Texture init: {w}x{h}")

            self._frame_texture.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
            self.preview.set_texture(self._frame_texture)

            self._decode_count += 1
            self._last_decoded_ts = jpeg_ts
        except Exception as e:
            self.log(f"UI decode error: {e}")

        self._update_metrics(jpeg_ts)

    def _update_metrics(self, frame_ts):
        now = time.time()
        if now - self._stat_t0 >= 1.0:
            dts = now - self._stat_t0
            fetchfps = self._fetch_count / dts
            decfps = self._decode_count / dts
            dispfps = self._display_count / dts
            delayms = int((now - frame_ts) * 1000) if frame_ts else -1

            delay_str = f"{delayms} ms" if delayms >= 0 else "-- ms"
            self.metrics.text = f"Delay: {delay_str} | Fetch: {fetchfps:.1f} | Decode: {decfps:.1f} | Display: {dispfps:.1f}"

            self._fetch_count = 0
            self._decode_count = 0
            self._display_count = 0
            self._stat_t0 = now

    # ---------- Author push ----------

    def _author_value(self, payload: str) -> str:
        s = (payload or "").strip()
        if not s:
            return ""
        return s[: int(self.author_max_chars)]

    def _maybe_commit_author(self, payload: str, source: str):
        value = self._author_value(payload)
        if not value:
            return
        if not self.connected:
            self.log(f"Author update skipped ({source}): not connected")
            return
        if self._last_committed_author == value:
            return
        if self._author_update_in_flight:
            return

        self._author_update_in_flight = True
        Clock.schedule_once(lambda *_: setattr(self.qr_status, "text", f"Author: updating ({source})"), 0)

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
                timeout=8.0,
            )
            if not st_put.startswith("200"):
                raise Exception(f"PUT failed {st_put}")

            st_get, data = self._json_call("GET", "/ccapi/ver100/functions/registeredname/author", None, timeout=8.0)
            if (not st_get.startswith("200")) or (not isinstance(data, dict)):
                raise Exception(f"GET failed {st_get}")

            got = (data.get("author") or "").strip()
            ok = (got == value)
        except Exception as e:
            err = str(e)

        def finish(_dt):
            self._author_update_in_flight = False
            if ok:
                self._last_committed_author = value
                self.log(f"Author updated/verified ({source}): {value}")
                self.qr_status.text = "Author: updated"
            else:
                self.log(f"Author verify failed ({source}). wrote={value} read={got} err={err}")
                self.qr_status.text = "Author: verify failed"

        Clock.schedule_once(finish, 0)

    # ---------- CSV (load + headers + student picker) ----------

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

            self.log("Opening Android file picker (SAF)")
            mActivity.startActivityForResult(intent, self._csv_req_code)
        except Exception as e:
            self.log(f"Failed to open Android picker: {e}")

    def _on_android_activity_result(self, requestcode, resultcode, intent):
        if requestcode != getattr(self, "_csv_req_code", 4242):
            return
        if resultcode != -1 or intent is None:
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
                takeflags = flags & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
                mActivity.getContentResolver().takePersistableUriPermission(uri, takeflags)
            except Exception:
                pass

            data = self._read_android_uri_bytes(uri)
            self._parse_csv_bytes(data)
            self.log(f"CSV loaded (Android): {len(self.csv_rows)} rows")
        except Exception as e:
            self.log(f"CSV load failed (Android): {e}")

    def _read_android_uri_bytes(self, uri):
        from android import mActivity
        cr = mActivity.getContentResolver()
        stream = cr.openInputStream(uri)
        if stream is None:
            raise Exception("openInputStream returned null")

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
        if platform == "android":
            return self._open_csv_saf()

        content = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(6))
        chooser = FileChooserListView(filters=["*.csv"], size_hint=(1, 1))
        content.add_widget(chooser)

        statuslbl = Label(text="Pick a CSV file", size_hint=(1, None), height=dp(24), font_size=sp(12))
        content.add_widget(statuslbl)

        btns = BoxLayout(size_hint=(1, None), height=dp(36), spacing=dp(6))
        btnok = Button(text="Load")
        btncancel = Button(text="Cancel")
        btns.add_widget(btnok)
        btns.add_widget(btncancel)
        content.add_widget(btns)

        popup = Popup(title="Load CSV", content=content, size_hint=(0.9, 0.9))

        def doload(*_):
            if not chooser.selection:
                statuslbl.text = "No file selected"
                return
            path = chooser.selection[0]
            try:
                with open(path, "rb") as f:
                    data = f.read()
                self._parse_csv_bytes(data)
                statuslbl.text = f"Loaded {len(self.csv_rows)} rows"
                popup.dismiss()
            except Exception as e:
                statuslbl.text = f"Error: {e}"
                self.log(f"CSV load error: {e}")

        btnok.bind(on_release=doload)
        btncancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

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

        preferred = ["LASTNAME", "FIRSTNAME", "GRADE", "TEACHER", "STUDENTID"]
        if not self.selected_headers:
            sel = [h for h in preferred if h in headers]
            self.selected_headers = sel if sel else (headers[:3] if headers else [])

        # Default sort key if available
        if not self.student_list_key:
            self.student_list_key = "LASTNAME" if "LASTNAME" in headers else (headers[0] if headers else "")

    def _build_sort_filter_controls(self, parent, on_change_callback):
        """
        Adds shared Sort/Filter controls to a parent layout and wires changes to on_change_callback().
        Returns (filter_row, key_button) so callers can show/hide filter_row and update key_button text.
        """
        controls = BoxLayout(size_hint=(1, None), height=dp(36), spacing=dp(6))
        controls.add_widget(Label(text="Mode", size_hint=(None, 1), width=dp(50), font_size=sp(12)))

        updating = {"flag": False}

        cb_sort = CheckBox(active=(self.student_list_mode != "filter"), size_hint=(None, 1), width=dp(32))
        controls.add_widget(cb_sort)
        controls.add_widget(Label(text="Sort", size_hint=(None, 1), width=dp(40), font_size=sp(12)))

        cb_filter = CheckBox(active=(self.student_list_mode == "filter"), size_hint=(None, 1), width=dp(32))
        controls.add_widget(cb_filter)
        controls.add_widget(Label(text="Filter", size_hint=(None, 1), width=dp(45), font_size=sp(12)))

        key_btn = Button(text=f"Key: {self.student_list_key or '(none)'}", size_hint=(1, 1), font_size=sp(12))
        controls.add_widget(key_btn)
        parent.add_widget(controls)

        filter_row = BoxLayout(size_hint=(1, None), height=dp(36), spacing=dp(6))
        filter_row.add_widget(Label(text="Filter text", size_hint=(None, 1), width=dp(90), font_size=sp(12)))
        filter_input = TextInput(text=self.student_filter_text or "", multiline=False, font_size=sp(12))
        filter_row.add_widget(filter_input)
        parent.add_widget(filter_row)

        def update_filter_visibility():
            show = (self.student_list_mode == "filter")
            filter_row.opacity = 1 if show else 0
            filter_row.disabled = not show

        def set_mode(mode: str):
            updating["flag"] = True
            self.student_list_mode = mode
            cb_sort.active = (mode != "filter")
            cb_filter.active = (mode == "filter")
            updating["flag"] = False
            update_filter_visibility()
            on_change_callback()

        def on_sort_active(_inst, val):
            if updating["flag"]:
                return
            if val:
                set_mode("sort")

        def on_filter_active(_inst, val):
            if updating["flag"]:
                return
            if val:
                set_mode("filter")

        cb_sort.bind(active=on_sort_active)
        cb_filter.bind(active=on_filter_active)

        def on_filter_text(_inst, val):
            self.student_filter_text = filter_input.text
            if self.student_list_mode == "filter":
                on_change_callback()

        filter_input.bind(text=on_filter_text)

        # Key dropdown
        key_dd = DropDown(auto_dismiss=True)
        for h in (self.csv_headers or []):
            b = Button(text=h, size_hint_y=None, height=dp(36), font_size=sp(12))
            self._style_menu_button(b)
            b.bind(on_release=lambda btn, header=h: key_dd.select(header))
            key_dd.add_widget(b)

        def on_key_select(_dd, header):
            self.student_list_key = header
            key_btn.text = f"Key: {header}"
            on_change_callback()

        key_dd.bind(on_select=on_key_select)
        key_btn.bind(on_release=lambda *_: key_dd.open(key_btn))

        update_filter_visibility()
        return filter_row, key_btn

    def _open_headers_popup(self):
        if not self.csv_headers:
            self.log("No CSV loaded; cannot pick headers.")
            return

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        root.add_widget(Label(text="Select columns to include in Author (joined with ', ')", size_hint=(1, None), height=dp(36), font_size=sp(12)))

        # Sort/filter controls live here too
        def _changed():
            pass

        self._build_sort_filter_controls(root, on_change_callback=_changed)

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

            def toggle(_cb, val, header=h):
                if val:
                    if header not in self.selected_headers:
                        self.selected_headers.append(header)
                else:
                    if header in self.selected_headers:
                        self.selected_headers.remove(header)

            cb.bind(active=toggle)
            row.add_widget(lbl)
            row.add_widget(cb)
            inner.add_widget(row)

        root.add_widget(sv)

        btns = BoxLayout(size_hint=(1, None), height=dp(36), spacing=dp(6))
        btnok = Button(text="OK")
        btncancel = Button(text="Cancel")
        btns.add_widget(btnok)
        btns.add_widget(btncancel)
        root.add_widget(btns)

        popup = Popup(title="Select CSV columns", content=root, size_hint=(0.92, 0.92))

        def dook(*_):
            self.log(f"Selected headers: {self.selected_headers}")
            popup.dismiss()

        btnok.bind(on_release=dook)
        btncancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    def _student_row_display(self, row: dict) -> str:
        # Prefer a readable line, fallback to first few selected headers.
        keys = self.csv_headers or []
        def get(k): return (row.get(k) or "").strip()

        if "LASTNAME" in keys and "FIRSTNAME" in keys:
            base = f"{get('LASTNAME')}, {get('FIRSTNAME')}".strip(", ").strip()
            extras = []
            if "GRADE" in keys and get("GRADE"):
                extras.append(f"G{get('GRADE')}")
            if "TEACHER" in keys and get("TEACHER"):
                extras.append(get("TEACHER"))
            if "STUDENTID" in keys and get("STUDENTID"):
                extras.append(get("STUDENTID"))
            if extras:
                return f"{base}  ({' | '.join(extras)})"
            return base

        show = self.selected_headers or keys[:3]
        return " | ".join((get(k) for k in show if get(k))) or "(empty row)"

    def _apply_student_list_mode(self, rows):
        key = (self.student_list_key or "").strip()
        out = list(rows or [])

        if self.student_list_mode == "filter":
            needle = (self.student_filter_text or "").strip().lower()
            if needle and key:
                out = [r for r in out if needle in ((r.get(key) or "").lower())]
            elif needle:
                # if no key, filter across all fields
                out = [r for r in out if needle in " ".join((str(v or "") for v in r.values())).lower()]

        # Sort (always try a stable sort if key exists)
        if key:
            out.sort(key=lambda r: (r.get(key) or "").lower())
        return out

    def _build_student_payload(self, row: dict) -> str:
        parts = []
        for h in (self.selected_headers or []):
            v = (row.get(h) or "").strip()
            if v:
                parts.append(v)
        return ", ".join(parts)

    def _open_student_picker(self):
        if not self.csv_rows:
            self.log("No CSV loaded; cannot open student picker.")
            return

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))

        selected_lbl = Label(text="Selected: none", size_hint=(1, None), height=dp(22), font_size=sp(12))
        root.add_widget(selected_lbl)

        state = {"selected_row": None}

        def rebuild_list():
            inner.clear_widgets()
            rows = self._apply_student_list_mode(self.csv_rows)
            max_rows = 400
            shown = rows[:max_rows]
            for r in shown:
                line = self._student_row_display(r)
                b = Button(text=line, size_hint_y=None, height=dp(36), font_size=sp(11))
                b.halign = "left"

                def select_row(_btn, row=r):
                    state["selected_row"] = row
                    payload = self._build_student_payload(row)
                    self.manual_payload = payload
                    selected_lbl.text = f"Selected: {self._student_row_display(row)}"

                b.bind(on_release=select_row)
                inner.add_widget(b)

            if len(rows) > max_rows:
                inner.add_widget(Label(text=f"(showing first {max_rows} of {len(rows)})", size_hint_y=None, height=dp(22), font_size=sp(11)))

        # Sort/filter controls (same as headers popup)
        self._build_sort_filter_controls(root, on_change_callback=rebuild_list)

        sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        inner = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        inner.bind(minimum_height=inner.setter("height"))
        sv.add_widget(inner)
        root.add_widget(sv)

        btns = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        push_btn = Button(text="Push to Author")
        close_btn = Button(text="Close")
        btns.add_widget(push_btn)
        btns.add_widget(close_btn)
        root.add_widget(btns)

        popup = Popup(title="Students", content=root, size_hint=(0.95, 0.95))

        def do_push(*_):
            if not state["selected_row"]:
                self.log("No student selected; nothing to push.")
                return
            payload = self._build_student_payload(state["selected_row"])
            self.manual_payload = payload
            self._maybe_commit_author(payload, source="student")

        push_btn.bind(on_release=do_push)
        close_btn.bind(on_release=lambda *_: popup.dismiss())

        rebuild_list()
        popup.open()

    # ---------- Thumbnails / contents ----------

    def _list_all_images(self):
        images = []
        status, root = self._json_call("GET", "/ccapi/ver120/contents", None, timeout=8.0)
        self.log(f"/ccapi/ver120/contents -> {status}")
        if (not status.startswith("200")) or (not isinstance(root, dict)) or ("path" not in root):
            return images

        for p in root.get("path") or []:
            st_dir, dirs = self._json_call("GET", p, None, timeout=8.0)
            if (not st_dir.startswith("200")) or (not isinstance(dirs, dict)) or ("path" not in dirs):
                continue

            for d in dirs.get("path") or []:
                st_num, num = self._json_call("GET", f"{d}?kind=number", None, timeout=8.0)
                if (not st_num.startswith("200")) or (not isinstance(num, dict)) or ("pagenumber" not in num):
                    continue

                pages = int(num.get("pagenumber") or 0)
                for page in range(1, pages + 1):
                    st_files, fdata = self._json_call("GET", f"{d}?page={page}", None, timeout=8.0)
                    if (not st_files.startswith("200")) or (not isinstance(fdata, dict)) or ("path" not in fdata):
                        continue
                    for f in fdata.get("path") or []:
                        images.append(f)

        return images

    def _download_thumb_for_path(self, ccapipath: str):
        thumburl = f"https://{self.camera_ip}{ccapipath}?kind=thumbnail"
        self.log(f"Downloading thumbnail: {thumburl}")

        try:
            resp = self._session.get(thumburl, stream=True, timeout=10.0)
            self.log(f"Thumb status: {resp.status_code} {resp.reason}")
            if resp.status_code != 200:
                return
            thumbbytes = resp.content
        except Exception as e:
            self.log(f"Thumbnail download error: {e}")
            return

        # Save thumbnail to disk for inspection
        try:
            os.makedirs(self.thumb_dir, exist_ok=True)
            name = os.path.basename(ccapipath) or "image"
            if not name.lower().endswith((".jpg", ".jpeg")):
                name = name + ".jpg"
            outpath = os.path.join(self.thumb_dir, name)
            with open(outpath, "wb") as f:
                f.write(thumbbytes)
            self.log(f"Saved thumbnail: {outpath}")
        except Exception as e:
            self.log(f"Saving thumbnail error: {e}")

        # Decode to texture
        try:
            pil = PILImage.open(BytesIO(thumbbytes)).convert("RGB")
            pil.thumbnail((200, 200))
            w, h = pil.size
            tex = Texture.create(size=(w, h), colorfmt="rgb")
            tex.flip_vertical()
            tex.blit_buffer(pil.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        except Exception as e:
            self.log(f"Thumbnail decode error: {e}")
            return

        self._thumb_textures.insert(0, tex)
        self._thumb_paths.insert(0, ccapipath)
        self._thumb_textures = self._thumb_textures[:5]
        self._thumb_paths = self._thumb_paths[:5]

        def update(_dt):
            for idx, img in enumerate(self._thumb_images):
                img.texture = self._thumb_textures[idx] if idx < len(self._thumb_textures) else None

        Clock.schedule_once(update, 0)

    def download_and_thumbnail_latest(self):
        if not self.connected:
            self.log("Not connected; cannot fetch contents.")
            return
        images = self._list_all_images()
        self.log(f"Contents: {len(images)} total entries")
        if not images:
            self.log("No images found on camera.")
            return

        jpgs = [p for p in images if p.lower().endswith((".jpg", ".jpeg"))]
        if not jpgs:
            self.log("No JPG files found.")
            return

        latest = jpgs[-1]
        self._download_thumb_for_path(latest)
        self._last_seen_image = latest

    def start_polling_new_images(self):
        if self._poll_event is not None:
            return
        self.log(f"Starting image poller every {self.poll_interval_s}s")
        self._poll_event = Clock.schedule_interval(self._poll_new_images, self.poll_interval_s)

    def stop_polling_new_images(self):
        if self._poll_event is not None:
            self._poll_event.cancel()
            self._poll_event = None
        self.log("Image poller stopped")

    def _poll_new_images(self, _dt):
        if not self.connected:
            return

        images = self._list_all_images()
        if not images:
            return

        jpgs = [p for p in images if p.lower().endswith((".jpg", ".jpeg"))]
        if not jpgs:
            return

        if self._last_seen_image is None:
            self._last_seen_image = jpgs[-1]
            self.log(f"Poll baseline set to {self._last_seen_image}")
            return

        try:
            idx = jpgs.index(self._last_seen_image)
            new_items = jpgs[idx + 1 :]
        except ValueError:
            self.log("Poll: lastseen not found; resetting baseline")
            self._last_seen_image = jpgs[-1]
            return

        if not new_items:
            return

        for path in new_items:
            self.log(f"New image detected: {path}")
            self._download_thumb_for_path(path)
            self._last_seen_image = path

    # ---------- Thumb tap viewer ----------

    def _on_thumb_touch(self, imagewidget, touch):
        if not imagewidget.collide_point(*touch.pos):
            return False
        idx = getattr(imagewidget, "thumb_index", None)
        if idx is None or idx >= len(self._thumb_paths):
            return False
        ccapipath = self._thumb_paths[idx]
        tex = self._thumb_textures[idx]
        self._open_thumb_viewer(ccapipath, tex)
        return True

    def _open_thumb_viewer(self, ccapipath: str, texture: Texture):
        was_live = self.live_running
        if was_live:
            self.stop_live_view()

        scatter = Scatter(do_rotation=False, do_translation=True, do_scale=True)
        scatter.size_hint = (1, 1)
        img = Image(texture=texture, allow_stretch=True, keep_ratio=True)
        img.size_hint = (1, 1)
        scatter.add_widget(img)

        root = BoxLayout(orientation="vertical")
        root.add_widget(scatter)

        btnbar = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6), padding=[dp(6), dp(4), dp(6), dp(4)])
        label = Label(text=os.path.basename(ccapipath) or "Image", size_hint=(1, 1), font_size=sp(12))
        closebtn = Button(text="Close viewer", size_hint=(None, 1), width=dp(140))
        btnbar.add_widget(label)
        btnbar.add_widget(closebtn)
        root.add_widget(btnbar)

        popup = Popup(title="Image review (thumbnail)", content=root, size_hint=(0.95, 0.95))

        def close(*_):
            popup.dismiss()
            if was_live:
                self.start_live_view()

        closebtn.bind(on_release=close)
        popup.bind(on_dismiss=lambda *_: close())
        popup.open()

    # ---------- Lifecycle ----------

    def on_stop(self):
        try:
            self.stop_live_view()
        except Exception:
            pass
        try:
            self.stop_polling_new_images()
        except Exception:
            pass


if __name__ == "__main__":
    VolumeToolkitApp().run()
