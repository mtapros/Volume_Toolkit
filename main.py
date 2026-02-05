# Android-focused Volume Toolkit (threaded decoder + background poller)
#
# v2.1.1 (FULL)
#
# This is the feature-complete line (CSV + Subject List + AutoFetch + QR + thumb review + log overlay),
# with the v2.1.1 fix:
#   Full-res fetch primary endpoint now matches the known-working iteration:
#     Primary:  GET https://<camera-ip><ccapi_path>
#     Fallback: GET https://<camera-ip>/ccapi/ver100/contents/<sdpath> (url-encoded)
#
# v2.1.0 behaviors retained:
# - QR Detect button uses On/Off color scheme (green/yellow on, red/white off).
# - Live preview placement locked (no drag). Zoom within fixed window via UV crop.
# - Same zoom for thumbnail review.
# - Active reviewed thumb highlighted (green border).
# - Close review by tapping same thumb again (toggle). (Preview tap does not close.)
# - Bottom log overlay (non-modal) with auto-scroll.
# - Stills rotate + center-crop to portrait 2:3 to fill preview exactly.
#
# Note on pinch zoom:
# - This build includes a reliable zoom gesture: drag up/down on the right 25% of the preview.
#   (True 2-finger pinch can be added with touch tracking if needed.)
#
import os
import threading
import time
from datetime import datetime
from io import BytesIO
import csv
import queue
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

    preview_rotation = NumericProperty(270)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Texture drawn on a Rectangle so we can zoom by changing UVs.
        self._tex_rect = None
        self._tex = None

        self.zoom = 1.0
        self.zoom_min = 1.0
        self.zoom_max = 4.0

        with self.canvas:
            Color(0, 0, 0, 1)
            self._bg = Rectangle(pos=self.pos, size=self.size)

        with self.canvas.after:
            self._c_border = Color(0.2, 0.6, 1.0, 1.0)
            self._ln_border = Line(width=2)

            self._c_grid = Color(1.0, 0.6, 0.0, 0.85)
            self._ln_grid_list = []

            self._c_57 = Color(1.0, 0.2, 0.2, 0.95)
            self._ln_57 = Line(width=2)

            self._c_810 = Color(1.0, 0.9, 0.2, 0.95)
            self._ln_810 = Line(width=2)

            self._c_oval = Color(0.7, 0.2, 1.0, 0.95)
            self._ln_oval = Line(width=2)

        self.bind(pos=self._redraw, size=self._redraw)
        self.bind(
            show_border=self._redraw, show_grid=self._redraw, show_57=self._redraw,
            show_810=self._redraw, show_oval=self._redraw,
            grid_n=self._redraw,
            oval_cx=self._redraw, oval_cy=self._redraw, oval_w=self._redraw, oval_h=self._redraw
        )
        self._redraw()

    def reset_zoom(self):
        self.zoom = 1.0
        self._update_tex_uv()

    def set_texture(self, texture: Texture):
        self._tex = texture
        if self._tex_rect is None:
            with self.canvas:
                self._tex_rect = Rectangle(texture=self._tex, pos=self.pos, size=self.size)
        else:
            self._tex_rect.texture = self._tex
        self._update_tex_uv()
        self._redraw()

    def _update_tex_uv(self):
        if not self._tex_rect or not self._tex:
            return
        z = max(self.zoom_min, min(float(self.zoom), self.zoom_max))
        self.zoom = z
        w = 1.0 / z
        h = 1.0 / z
        u0 = (1.0 - w) / 2.0
        v0 = (1.0 - h) / 2.0
        u1 = u0 + w
        v1 = v0 + h
        self._tex_rect.tex_coords = (u0, v0, u1, v0, u1, v1, u0, v1)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super().on_touch_up(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_move(touch)

        # Reliable zoom control: drag up/down on right edge region
        x, _y = touch.pos
        rx = self.x + self.width * 0.75
        if x >= rx:
            dy = touch.dy
            self.zoom = max(self.zoom_min, min(self.zoom_max, self.zoom + dy * 0.01))
            self._update_tex_uv()
            return True
        return True

    def _drawn_rect(self):
        return (self.x, self.y, self.width, self.height)

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
        self._bg.pos = self.pos
        self._bg.size = self.size
        if self._tex_rect is not None:
            self._tex_rect.pos = self.pos
            self._tex_rect.size = self.size

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
                self.canvas.after.remove(line)
            except Exception:
                pass
        self._ln_grid_list = []

        if self.show_grid and n >= 2:
            for i in range(1, n):
                x = fx + fw * (i / n)
                line = Line(points=[x, fy, x, fy + fh], width=2)
                self.canvas.after.add(line)
                self._ln_grid_list.append(line)
            for i in range(1, n):
                y = fy + fh * (i / n)
                line = Line(points=[fx, y, fx + fw, y], width=2)
                self.canvas.after.add(line)
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
    capture_type = StringProperty(CaptureType.JPG)
    STILL_TARGET_ASPECT = 2.0 / 3.0  # portrait

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.connected = False
        self.camera_ip = "172.25.162.76"

        self.live_running = False
        self.session_started = False

        self._lock = threading.Lock()
        self._last_decoded_ts = 0.0

        self._display_event = None
        self._fetch_thread = None

        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        self._log_lines = []
        self._max_log_lines = 400

        self._frame_texture = None
        self._frame_size = None

        # QR
        self.qr_enabled = False
        self.qr_interval_s = 0.15
        self.qr_new_gate_s = 0.70
        self._qr_detector = cv2.QRCodeDetector()
        self._qr_thread = None
        self._qr_seen = set()
        self._qr_last_add_time = 0.0

        self._latest_decoded_bgr = None
        self._latest_decoded_bgr_ts = 0.0

        # Decoder queue (threaded decode)
        self._decode_queue = queue.Queue(maxsize=2)
        self._decoder_stop = threading.Event()
        self._decoder_thread = threading.Thread(target=self._decoder_loop, daemon=True)
        self._decoder_thread.start()

        # Payloads
        self.author_max_chars = 60
        self._author_update_in_flight = False
        self._last_committed_author = None

        self._current_payload = ""  # QR
        self._csv_payload = ""      # CSV
        self._current_exif = ""     # camera author

        # CSV
        self.csv_headers = []
        self.csv_rows = []
        self.selected_headers = []
        self._subject_popup = None

        # Thumbs + review
        self._thumb_textures = []
        self._thumb_images = []
        self._thumb_paths = []
        self.thumb_dir = "thumbs"
        self._active_thumb_index = None
        self._thumb_border_lines = []

        # AutoFetch
        self.autofetch_enabled = False
        self._last_seen_image = None
        self._poll_thread = None
        self._poll_thread_stop = threading.Event()
        self.poll_interval_s = 2.0

        # HTTP
        self._session = requests.Session()
        self._session.verify = False

        # Android SAF
        self._android_activity_bound = False
        self._csv_req_code = 4242

        # Freeze state
        self._freeze_active = False
        self._freeze_ccapi_path = None
        self._freeze_request_id = 0

        # UI refs / popups
        self.dropdown = None

        # Log overlay
        self._log_overlay_visible = False
        self._log_overlay_label = None
        self._log_overlay_sv = None

    # ----------------- logging -----------------
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]
        self._append_log_overlay()

    def _clear_log(self):
        self._log_lines = []
        self._append_log_overlay()

    def _append_log_overlay(self):
        if self._log_overlay_label is None:
            return
        self._log_overlay_label.text = "\n".join(self._log_lines)
        if self._log_overlay_sv is not None:
            Clock.schedule_once(lambda *_: setattr(self._log_overlay_sv, "scroll_y", 0.0), 0)

    def _set_log_overlay_visible(self, visible: bool):
        self._log_overlay_visible = bool(visible)
        if not hasattr(self, "log_overlay"):
            return
        self.log_overlay.opacity = 1.0 if self._log_overlay_visible else 0.0
        self.log_overlay.disabled = not self._log_overlay_visible
        self.log_overlay.height = dp(220) if self._log_overlay_visible else 0

    # ----------------- exit -----------------
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

    # ----------------- styling -----------------
    @staticmethod
    def _set_btn_style(btn: Button, bg_rgba, fg_rgba):
        btn.background_normal = ""
        btn.background_down = ""
        btn.background_color = bg_rgba
        btn.color = fg_rgba

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

    # ----------------- build -----------------
    def build(self):
        outer = FloatLayout()
        main = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8), size_hint=(1, 1))
        outer.add_widget(main)

        header = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        self.exit_btn = Button(text="Exit", size_hint=(None, 1), width=dp(90), font_size=sp(14))
        header.add_widget(self.exit_btn)
        header.add_widget(Label(text="Volume Toolkit v2.1.1", font_size=sp(18)))
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

        row_exif = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(40))
        row_exif.add_widget(Label(text="Current EXIF", size_hint=(None, 1), width=dp(130), font_size=sp(14)))
        self.exif_label = Label(text="(not connected)", size_hint=(1, 1), font_size=sp(14),
                                halign="left", valign="middle")
        self.exif_label.bind(size=lambda *_: setattr(self.exif_label, "text_size", (self.exif_label.width, None)))
        row_exif.add_widget(self.exif_label)
        main.add_widget(row_exif)

        row_payload = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(40))
        row_payload.add_widget(Label(text="QR Payload", size_hint=(None, 1), width=dp(130), font_size=sp(14)))
        self.payload_label = Label(text="(none)", size_hint=(1, 1), font_size=sp(14),
                                   halign="left", valign="middle")
        self.payload_label.bind(size=lambda *_: setattr(self.payload_label, "text_size", (self.payload_label.width, None)))
        row_payload.add_widget(self.payload_label)
        main.add_widget(row_payload)

        row_csv_payload = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(28))
        row_csv_payload.add_widget(Label(text="CSV Payload", size_hint=(None, 1), width=dp(130), font_size=sp(13)))
        self.csv_payload_label = Label(text="(none)", size_hint=(1, 1), font_size=sp(13),
                                       halign="left", valign="middle")
        self.csv_payload_label.bind(size=lambda *_: setattr(self.csv_payload_label, "text_size", (self.csv_payload_label.width, None)))
        row_csv_payload.add_widget(self.csv_payload_label)
        main.add_widget(row_csv_payload)

        self.metrics = Label(text="Delay: -- ms | Fetch: 0 | Decode: 0 | Display: 0")

        main_area = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, 0.6))
        preview_holder = AnchorLayout(anchor_x="center", anchor_y="center", size_hint=(0.80, 1))
        self.preview = PreviewOverlay(size_hint=(None, None))
        preview_holder.add_widget(self.preview)
        main_area.add_widget(preview_holder)

        sidebar = BoxLayout(orientation="vertical", size_hint=(0.20, 1), spacing=dp(4))
        sidebar.add_widget(Label(text="Last 5", size_hint=(1, None), height=dp(20), font_size=sp(12)))

        self._thumb_images = []
        self._thumb_border_lines = []
        for idx in range(5):
            img = Image(size_hint=(1, 0.18), allow_stretch=True, keep_ratio=True)
            img.thumb_index = idx
            img.bind(on_touch_down=self._on_thumb_touch)

            with img.canvas.after:
                Color(0.2, 1.0, 0.2, 1.0)
                ln = Line(rectangle=(0, 0, 0, 0), width=3)

            img.bind(pos=lambda *_: self._update_thumb_borders(), size=lambda *_: self._update_thumb_borders())

            sidebar.add_widget(img)
            self._thumb_images.append(img)
            self._thumb_border_lines.append(ln)

        main_area.add_widget(sidebar)
        main.add_widget(main_area)

        bottom = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, None), height=dp(52))
        self.push_btn = Button(text="Push Payload", font_size=sp(16), disabled=True)
        self.subject_btn = Button(text="Subject List", font_size=sp(16), disabled=True)
        bottom.add_widget(self.push_btn)
        bottom.add_widget(self.subject_btn)
        main.add_widget(bottom)

        def fit_preview(*_):
            w = max(dp(220), preview_holder.width * 0.98)
            h = max(dp(220), preview_holder.height * 0.98)
            self.preview.size = (w, h)

        preview_holder.bind(pos=fit_preview, size=fit_preview)

        # Log overlay
        self.log_overlay = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            height=0,
            pos_hint={"x": 0, "y": 0},
            spacing=dp(4),
            padding=[dp(6), dp(6), dp(6), dp(6)],
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
        self._log_overlay_label = Label(text="", size_hint_y=None, halign="left", valign="top", font_size=sp(10), color=(1, 1, 1, 1))
        self._log_overlay_label.bind(width=lambda *_: setattr(self._log_overlay_label, "text_size", (self._log_overlay_label.width, None)))
        self._log_overlay_label.bind(texture_size=lambda *_: setattr(self._log_overlay_label, "height", self._log_overlay_label.texture_size[1]))
        self._log_overlay_sv.add_widget(self._log_overlay_label)
        self.log_overlay.add_widget(self._log_overlay_sv)

        outer.add_widget(self.log_overlay)
        self._set_log_overlay_visible(False)

        btn_close.bind(on_release=lambda *_: self._set_log_overlay_visible(False))
        btn_clear.bind(on_release=lambda *_: self._clear_log())

        # Bind actions
        self.exit_btn.bind(on_press=lambda *_: self.exit_app())
        self.connect_btn.bind(on_press=lambda *_: self.connect_camera())
        self.start_btn.bind(on_press=lambda *_: self.toggle_liveview())
        self.autofetch_btn.bind(on_press=lambda *_: self.toggle_autofetch())
        self.qr_btn.bind(on_press=lambda *_: self.toggle_qr_detect())
        self.push_btn.bind(on_press=lambda *_: self.push_payload())
        self.subject_btn.bind(on_press=lambda *_: self.open_subject_list())
        self.subject_btn.disabled = True

        self.dropdown = self._build_dropdown(fit_preview)
        self.menu_btn.bind(on_release=lambda *_: self.dropdown.open(self.menu_btn))

        self._reschedule_display_loop(12)

        self._apply_connect_btn_style()
        self._apply_live_btn_style()
        self._apply_autofetch_btn_style()
        self._apply_qr_btn_style()
        self._set_controls_idle()

        self.log("Android CCAPI GUI ready")
        return outer

    # ----------------- HTTP -----------------
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

    # ----------------- connect -----------------
    def connect_camera(self):
        if self.live_running:
            self.log("Connect disabled while live view is running. Stop live view first.")
            return
        self.log(f"Connecting to {self.camera_ip}:443")
        st, data = self._json_call("GET", "/ccapi/ver100/deviceinformation", None, timeout=8.0)
        if st.startswith("200") and data:
            self.connected = True
            self.log("Connected OK")
            self.refresh_exif()
        else:
            self.connected = False
            self._current_exif = ""
            self.exif_label.text = f"(connect failed: {st})"
            self.log(f"Connect failed: {st}")
        self._set_controls_idle()

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

    # ----------------- EXIF -----------------
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

    # ----------------- live view -----------------
    def toggle_liveview(self):
        if not self.connected:
            return
        if self.live_running:
            self.stop_liveview()
        else:
            self.start_liveview()

    def start_liveview(self):
        payload = {"liveviewsize": "small", "cameradisplay": "on"}
        self.log("Starting live view")
        st, _ = self._json_call("POST", "/ccapi/ver100/shooting/liveview", payload, timeout=10.0)
        if not st.startswith("200"):
            self.log(f"Live view start failed: {st}")
            return

        self.session_started = True
        self.live_running = True
        self.preview.reset_zoom()
        self._apply_live_btn_style()

        self._fetch_thread = threading.Thread(target=self._liveview_fetch_loop, daemon=True)
        self._fetch_thread.start()

        self._qr_thread = threading.Thread(target=self._qr_loop, daemon=True)
        self._qr_thread.start()

    def stop_liveview(self):
        self.live_running = False
        if self.session_started:
            try:
                self._json_call("DELETE", "/ccapi/ver100/shooting/liveview", None, timeout=6.0)
            except Exception:
                pass
            self.session_started = False
        self._apply_live_btn_style()

    # ----------------- QR -----------------
    def toggle_qr_detect(self):
        self.qr_enabled = not bool(self.qr_enabled)
        self._apply_qr_btn_style()
        self.log(f"QR detect {'enabled' if self.qr_enabled else 'disabled'}")

    # ----------------- AutoFetch -----------------
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

    # ----------------- payload pushing (QR/CSV chooser) -----------------
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
                timeout=8.0,
            )
            if not st_put.startswith("200"):
                raise Exception(f"PUT failed: {st_put}")

            st_get, data = self._json_call(
                "GET",
                "/ccapi/ver100/functions/registeredname/author",
                None,
                timeout=8.0,
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

    # ----------------- metrics loop -----------------
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

    # ----------------- live fetch + decoder -----------------
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

                def apply(_dt, rgb_bytes=rgb_bytes, w=w, h=h, ts=ts):
                    if self._freeze_active:
                        return
                    if self._frame_texture is None or self._frame_size != (w, h):
                        tex = Texture.create(size=(w, h), colorfmt="rgb")
                        tex.flip_vertical()
                        self._frame_texture = tex
                        self._frame_size = (w, h)
                        self.log(f"texture init size={w}x{h}")
                    self._frame_texture.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
                    self.preview.set_texture(self._frame_texture)
                    self._last_decoded_ts = ts

                Clock.schedule_once(apply, 0)
                self._decode_count += 1

            except Exception:
                continue

    # ----------------- QR loop -----------------
    def _qr_loop(self):
        last_processed_ts = 0.0
        while self.live_running:
            if not self.qr_enabled:
                time.sleep(0.10)
                continue

            with self._lock:
                bgr = None
                ts = self._latest_decoded_bgr_ts
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
        Clock.schedule_once(lambda *_: self._set_qr_payload(text), 0)

    def _set_qr_payload(self, payload: str):
        self._current_payload = (payload or "").strip()
        self.payload_label.text = self._current_payload if self._current_payload else "(none)"

    # ----------------- thumbs -----------------
    def _update_thumb_borders(self):
        for idx, img in enumerate(self._thumb_images):
            ln = self._thumb_border_lines[idx]
            if self._active_thumb_index == idx and idx < len(self._thumb_paths):
                x, y = img.pos
                w, h = img.size
                ln.rectangle = (x, y, w, h)
            else:
                ln.rectangle = (0, 0, 0, 0)

    def _close_review(self):
        self._freeze_active = False
        self._freeze_ccapi_path = None
        self._freeze_request_id += 1
        self._active_thumb_index = None
        self._update_thumb_borders()
        self.preview.reset_zoom()
        self.log("Review closed (thumb toggle)")

        # Return to live texture if available
        if self._frame_texture is not None:
            try:
                self.preview.set_texture(self._frame_texture)
            except Exception:
                pass

    def _on_thumb_touch(self, image_widget, touch):
        if not image_widget.collide_point(*touch.pos):
            return False
        idx = getattr(image_widget, "thumb_index", None)
        if idx is None or idx >= len(self._thumb_paths):
            return False

        if self._active_thumb_index == idx and self._freeze_active:
            self._close_review()
            return True

        ccapi_path = self._thumb_paths[idx]
        thumb_tex = self._thumb_textures[idx]

        self._freeze_active = True
        self._freeze_ccapi_path = ccapi_path
        self._freeze_request_id += 1
        rid = self._freeze_request_id

        self._active_thumb_index = idx
        self._update_thumb_borders()

        self.preview.set_texture(thumb_tex)
        self.preview.reset_zoom()

        self.log(f"REVIEW START idx={idx} rid={rid} path={ccapi_path}")
        threading.Thread(target=self._freeze_pipeline_for_thumb, args=(ccapi_path, rid), daemon=True).start()
        return True

    def _freeze_pipeline_for_thumb(self, ccapi_path: str, request_id: int):
        # Step 1: fetch thumb bytes (so we can crop to 2:3 before showing, even if sidebar thumb is square-ish)
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
            self.log(f"REVIEW THUMB DECODE/CROP ERROR {e}")
            return

        def apply_thumb(_dt):
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
                self.preview.reset_zoom()
                self.log(f"REVIEW THUMB APPLY w={w} h={h}")
            except Exception as e:
                self.log(f"REVIEW THUMB APPLY ERROR {e}")

        Clock.schedule_once(apply_thumb, 0)

        # Step 2: full-res (v2.1.1 endpoint logic)
        if self.capture_type == CaptureType.JPG:
            self._download_fullres_and_replace(ccapi_path, request_id)
        else:
            self.log("RAW selected; full-res RAW fetch not implemented.")

    # v2.1.1 endpoint logic: primary + fallback
    def _fullres_primary_url(self, ccapi_path: str) -> str:
        return f"https://{self.camera_ip}{ccapi_path}"

    def _fullres_fallback_url(self, ccapi_path: str) -> str:
        prefix = "/ccapi/ver120/contents/"
        if ccapi_path.startswith(prefix):
            sd_path = ccapi_path[len(prefix):]
        else:
            sd_path = ccapi_path.lstrip("/")
        sd_path_enc = quote(sd_path, safe="/")
        return f"https://{self.camera_ip}/ccapi/ver100/contents/{sd_path_enc}"

    def _download_fullres_and_replace(self, ccapi_path: str, request_id: int):
        urls = [
            ("PRIMARY", self._fullres_primary_url(ccapi_path)),
            ("FALLBACK", self._fullres_fallback_url(ccapi_path)),
        ]

        data = None
        used = None
        for name, url in urls:
            try:
                self.log(f"FULLRES {name} START url={url}")
                resp = self._session.get(url, timeout=25.0, stream=True)
                if resp.status_code != 200 or not resp.content:
                    self.log(f"FULLRES {name} FAIL status={resp.status_code} bytes={len(resp.content) if resp.content else 0}")
                    continue
                data = resp.content
                used = name
                self.log(f"FULLRES {name} OK bytes={len(data)}")
                break
            except Exception as e:
                self.log(f"FULLRES {name} ERROR {e}")

        if data is None:
            self.log("FULLRES GIVEUP (no endpoint succeeded)")
            return

        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise Exception("cv2.imdecode returned None")
            bgr = self._rotate_bgr(bgr)
            bgr = self._center_crop_bgr_to_aspect(bgr, self.STILL_TARGET_ASPECT)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb_bytes = rgb.tobytes()
        except Exception as e:
            self.log(f"FULLRES {used} DECODE/CROP ERROR {e}")
            return

        def apply_full(_dt):
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
                self.log(f"FULLRES {used} APPLY ERROR {e}")

        Clock.schedule_once(apply_full, 0)

    def _download_thumb_for_path(self, ccapi_path: str):
        thumb_url = f"https://{self.camera_ip}{ccapi_path}?kind=thumbnail"
        self.log(f"Downloading sidebar thumbnail (bg): {thumb_url}")
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
            pil = PILImage.open(BytesIO(thumb_bytes)).convert("RGB")
            # Rotate for sidebar appearance (match preview rotation in 90° steps)
            rot = int(self.preview.preview_rotation) % 360
            if rot == 90:
                pil = pil.transpose(PILImage.Transpose.ROTATE_270)  # PIL ROTATE_270 is CCW 270 == CW 90
            elif rot == 180:
                pil = pil.transpose(PILImage.Transpose.ROTATE_180)
            elif rot == 270:
                pil = pil.transpose(PILImage.Transpose.ROTATE_90)   # CCW 90
            pil.thumbnail((240, 240))
            w, h = pil.size
            rgb_bytes = pil.tobytes()
        except Exception as e:
            self.log(f"Thumbnail decode/rotate err (bg): {e}")
            return

        def _apply(_dt):
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

            for i, img in enumerate(self._thumb_images):
                img.texture = self._thumb_textures[i] if i < len(self._thumb_textures) else None

            if self._active_thumb_index is not None and self._active_thumb_index >= len(self._thumb_paths):
                self._active_thumb_index = None
            self._update_thumb_borders()

        Clock.schedule_once(_apply, 0)

    # ----------------- contents listing -----------------
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

    # ----------------- CSV (Android SAF) -----------------
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
                take_flags = flags & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
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
        btn_ok.bind(on_release=lambda *_: (self.log(f"Selected headers: {self.selected_headers}"), popup.dismiss()))
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

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
        btn_close = Button(text="Close")
        btn_bar.add_widget(btn_close)
        root.add_widget(btn_bar)

        popup = Popup(title="Subject List", content=root, size_hint=(0.98, 0.98))

        def compute_rows():
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
                def ok(r):
                    for h, v in filters.items():
                        if v not in ((r.get(h) or "").lower()):
                            return False
                    return True
                rows = [r for r in rows if ok(r)]

            if search:
                headers = self.selected_headers if self.selected_headers else self.csv_headers
                def hit(r):
                    for h in headers:
                        if search in ((r.get(h) or "").lower()):
                            return True
                    return False
                rows = [r for r in rows if hit(r)]

            if sort_h:
                try:
                    rows = sorted(rows, key=lambda r: (r.get(sort_h) or "").lower(), reverse=sort_desc)
                except Exception:
                    pass
            return rows

        def render():
            results_inner.clear_widgets()
            rows = compute_rows()
            for r in rows[:200]:
                payload = self._build_csv_payload_from_row(r)
                btn = Button(text=payload[:120], size_hint_y=None, height=dp(44), font_size=sp(12))

                def pick(val=payload):
                    self._csv_payload = val
                    self.csv_payload_label.text = val if val else "(none)"
                    self.log(f"CSV payload selected: {val}")
                    popup.dismiss()
                    self._set_controls_idle()

                btn.bind(on_release=lambda *_: pick())
                results_inner.add_widget(btn)

        ti_search.bind(text=lambda *_: render())
        ti_sort.bind(text=lambda *_: render())
        cb_desc.bind(active=lambda *_: render())
        for ti in filter_inputs.values():
            ti.bind(text=lambda *_: render())

        btn_close.bind(on_release=lambda *_: popup.dismiss())

        popup.open()
        render()

    # ----------------- menu -----------------
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
        dd.max_height = dp(650)

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
        add_button("Reset zoom", lambda: self.preview.reset_zoom())

        add_header("Display")
        add_toggle("Log overlay", False, lambda v: self._set_log_overlay_visible(v))
        add_button("Clear log", lambda: self._clear_log())

        add_header("CSV")
        add_button("Load CSV…", lambda: self._open_csv_menu_popup())
        add_button("Subject List", lambda: self.open_subject_list())

        return dd

    # ----------------- lifecycle -----------------
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
