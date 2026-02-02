# Android-focused Volume Toolkit (threaded decoder + background poller)
# Updated:
# - QR results now always override any CSV selection displayed in "Data Update:"
# - CSV controls moved under a bottom "CSV" button (Load CSV, Select Students, Select Headers)
# - Removed "Fetch latest image" from the main menu
# - Menu reorganized into sub-popups for Overlays, Capture, and Settings
# - Added an Autofetch toggle button on the main control row (behaves like Start/Stop)
# - CSV selection UI retained and integrated; QR will clear CSV selection when received
# - Defensive checks and request-id handling for full-res fetches retained

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

# Suppress insecure HTTPS warnings (camera may use self-signed certs)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# On Android prefer writing Kivy home to private dir to avoid permission issues
if os.environ.get("ANDROID_ARGUMENT"):
    private_dir = os.environ.get("ANDROID_PRIVATE")
    if private_dir:
        kivy_home = os.path.join(private_dir, ".kivy")
        os.makedirs(kivy_home, exist_ok=True)
        os.environ["KIVY_HOME"] = kivy_home


def pil_rotate_90s(img: PILImage.Image, ang: int) -> PILImage.Image:
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
        # Pillow compatibility fallback
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

    # Default rotation: device mounted rotated
    preview_rotation = NumericProperty(270)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Image widget used by live view (letterboxed when keep_ratio True)
        self.img = Image(allow_stretch=True, keep_ratio=True)
        try:
            self.img.fit_mode = "contain"
        except Exception:
            pass
        self.add_widget(self.img)

        lw = 2
        lw_qr = 6

        # Draw overlay lines into img.canvas.after so they appear above the image texture
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

            self._c_qr = Color(0.0, 1.0, 0.0, 0.95)
            self._ln_qr = Line(width=lw_qr, close=True)

        # dynamic list for grid lines so we can remove them cleanly
        self._ln_grid_list = []

        # Overlay rectangle state (we will create it in img.canvas.after so it sits above texture)
        self._overlay_texture = None
        self._overlay_rect = None
        self._overlay_rect_color = None

        # update overlay rect on moves/resizes
        self.bind(pos=self._update_overlay_rect, size=self._update_overlay_rect)
        self.img.bind(pos=self._update_overlay_rect, size=self._update_overlay_rect)

        self.bind(
            pos=self._redraw, size=self._redraw,
            show_border=self._redraw, show_grid=self._redraw, show_57=self._redraw,
            show_810=self._redraw, show_oval=self._redraw, show_qr=self._redraw,
            grid_n=self._redraw,
            oval_cx=self._redraw, oval_cy=self._redraw, oval_w=self._redraw, oval_h=self._redraw
        )

        self._qr_points_px = None
        self._redraw()

    # ---------- overlay rectangle helpers (drawn into img.canvas.after and sized to drawn rect) ----------
    def set_overlay_texture(self, texture: Texture):
        """
        Draw the given texture into a Rectangle that matches the drawn image bounds
        (letterboxed area). Place the Rectangle in the image's canvas.after so
        it appears above the image texture but underneath the overlay lines.
        """
        # Remove any previous overlay
        self.clear_overlay_texture()
        if texture is None:
            return

        # compute image drawn rect (x,y,w,h) where the Image actually displays its texture
        dx, dy, iw, ih = self._drawn_rect()
        self._overlay_texture = texture

        # Temporarily remove static overlay instructions from img.canvas.after so we can
        # add the rectangle first and then re-add the overlay lines (so lines draw on top).
        static_instr = [
            getattr(self, "_c_border", None), getattr(self, "_ln_border", None),
            getattr(self, "_c_57", None), getattr(self, "_ln_57", None),
            getattr(self, "_c_810", None), getattr(self, "_ln_810", None),
            getattr(self, "_c_oval", None), getattr(self, "_ln_oval", None),
            getattr(self, "_c_qr", None), getattr(self, "_ln_qr", None),
        ]
        try:
            for instr in static_instr:
                if instr is not None:
                    try:
                        self.img.canvas.after.remove(instr)
                    except Exception:
                        pass
        except Exception:
            pass

        # Create overlay rectangle inside img.canvas.after so it sits above image texture
        try:
            with self.img.canvas.after:
                self._overlay_rect_color = Color(1.0, 1.0, 1.0, 1.0)
                self._overlay_rect = Rectangle(texture=self._overlay_texture, pos=(dx, dy), size=(iw, ih))
        except Exception:
            # Fallback: if img.canvas.after fails, create in widget canvas.before (less ideal)
            try:
                with self.canvas.before:
                    self._overlay_rect_color = Color(1.0, 1.0, 1.0, 1.0)
                    self._overlay_rect = Rectangle(texture=self._overlay_texture, pos=(dx, dy), size=(iw, ih))
            except Exception:
                self._overlay_rect = None
                self._overlay_rect_color = None
                self._overlay_texture = None
                return

        # Re-add static overlay instructions so they draw after the rectangle (on top)
        try:
            for instr in static_instr:
                if instr is not None:
                    try:
                        self.img.canvas.after.add(instr)
                    except Exception:
                        pass
        except Exception:
            pass

    def clear_overlay_texture(self):
        # Remove overlay rectangle and its color safely from whichever canvas it lives in.
        try:
            if self._overlay_rect is not None:
                try:
                    self.img.canvas.after.remove(self._overlay_rect)
                except Exception:
                    try:
                        self.canvas.before.remove(self._overlay_rect)
                    except Exception:
                        pass
                self._overlay_rect = None
            if self._overlay_rect_color is not None:
                try:
                    self.img.canvas.after.remove(self._overlay_rect_color)
                except Exception:
                    try:
                        self.canvas.before.remove(self._overlay_rect_color)
                    except Exception:
                        pass
                self._overlay_rect_color = None
            self._overlay_texture = None
        except Exception:
            pass

    def _update_overlay_rect(self, *args):
        """
        Keep overlay rectangle aligned to the actual drawn image rect (letterboxed area).
        """
        if self._overlay_rect is None:
            return
        try:
            dx, dy, iw, ih = self._drawn_rect()
            self._overlay_rect.pos = (dx, dy)
            self._overlay_rect.size = (iw, ih)
        except Exception:
            pass

    # ---------- other preview helpers ----------
    def set_texture(self, texture):
        # Used by liveview: set the Image texture (letterboxed behavior)
        self.img.texture = texture
        self._redraw()

    def set_qr(self, points_px):
        self._qr_points_px = points_px
        self._redraw()

    def _drawn_rect(self):
        # Return the rectangle where Image would draw the texture (centered/letterboxed)
        wx, wy = self.img.pos
        ww, wh = self.img.size
        try:
            iw, ih = self.img.norm_image_size
        except Exception:
            # fallback to whole widget
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

        # remove previous grid elements
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

        # draw grid lines explicitly
        n = int(self.grid_n)
        if self.show_grid and n >= 2:
            for i in range(1, n):
                x = fx + fw * (i / n)
                col = Color(1.0, 0.6, 0.0, 0.85)
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

        # Default IP (editable in UI)
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

        # internal logging
        self._log_lines = []
        self._max_log_lines = 300
        self.show_log = False

        # frame texture currently used for liveview
        self._frame_texture = None
        self._frame_size = None

        self.dropdown = None

        # QR state
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

        # latest decoded BGR for QR thread
        self._latest_decoded_bgr = None
        self._latest_decoded_bgr_ts = 0.0

        # decoder queue + thread
        self._decode_queue = queue.Queue(maxsize=2)
        self._decoder_thread = None
        self._decoder_stop = threading.Event()

        # overlay state
        self._overlay_active = False
        self._overlay_thumb_index = None

        # thumbnail highlight
        self._highlighted_thumb_index = None
        self._thumb_highlight_lines = {}

        self._qr_highlight_event = None

        # author / CSV
        self.author_max_chars = 60
        self._last_committed_author = None
        self._author_update_in_flight = False
        self.manual_payload = ""

        self.csv_headers = []
        self.csv_rows = []
        # selected_headers doubles as the order/selection for author assembly
        self.selected_headers = []
        self._headers_popup = None

        # thumbnails
        self._thumb_textures = []
        self._thumb_images = []
        self._thumb_paths = []
        self._thumb_saved_paths = []

        # track pending full-res fetches to avoid stale-applying results
        # maps thumb_index -> request_id (float)
        self._pending_full_fetches = {}

        # storage
        self.download_dir = "downloads"
        self.thumb_dir = "thumbs"

        self._last_seen_image = None
        self._poll_thread = None
        self._poll_thread_stop = threading.Event()
        self.poll_interval_s = 2.0

        self.save_full_size = False

        # HTTP session (insecure certs allowed for camera)
        self._session = requests.Session()
        self._session.verify = False

        # Android SAF
        self._android_activity_bound = False
        self._csv_req_code = 4242

        # UI refs (populated in build)
        self.header = None
        self.preview_holder = None

        # CSV selection UI state
        self._column_filters = {}      # header -> filter string
        self._column_sorts = {}        # header -> None | 'asc' | 'desc' (single-column sort supported)
        self._selected_csv_row = None  # dict row selected by user
        self._selected_author_payload = None  # last assembled payload from CSV selection

        # Autofetch state (toggle on main UI)
        self.autofetch_running = False

    # ---------- texture helpers (main-thread creation) ----------
    def _create_texture_from_rgb(self, rgb_bytes, w, h, flip_vertical=True):
        try:
            tex = Texture.create(size=(w, h), colorfmt="rgb")
            if flip_vertical:
                tex.flip_vertical()
            tex.blit_buffer(rgb_bytes, colorfmt="rgb", bufferfmt="ubyte")
            return tex
        except Exception as e:
            self._log_internal(f"texture create error: {e}")
            return None

    def _create_texture_from_bgr_np(self, bgr, flip_vertical=True):
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            return self._create_texture_from_rgb(rgb.tobytes(), w, h, flip_vertical=flip_vertical)
        except Exception as e:
            self._log_internal(f"texture from bgr err: {e}")
            return None

    def _create_texture_from_jpeg_bytes(self, jpeg_bytes, rotate=0, flip_vertical=True):
        """
        Decode JPEG bytes (on the main thread) and create a Texture.
        Use sparingly; preferred path decodes in background then schedules this function
        to create the texture from RGB bytes.
        """
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            if rotate == 90:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            elif rotate == 270:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return self._create_texture_from_bgr_np(bgr, flip_vertical=flip_vertical)
        except Exception as e:
            self._log_internal(f"jpeg->texture err: {e}")
            return None

    def _create_texture_from_jpeg_file(self, path, rotate=0, flip_vertical=True):
        try:
            with open(path, "rb") as f:
                data = f.read()
            return self._create_texture_from_jpeg_bytes(data, rotate=rotate, flip_vertical=flip_vertical)
        except Exception as e:
            self._log_internal(f"jpeg file->texture err: {e}")
            return None

    # ---------- networking ----------
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

    # ---------- build / UI ----------
    def build(self):
        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        # Header
        self.header = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        self.header_title = Label(text="Volume Toolkit v1.0.6", font_size=sp(18))
        self.header.add_widget(self.header_title)
        self.menu_btn = Button(text="Menu", size_hint=(None, 1), width=dp(90), font_size=sp(16))
        self.header.add_widget(self.menu_btn)
        root.add_widget(self.header)

        # Control row: Connect / Start live / Autofetch toggle
        row2 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.connect_btn = Button(text="Connect", font_size=sp(16), size_hint=(None, 1), width=dp(120))
        self._style_connect_button(initial=True)
        self.start_btn = Button(text="Start", disabled=True, font_size=sp(16), size_hint=(None, 1), width=dp(120))
        self._style_start_button(stopped=True)
        self.autofetch_btn = Button(text="Autofetch Off", font_size=sp(14), size_hint=(None, 1), width=dp(140))
        self._style_autofetch_button(running=False)
        row2.add_widget(self.connect_btn)
        row2.add_widget(self.start_btn)
        row2.add_widget(self.autofetch_btn)
        root.add_widget(row2)

        # Data Update / status labels (renamed from QR to Data Update)
        # This label shows either QR result or CSV selected row summary
        self.qr_last_label = Label(text="Data Update: none", size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.qr_last_label)
        self.status = Label(text="Status: not connected", size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.status)
        self.qr_status = Label(text="", size_hint=(1, None), height=dp(18), font_size=sp(11))
        root.add_widget(self.qr_status)

        # Main area: preview (80%) + thumbs (20%)
        main_area = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, 0.6))

        # Preview holder and overlay layer
        self.preview_holder = AnchorLayout(anchor_x="center", anchor_y="center", size_hint=(0.80, 1))
        self.preview_scatter = Scatter(do_translation=False, do_scale=False, do_rotation=False, size_hint=(None, None))
        self.preview = PreviewOverlay(size_hint=(None, None))
        # bind the preview touch handler (method exists)
        self.preview.bind(on_touch_down=self._on_preview_touch)
        self.preview_scatter.add_widget(self.preview)
        self.preview_holder.add_widget(self.preview_scatter)

        main_area.add_widget(self.preview_holder)

        # Thumbnails sidebar
        sidebar = BoxLayout(orientation="vertical", size_hint=(0.20, 1), spacing=dp(4))
        sidebar.add_widget(Label(text="Last 5", size_hint=(1, None), height=dp(20), font_size=sp(12)))
        for idx in range(5):
            img = Image(size_hint=(1, None), height=dp(100), allow_stretch=True, keep_ratio=True)
            img.thumb_index = idx
            img.bind(on_touch_down=self._on_thumb_touch)
            img.bind(pos=self._make_thumb_pos_updater(idx), size=self._make_thumb_pos_updater(idx))
            sidebar.add_widget(img)
            self._thumb_images.append(img)
        main_area.add_widget(sidebar)

        root.add_widget(main_area)

        # Footer: CSV button and Force update button
        footer = BoxLayout(orientation="horizontal", size_hint=(1, None), height=dp(48), spacing=dp(6))
        footer.add_widget(Label())  # spacer to push buttons to the right side
        self.csv_btn = Button(text="CSV", size_hint=(None, 1), width=dp(140))
        self.force_update_btn = Button(text="Force update", size_hint=(None, 1), width=dp(140))
        footer.add_widget(self.csv_btn)
        footer.add_widget(self.force_update_btn)
        root.add_widget(footer)

        # Fit preview to holder
        def fit_preview_to_holder(*_):
            w = max(dp(220), self.preview_holder.width * 0.98)
            h = max(dp(220), self.preview_holder.height * 0.98)
            self.preview_scatter.size = (w, h)
            self.preview.size = (w, h)
            self.preview_scatter.pos = (
                self.preview_holder.x + (self.preview_holder.width - w) / 2.0,
                self.preview_holder.y + (self.preview_holder.height - h) / 2.0
            )
            # overlay rectangle must track preview drawn rect
            self.preview._update_overlay_rect()
            # update thumb highlight positions
            for idx in range(len(self._thumb_images)):
                self._update_thumb_highlight_pos(idx)

        self._fit_preview_to_holder = fit_preview_to_holder
        self.preview_holder.bind(pos=fit_preview_to_holder, size=fit_preview_to_holder)

        # Log area hidden by default
        self.log_holder = BoxLayout(orientation="vertical", size_hint=(1, None), height=0)
        log_sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.log_label = Label(text="", size_hint_y=None, halign="left", valign="top", font_size=sp(11))
        self.log_label.bind(width=lambda *_: setattr(self.log_label, "text_size", (self.log_label.width, None)))
        self.log_label.bind(texture_size=lambda *_: setattr(self.log_label, "height", self.log_label.texture_size[1]))
        log_sv.add_widget(self.log_label)
        self.log_holder.add_widget(log_sv)
        root.add_widget(self.log_holder)

        # Menu and bindings
        self.dropdown = self._build_dropdown()
        self.menu_btn.bind(on_release=lambda *_: self.dropdown.open(self.menu_btn))

        self.connect_btn.bind(on_press=lambda *_: self.connect_camera())
        self.start_btn.bind(on_press=lambda *_: self._on_start_pressed())
        self.autofetch_btn.bind(on_press=lambda *_: self._on_autofetch_pressed())
        self.csv_btn.bind(on_release=lambda *_: self._open_csv_menu())
        self.force_update_btn.bind(on_release=lambda *_: self._force_update_from_selection())

        # Start decoder thread now that preview exists (avoid race)
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

    # ---------- UI helpers ----------
    def _style_autofetch_button(self, running=False):
        try:
            self.autofetch_btn.background_normal = ""
            self.autofetch_btn.background_down = ""
            if running:
                self.autofetch_btn.background_color = (0.1, 0.6, 0.1, 1.0)
                self.autofetch_btn.text = "Autofetch On"
            else:
                self.autofetch_btn.background_color = (0.4, 0.4, 0.4, 1.0)
                self.autofetch_btn.text = "Autofetch Off"
        except Exception:
            pass

    def _on_autofetch_pressed(self):
        if not self.autofetch_running:
            # start poller
            self.start_polling_new_images()
            self.autofetch_running = True
            self._style_autofetch_button(running=True)
        else:
            self.stop_polling_new_images()
            self.autofetch_running = False
            self._style_autofetch_button(running=False)

    # ---------- helper to create per-thumb pos updaters ----------
    def _make_thumb_pos_updater(self, idx):
        def _updater(instance, value):
            self._update_thumb_highlight_pos(idx)
        return _updater

    # ---------- thumbnail highlight helpers ----------
    def _highlight_thumb(self, idx):
        self._clear_thumb_highlight()
        if idx is None or idx >= len(self._thumb_images):
            return
        img = self._thumb_images[idx]
        try:
            with img.canvas.after:
                col = Color(1.0, 0.8, 0.0, 1.0)  # yellow-ish
                ln = Line(rectangle=(img.x, img.y, img.width, img.height), width=2)
            self._thumb_highlight_lines[idx] = (col, ln)
            self._highlighted_thumb_index = idx
        except Exception:
            self._thumb_highlight_lines = {}
            self._highlighted_thumb_index = None

    def _update_thumb_highlight_pos(self, idx):
        if idx not in self._thumb_highlight_lines:
            return
        img = self._thumb_images[idx]
        col, ln = self._thumb_highlight_lines[idx]
        try:
            ln.rectangle = (img.x, img.y, img.width, img.height)
        except Exception:
            pass

    def _clear_thumb_highlight(self):
        for k, (col, ln) in list(self._thumb_highlight_lines.items()):
            try:
                widget = self._thumb_images[k]
                widget.canvas.after.remove(col)
                widget.canvas.after.remove(ln)
            except Exception:
                pass
        self._thumb_highlight_lines = {}
        self._highlighted_thumb_index = None

    # ---------- CSV menu (footer CSV button) ----------
    def _open_csv_menu(self):
        # Small popup with CSV actions
        content = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(8))
        b_load = Button(text="Load CSV (Android SAF)", size_hint=(1, None), height=dp(44))
        b_select = Button(text="Select Students", size_hint=(1, None), height=dp(44))
        b_headers = Button(text="Select Headers", size_hint=(1, None), height=dp(44))
        b_close = Button(text="Close", size_hint=(1, None), height=dp(44))
        content.add_widget(b_load)
        content.add_widget(b_select)
        content.add_widget(b_headers)
        content.add_widget(b_close)
        popup = Popup(title="CSV", content=content, size_hint=(0.6, 0.5))
        b_load.bind(on_release=lambda *_: (popup.dismiss(), self._open_csv_filechooser()))
        b_select.bind(on_release=lambda *_: (popup.dismiss(), self._open_student_selector_popup()))
        b_headers.bind(on_release=lambda *_: (popup.dismiss(), self._open_headers_popup()))
        b_close.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    # ---------- CSV selector popup implementation ----------
    def _open_student_selector_popup(self):
        """
        Opens a popup that allows:
        - reorder of selected_headers (up/down)
        - filter per selected column
        - single-column sort (toggle asc/desc/none)
        - scrollable list of filtered rows
        - selecting a row to set as the current Data Update payload
        """
        if not self.csv_rows or not self.csv_headers:
            # No CSV loaded
            popup = Popup(title="No CSV",
                          content=Label(text="No CSV loaded. Load a CSV to select students."),
                          size_hint=(0.6, 0.3))
            popup.open()
            return

        # Ensure selected_headers has default if empty
        if not self.selected_headers:
            # default choose first 3 or all
            self.selected_headers = self.csv_headers[:3] if len(self.csv_headers) >= 3 else list(self.csv_headers)

        # local widgets and state
        popup_root = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(6))

        # Column controls (reorder, filter, sort) - vertical list
        cols_area = ScrollView(size_hint=(1, None), height=dp(160))
        cols_inner = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        cols_inner.bind(minimum_height=cols_inner.setter("height"))
        cols_area.add_widget(cols_inner)

        # Helper to refresh the student list
        list_area = ScrollView(size_hint=(1, 1))
        list_grid = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(2))
        list_grid.bind(minimum_height=list_grid.setter("height"))
        list_area.add_widget(list_grid)

        # Create per-column controls
        col_controls = {}
        for h in list(self.selected_headers):
            row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))
            lbl = Label(text=h, size_hint=(0.28, 1), halign="left", valign="middle", font_size=sp(12))
            lbl.bind(size=lbl.setter("text_size"))
            up = Button(text="▲", size_hint=(None, 1), width=dp(34))
            down = Button(text="▼", size_hint=(None, 1), width=dp(34))
            filt = TextInput(text=self._column_filters.get(h, ""), multiline=False, size_hint=(0.4, 1), font_size=sp(12))
            sort_b = Button(text="⋯", size_hint=(None, 1), width=dp(40))
            row.add_widget(lbl)
            row.add_widget(up)
            row.add_widget(down)
            row.add_widget(filt)
            row.add_widget(sort_b)
            cols_inner.add_widget(row)
            col_controls[h] = {"label": lbl, "up": up, "down": down, "filter": filt, "sort": sort_b}

        displayed_row_buttons = []

        def reorder_column_up(header):
            idx = self.selected_headers.index(header)
            if idx > 0:
                self.selected_headers[idx], self.selected_headers[idx - 1] = self.selected_headers[idx - 1], self.selected_headers[idx]
                rebuild_column_controls()
                refresh_student_list()

        def reorder_column_down(header):
            idx = self.selected_headers.index(header)
            if idx < len(self.selected_headers) - 1:
                self.selected_headers[idx], self.selected_headers[idx + 1] = self.selected_headers[idx + 1], self.selected_headers[idx]
                rebuild_column_controls()
                refresh_student_list()

        def cycle_sort(header, btn):
            cur = self._column_sorts.get(header)
            if cur is None:
                new = "asc"
            elif cur == "asc":
                new = "desc"
            else:
                new = None
            # clear all
            for k in list(self._column_sorts.keys()):
                self._column_sorts[k] = None
            if new:
                self._column_sorts[header] = new
            # update buttons
            for h2, ctrls in col_controls.items():
                txt = "⋯"
                if self._column_sorts.get(h2) == "asc":
                    txt = "↑"
                elif self._column_sorts.get(h2) == "desc":
                    txt = "↓"
                ctrls["sort"].text = txt
            refresh_student_list()

        def rebuild_column_controls():
            cols_inner.clear_widgets()
            old_filters = dict(self._column_filters)
            old_sorts = dict(self._column_sorts)
            new_col_controls = {}
            for h in self.selected_headers:
                row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))
                lbl = Label(text=h, size_hint=(0.28, 1), halign="left", valign="middle", font_size=sp(12))
                lbl.bind(size=lbl.setter("text_size"))
                up = Button(text="▲", size_hint=(None, 1), width=dp(34))
                down = Button(text="▼", size_hint=(None, 1), width=dp(34))
                filt = TextInput(text=old_filters.get(h, ""), multiline=False, size_hint=(0.4, 1), font_size=sp(12))
                sort_b = Button(text="⋯", size_hint=(None, 1), width=dp(40))
                st = old_sorts.get(h)
                if st == "asc":
                    sort_b.text = "↑"
                elif st == "desc":
                    sort_b.text = "↓"
                row.add_widget(lbl)
                row.add_widget(up)
                row.add_widget(down)
                row.add_widget(filt)
                row.add_widget(sort_b)
                cols_inner.add_widget(row)
                new_col_controls[h] = {"label": lbl, "up": up, "down": down, "filter": filt, "sort": sort_b}
            col_controls.clear()
            col_controls.update(new_col_controls)
            wire_column_control_callbacks()

        def wire_column_control_callbacks():
            for h, ctrls in col_controls.items():
                ctrls["up"].unbind(on_release=None)
                ctrls["down"].unbind(on_release=None)
                ctrls["sort"].unbind(on_release=None)
                ctrls["filter"].unbind(text=None)
                ctrls["up"].bind(on_release=lambda inst, header=h: reorder_column_up(header))
                ctrls["down"].bind(on_release=lambda inst, header=h: reorder_column_down(header))
                ctrls["sort"].bind(on_release=lambda inst, header=h, btn=ctrls["sort"]: cycle_sort(header, btn))
                ctrls["filter"].bind(text=lambda inst, txt, header=h: _set_filter(header, txt))

        def _set_filter(header, txt):
            self._column_filters[header] = txt.strip()
            refresh_student_list()

        wire_column_control_callbacks()

        def apply_filters_and_sort():
            rows = list(self.csv_rows)
            for h in list(self.selected_headers):
                f = (self._column_filters.get(h) or "").strip()
                if f:
                    f_lower = f.lower()
                    rows = [r for r in rows if f_lower in (str(r.get(h, "")).lower())]
            sort_col = None
            sort_dir = None
            for h in self.selected_headers:
                s = self._column_sorts.get(h)
                if s in ("asc", "desc"):
                    sort_col = h
                    sort_dir = s
                    break
            if sort_col:
                try:
                    rows = sorted(rows, key=lambda r: (r.get(sort_col) or ""), reverse=(sort_dir == "desc"))
                except Exception:
                    pass
            return rows

        def refresh_student_list():
            list_grid.clear_widgets()
            displayed_row_buttons.clear()
            rows = apply_filters_and_sort()
            for ridx, r in enumerate(rows):
                pieces = []
                for h in self.selected_headers:
                    pieces.append(str(r.get(h, "")))
                display = " | ".join(pieces)
                btn = Button(text=display, size_hint_y=None, height=dp(36), halign="left")
                btn.bind(on_release=lambda inst, row=r, idx=ridx: _on_row_selected(row, inst))
                list_grid.add_widget(btn)
                displayed_row_buttons.append(btn)

        def _on_row_selected(row, widget):
            # selecting a CSV row should set CSV selection and display it in Data Update label
            self._selected_csv_row = row
            parts = []
            for h in self.selected_headers:
                parts.append(str(row.get(h, "")).strip())
            payload = "_".join([p for p in parts if p])
            self._selected_author_payload = payload
            # clear any QR selection override state (QR should still override later, but CSV selection is active now)
            self._latest_qr_text = ""
            # show CSV payload
            Clock.schedule_once(lambda *_: setattr(self.qr_last_label, "text", f"Data Update: {payload}"[:200]), 0)
            # highlight selected button
            for b in displayed_row_buttons:
                try:
                    b.background_color = (1, 1, 1, 1)
                except Exception:
                    pass
            try:
                widget.background_color = (0.9, 0.9, 0.5, 1)
            except Exception:
                pass

        # Footer buttons for popup
        popup_btns = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        btn_select = Button(text="Confirm selection")
        btn_force = Button(text="Force update")
        btn_close = Button(text="Close")
        popup_btns.add_widget(btn_select)
        popup_btns.add_widget(btn_force)
        popup_btns.add_widget(btn_close)

        def do_confirm(*_):
            if self._selected_author_payload:
                self._log_internal(f"CSV selection confirmed: {self._selected_author_payload}")
            popup.dismiss()

        def do_force(*_):
            if not self._selected_author_payload:
                self._log_internal("No CSV row selected to force update")
                notice = Popup(title="No selection", content=Label(text="Select a student row first."), size_hint=(0.6, 0.3))
                notice.open()
                return
            self._maybe_commit_author(self._selected_author_payload, source="csv")
            popup.dismiss()

        btn_select.bind(on_release=lambda *_: do_confirm())
        btn_force.bind(on_release=lambda *_: do_force())
        btn_close.bind(on_release=lambda *_: popup.dismiss())

        popup_root.add_widget(Label(text="Columns (reorder / filter / sort):", size_hint=(1, None), height=dp(20)))
        popup_root.add_widget(cols_area)
        popup_root.add_widget(Label(text="Rows:", size_hint=(1, None), height=dp(18)))
        popup_root.add_widget(list_area)
        popup_root.add_widget(popup_btns)

        popup = Popup(title="Select student from CSV", content=popup_root, size_hint=(0.95, 0.85))
        refresh_student_list()
        popup.open()

    def _force_update_from_selection(self):
        """
        Force pushing the currently selected CSV-derived payload to camera author field.
        If no CSV selection exists show user a notice.
        """
        if not self._selected_author_payload:
            self._log_internal("Force update requested but no CSV selection available")
            notice = Popup(title="No selection", content=Label(text="No student selected. Use 'CSV' first."), size_hint=(0.6, 0.3))
            notice.open()
            return
        self._log_internal(f"Forcing update with CSV payload: {self._selected_author_payload}")
        self._maybe_commit_author(self._selected_author_payload, source="manual-csv")

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

    # ---------- styling / menu / logging ----------
    def _style_connect_button(self, initial=False):
        self.connect_btn.background_normal = ""
        self.connect_btn.background_down = ""
        self.connect_btn.background_color = (0.06, 0.45, 0.75, 1.0)
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

    # ---------- reorganized menu: Build dropdown (top-level) and sub-popups ----------
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

        def add_button_to_dd(text, fn):
            b = Button(text=text, size_hint_y=None, height=dp(40), font_size=sp(13))
            self._style_menu_button(b)
            b.bind(on_release=lambda *_: (fn(), dd.dismiss()))
            dd.add_widget(b)

        # Framing
        add_header("Framing")
        add_button_to_dd("Reset framing", lambda: self._fit_preview_to_holder())

        # Submenus implemented as popups for clarity and placement control
        add_header("Groups")
        add_button_to_dd("Overlays...", lambda: self._open_overlays_popup())
        add_button_to_dd("Capture...", lambda: self._open_capture_popup())
        add_button_to_dd("Settings...", lambda: self._open_settings_popup())

        add_header("CSV")
        # CSV actions moved to CSV footer button; keep a shortcut here to open the CSV menu
        add_button_to_dd("CSV tools...", lambda: self._open_csv_menu())

        add_header("QR & Author")
        add_button_to_dd("Push payload (Author)", lambda: self._maybe_commit_author(self.manual_payload, source="manual"))

        add_header("Debug")
        add_button_to_dd("Dump /ccapi", lambda: self.dump_ccapi())

        return dd

    # Overlays popup
    def _open_overlays_popup(self):
        content = BoxLayout(orientation="vertical", padding=dp(6), spacing=dp(6))
        content.add_widget(Label(text="Overlays", size_hint=(1, None), height=dp(28)))
        def mk_toggle(text, prop, initial):
            row = BoxLayout(size_hint=(1, None), height=dp(34))
            lbl = Label(text=text, size_hint=(0.7, 1))
            cb = CheckBox(active=initial, size_hint=(0.3, 1))
            cb.bind(active=lambda inst, val: setattr(self.preview, prop, val))
            row.add_widget(lbl)
            row.add_widget(cb)
            content.add_widget(row)
        mk_toggle("Border (blue)", "show_border", self.preview.show_border)
        mk_toggle("Grid (orange)", "show_grid", self.preview.show_grid)
        mk_toggle("Crop 5:7 (red)", "show_57", self.preview.show_57)
        mk_toggle("Crop 8:10 (yellow)", "show_810", self.preview.show_810)
        mk_toggle("Oval (purple)", "show_oval", self.preview.show_oval)
        mk_toggle("QR overlay", "show_qr", self.preview.show_qr)
        btn_close = Button(text="Close", size_hint=(1, None), height=dp(40))
        content.add_widget(btn_close)
        popup = Popup(title="Overlays", content=content, size_hint=(0.6, 0.6))
        btn_close.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    # Capture popup
    def _open_capture_popup(self):
        content = BoxLayout(orientation="vertical", padding=dp(6), spacing=dp(6))
        content.add_widget(Label(text="Capture", size_hint=(1, None), height=dp(28)))
        row = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(4))
        row.add_widget(Label(text="Capture:", size_hint=(None, 1), width=dp(70)))
        def mk_btn(label, ctype):
            b = Button(text=label)
            b.bind(on_release=lambda *_: (setattr(self, "capture_type", ctype), self._log_internal(f"Capture type set to {ctype}")))
            return b
        row.add_widget(mk_btn("JPG", CaptureType.JPG))
        row.add_widget(mk_btn("RAW", CaptureType.RAW))
        row.add_widget(mk_btn("Both", CaptureType.BOTH))
        content.add_widget(row)
        btn_close = Button(text="Close", size_hint=(1, None), height=dp(40))
        content.add_widget(btn_close)
        popup = Popup(title="Capture", content=content, size_hint=(0.6, 0.5))
        btn_close.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    # Settings popup
    def _open_settings_popup(self):
        content = BoxLayout(orientation="vertical", padding=dp(6), spacing=dp(6))
        content.add_widget(Label(text="Settings", size_hint=(1, None), height=dp(28)))
        btn_ip = Button(text="IP settings…", size_hint=(1, None), height=dp(44))
        btn_fps = Button(text="Display FPS…", size_hint=(1, None), height=dp(44))
        cb_row = BoxLayout(size_hint=(1, None), height=dp(36))
        lbl = Label(text="Show log", size_hint=(0.7, 1))
        cb = CheckBox(active=self.show_log, size_hint=(0.3, 1))
        cb.bind(active=lambda inst, val: self._set_log_visible(val))
        cb_row.add_widget(lbl)
        cb_row.add_widget(cb)
        content.add_widget(btn_ip)
        content.add_widget(btn_fps)
        content.add_widget(cb_row)
        btn_close = Button(text="Close", size_hint=(1, None), height=dp(40))
        content.add_widget(btn_close)
        popup = Popup(title="Settings", content=content, size_hint=(0.6, 0.5))
        btn_ip.bind(on_release=lambda *_: (popup.dismiss(), self._open_ip_popup()))
        btn_fps.bind(on_release=lambda *_: (popup.dismiss(), self._open_fps_popup()))
        btn_close.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    # ---------- FPS / IP popups (unchanged) ----------
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

    # ---------- Android CSV (SAF) helpers (unchanged) ----------
    def _bind_android_activity_once(self):
        if getattr(self, "_android_activity_bound", False):
            return
        if platform != "android":
            return
        try:
            from android import activity
            activity.bind(on_activity_result=self._on_android_activity_result)
            self._android_activity_bound = True
        except Exception as e:
            self._log_internal(f"Android activity bind failed: {e}")

    def _open_csv_saf(self):
        self._bind_android_activity_once()
        try:
            from android import mActivity
            from jnius import autoclass

            Intent = autoclass("android.content.Intent")
            intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            intent.setType("*/*")
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            self._log_internal("Opening Android file picker…")
            mActivity.startActivityForResult(intent, self._csv_req_code)
        except Exception as e:
            self._log_internal(f"Failed to open Android picker: {e}")

    def _on_android_activity_result(self, request_code, result_code, intent):
        if request_code != getattr(self, "_csv_req_code", 4242):
            return
        if result_code != -1 or intent is None:
            self._log_internal("CSV picker canceled")
            return
        try:
            from android import mActivity
            from jnius import cast, autoclass

            Intent = autoclass("android.content.Intent")
            uri = cast("android.net.Uri", intent.getData())
            if uri is None:
                self._log_internal("CSV picker returned no URI")
                return

            try:
                flags = intent.getFlags()
                take_flags = flags & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
                mActivity.getContentResolver().takePersistableUriPermission(uri, take_flags)
            except Exception:
                pass

            data = self._read_android_uri_bytes(uri)
            self._parse_csv_bytes(data)
            self._log_internal(f"CSV loaded from picker: {len(self.csv_rows)} rows")
        except Exception as e:
            self._log_internal(f"CSV load failed (Android): {e}")

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
        if platform != 'android':
            self._log_internal("CSV load is Android-only (SAF). Please run on-device to load CSV.")
            return
        return self._open_csv_saf()

    def _parse_csv_bytes(self, b: bytes):
        self._log_internal(f"CSV size: {len(b)} bytes")
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
        self._log_internal(f"CSV headers: {headers}")
        self._log_internal(f"CSV rows: {len(rows)}")
        preferred = ["LAST_NAME", "FIRST_NAME", "GRADE", "TEACHER", "STUDENT_ID"]
        self.selected_headers = [h for h in preferred if h in headers]
        if not self.selected_headers and headers:
            self.selected_headers = headers[:3]

    def _open_headers_popup(self):
        if not self.csv_headers:
            self._log_internal("No CSV loaded; cannot pick headers")
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
            self._log_internal(f"Selected headers: {self.selected_headers}")
            popup.dismiss()
        btn_ok.bind(on_release=do_ok)
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()
        self._headers_popup = popup

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
        Clock.schedule_once(lambda *_: setattr(self.qr_status, "text", f"Data Update: updating… ({source})"), 0)
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
                self.qr_status.text = "Data Update: updated ✓"
            else:
                self._log_internal(f"Author verify failed ({source}). wrote='{value}' read='{got}' err='{err}'")
                self.qr_status.text = "Data Update: verify failed ✗"
        Clock.schedule_once(_finish, 0)

    # ---------- liveview / decoder / QR ----------
    def _set_qr_enabled(self, enabled: bool):
        self.qr_enabled = bool(enabled)
        if not self.qr_enabled and not self._qr_temp_active:
            self._set_qr_ui(None, None, note="Data Update: none")
        elif self.qr_enabled:
            self._set_qr_ui(None, None, note="Data Update: on")

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
        try:
            self._update_metrics(self._last_decoded_ts)
        except Exception:
            pass

    def _update_metrics(self, frame_ts):
        now = time.time()
        if now - self._stat_t0 >= 1.0:
            if getattr(self, "show_log", False):
                self._refresh_log_view()
            self._fetch_count = 0
            self._decode_count = 0
            self._display_count = 0
            self._stat_t0 = now

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
        self._set_qr_ui(None, None, note="Data Update: none")
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

            # defensive: ensure preview exists
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
                        # if overlay active, don't replace overlay
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

        # When a QR is read, it should override any CSV selection currently displayed.
        # Clear CSV selection state so the QR becomes the active Data Update source.
        if text:
            self._selected_csv_row = None
            self._selected_author_payload = None

        if text:
            if (text not in self._qr_seen) and (now - self._qr_last_add_time >= self.qr_new_gate_s):
                self._qr_seen.add(text)
                self._qr_last_add_time = now
                self._log_internal(f"QR: {text}")
            # auto-commit from QR if desired
            self._maybe_commit_author(text, source="qr")

        if text:
            self._latest_qr_text = text
            Clock.schedule_once(lambda *_: setattr(self.qr_last_label, "text", f"Data Update: {text}"[:200]), 0)
        elif self._latest_qr_text:
            Clock.schedule_once(lambda *_: setattr(self.qr_last_label, "text", f"Data Update: {self._latest_qr_text}"[:200]), 0)

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

        note = f"Data Update: {text[:80]}" if text else ("QR: detected" if points else ("Data Update: on" if self.qr_enabled else "Data Update: none"))
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

    def _set_qr_ui(self, text, points, note="Data Update: none"):
        if text:
            # QR becomes the visible Data Update and clears any CSV selection
            self._selected_csv_row = None
            self._selected_author_payload = None
            self._latest_qr_text = text
            self.qr_last_label.text = f"Data Update: {text}"
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
        # only respond to taps inside the PreviewOverlay region
        if not instance.collide_point(*touch.pos):
            return False
        # Start a temporary QR pulse scan for 1 second (or until detection)
        self._start_pulse_qr()
        return True

    def _start_pulse_qr(self):
        if getattr(self, "_qr_pulse_event", None):
            try:
                self._qr_pulse_event.cancel()
            except Exception:
                pass
            self._qr_pulse_event = None

        self._qr_temp_active = True
        self._set_qr_ui(None, None, note="Data Update: scanning…")
        try:
            self._qr_pulse_event = Clock.schedule_once(lambda *_: self._end_pulse_qr(), 1.0)
        except Exception:
            def _bg_end():
                time.sleep(1.0)
                Clock.schedule_once(lambda *_: self._end_pulse_qr(), 0)
            threading.Thread(target=_bg_end, daemon=True).start()

    def _end_pulse_qr(self, *args):
        if getattr(self, "_qr_pulse_event", None):
            try:
                self._qr_pulse_event.cancel()
            except Exception:
                pass
            self._qr_pulse_event = None
        self._qr_temp_active = False
        if not getattr(self, "qr_enabled", False):
            self._set_qr_ui(None, None, note="Data Update: none")
        else:
            self._set_qr_ui(None, None, note="Data Update: on")

    # ---------- thumbnails, overlay, and full-res swap ----------
    def _download_thumb_for_path(self, ccapi_path: str):
        """
        Download thumbnail bytes (bg), save to disk, decode/rotate in bg,
        then create Kivy Texture on main thread and update the strip.
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

        # Save thumbnail to disk (so we can reconstruct oriented preview textures later)
        out_path = None
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
            out_path = None

        # Decode using PIL, rotate to preview orientation, create rgb bytes
        try:
            pil = PILImage.open(BytesIO(thumb_bytes)).convert("RGB")
            rot = getattr(self.preview, "preview_rotation", 0) if hasattr(self, "preview") else 0
            if rot:
                pil = pil_rotate_90s(pil, rot)
            pil.thumbnail((200, 200))
            w, h = pil.size
            rgb_bytes = pil.tobytes()
        except Exception as e:
            self._log_internal(f"Thumbnail decode err (bg): {e}")
            return

        def _make_texture_and_update(_dt, rgb_bytes=rgb_bytes, w=w, h=h, ccapi_path=ccapi_path, out_path=out_path):
            try:
                tex = self._create_texture_from_rgb(rgb_bytes, w, h, flip_vertical=True)
                if tex is None:
                    return
            except Exception as e:
                self._log_internal(f"Texture create/blit err: {e}")
                return

            # Insert at front (most recent first)
            self._thumb_textures.insert(0, tex)
            self._thumb_paths.insert(0, ccapi_path)
            self._thumb_saved_paths.insert(0, out_path if out_path else "")
            self._thumb_textures = self._thumb_textures[:5]
            self._thumb_paths = self._thumb_paths[:5]
            self._thumb_saved_paths = self._thumb_saved_paths[:5]

            for idx, img in enumerate(self._thumb_images):
                if idx < len(self._thumb_textures):
                    img.texture = self._thumb_textures[idx]
                else:
                    img.texture = None

        Clock.schedule_once(_make_texture_and_update, 0)

    def _background_download_latest(self):
        # removed direct menu action; kept for internal use
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

    def _on_thumb_touch(self, image_widget, touch):
        if not image_widget.collide_point(*touch.pos):
            return False
        idx = getattr(image_widget, "thumb_index", None)
        if idx is None:
            return False
        if idx >= len(self._thumb_paths):
            return False

        ccapi_path = self._thumb_paths[idx]
        saved_path = self._thumb_saved_paths[idx] if idx < len(self._thumb_saved_paths) else None
        rot = getattr(self.preview, "preview_rotation", 0) if hasattr(self, "preview") else 0

        # If user taps the thumbnail that is already selected (overlay active and same index), close overlay
        if self._overlay_active and (self._overlay_thumb_index == idx):
            # close overlay
            self._log_internal(f"Thumbnail {idx} tapped while overlay active: closing overlay")
            # remove overlay and highlight
            self._clear_thumb_highlight()
            self.preview.clear_overlay_texture()
            self._overlay_active = False
            self._overlay_thumb_index = None
            # cancel any pending full-res fetch for this thumb (so it won't try to apply later)
            try:
                if idx in self._pending_full_fetches:
                    del self._pending_full_fetches[idx]
            except Exception:
                pass
            # restore live view texture if present
            if self._frame_texture is not None:
                try:
                    self.preview.set_texture(self._frame_texture)
                except Exception:
                    pass
            return True

        # Otherwise open overlay for this thumbnail
        # Highlight the tapped thumb
        self._highlight_thumb(idx)

        # Prefer to create preview texture from saved file so orientation is consistent
        if saved_path and os.path.exists(saved_path):
            # create texture on main thread
            tex = self._create_texture_from_jpeg_file(saved_path, rotate=rot, flip_vertical=True)
            if tex:
                Clock.schedule_once(lambda *_: self._show_overlay_with_texture(tex, idx), 0)
                threading.Thread(target=self._fetch_full_and_replace, args=(ccapi_path, idx), daemon=True).start()
                return True

        # fallback to in-memory texture
        if idx < len(self._thumb_textures):
            thumb_tex = self._thumb_textures[idx]
            Clock.schedule_once(lambda *_: self._show_overlay_with_texture(thumb_tex, idx), 0)
            threading.Thread(target=self._fetch_full_and_replace, args=(ccapi_path, idx), daemon=True).start()
            return True

        # otherwise download thumb then overlay
        threading.Thread(target=self._download_thumb_and_overlay, args=(ccapi_path, idx), daemon=True).start()
        return True

    def _download_thumb_and_overlay(self, ccapi_path, idx):
        self._download_thumb_for_path(ccapi_path)
        if idx < len(self._thumb_textures):
            tex = self._thumb_textures[idx]
            Clock.schedule_once(lambda *_: self._show_overlay_with_texture(tex, idx), 0)
            threading.Thread(target=self._fetch_full_and_replace, args=(ccapi_path, idx), daemon=True).start()

    def _show_overlay_with_texture(self, texture: Texture, thumb_index: int):
        """
        Put the low-res texture into the overlay Rectangle that exactly matches the preview widget's drawn rect.
        """
        if texture is None:
            return
        # Set overlay_active and remember index
        self._overlay_active = True
        self._overlay_thumb_index = thumb_index
        # Set the overlay rectangle texture (this will be drawn to exactly the preview.drawn_rect)
        try:
            self.preview.set_overlay_texture(texture)
        except Exception as e:
            self._log_internal(f"Failed to set overlay texture: {e}")

    def _fetch_full_and_replace(self, ccapi_path: str, thumb_index: int):
        """
        Download full-res JPG in the background, decode to RGB bytes, then schedule
        creation of the Kivy Texture on the main thread. If the user selects a different
        thumbnail before this finishes, the result is discarded.
        """
        # create a request id so we can detect staleness
        request_id = time.time()
        self._pending_full_fetches[thumb_index] = request_id

        full_url = f"https://{self.camera_ip}{ccapi_path}"
        self._log_internal(f"Fetching full-res: {full_url} (req={request_id})")
        try:
            resp = self._session.get(full_url, timeout=20.0, stream=True)
            if resp.status_code != 200 or not resp.content:
                self._log_internal(f"Full image download failed: {resp.status_code} (req={request_id})")
                # clean up tracking
                try:
                    if self._pending_full_fetches.get(thumb_index) == request_id:
                        del self._pending_full_fetches[thumb_index]
                except Exception:
                    pass
                return
            data = resp.content
        except Exception as e:
            self._log_internal(f"Full image download err: {e} (req={request_id})")
            try:
                if self._pending_full_fetches.get(thumb_index) == request_id:
                    del self._pending_full_fetches[thumb_index]
            except Exception:
                pass
            return

        # Decode + rotate in background
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self._log_internal(f"cv2.imdecode returned None for full image (req={request_id})")
                try:
                    if self._pending_full_fetches.get(thumb_index) == request_id:
                        del self._pending_full_fetches[thumb_index]
                except Exception:
                    pass
                return

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
        except Exception as e:
            self._log_internal(f"Full image decode err (bg): {e} (req={request_id})")
            try:
                if self._pending_full_fetches.get(thumb_index) == request_id:
                    del self._pending_full_fetches[thumb_index]
            except Exception:
                pass
            return

        # Create the Kivy texture on the main thread and apply it, but only if still current
        def _apply_full_on_main(_dt, rgb_bytes=rgb_bytes, w=w, h=h, thumb_index=thumb_index, request_id=request_id):
            # If another selection happened, the pending id won't match
            cur_req = self._pending_full_fetches.get(thumb_index)
            if cur_req != request_id:
                self._log_internal(f"Full-res fetch for thumb {thumb_index} aborted (stale req={request_id}, cur={cur_req})")
                try:
                    if thumb_index in self._pending_full_fetches and self._pending_full_fetches[thumb_index] == request_id:
                        del self._pending_full_fetches[thumb_index]
                except Exception:
                    pass
                return

            # Also ensure overlay is still active and targeting this thumb
            if (not getattr(self, "_overlay_active", False)) or (self._overlay_thumb_index != thumb_index):
                self._log_internal(f"Full-res fetch for thumb {thumb_index} aborted: overlay not active or changed (req={request_id})")
                try:
                    if thumb_index in self._pending_full_fetches:
                        del self._pending_full_fetches[thumb_index]
                except Exception:
                    pass
                return

            tex = self._create_texture_from_rgb(rgb_bytes, w, h, flip_vertical=True)
            if not tex:
                self._log_internal(f"Failed to create full-res texture on main thread (req={request_id})")
                try:
                    if thumb_index in self._pending_full_fetches:
                        del self._pending_full_fetches[thumb_index]
                except Exception:
                    pass
                return

            try:
                self.preview.set_overlay_texture(tex)
            except Exception:
                pass

            # Update thumbnail strip visual feedback if still relevant
            try:
                if thumb_index < len(self._thumb_textures):
                    self._thumb_textures[thumb_index] = tex
                    if thumb_index < len(self._thumb_images):
                        self._thumb_images[thumb_index].texture = tex
            except Exception:
                pass

            self._log_internal(f"Full-res applied to overlay for thumb {thumb_index} (req={request_id})")

            # cleanup pending map
            try:
                if thumb_index in self._pending_full_fetches and self._pending_full_fetches[thumb_index] == request_id:
                    del self._pending_full_fetches[thumb_index]
            except Exception:
                pass

        Clock.schedule_once(_apply_full_on_main, 0)

    # ---------- contents / poller ----------
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

    def start_polling_new_images(self):
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._log_internal(f"Starting image poller every {self.poll_interval_s}s (background thread)")
        self._poll_thread_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)
        self._poll_thread.start()
        # reflect UI state
        self.autofetch_running = True
        self._style_autofetch_button(running=True)

    def stop_polling_new_images(self):
        if self._poll_thread is None:
            return
        self._log_internal("Stopping image poller (background thread)")
        self._poll_thread_stop.set()
        self._poll_thread = None
        self.autofetch_running = False
        self._style_autofetch_button(running=False)

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
