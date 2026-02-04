# Volume Toolkit - main.py
# Full drop-in file implementing:
# - live view fetching + background decoder thread
# - thumbnail strip and full-res overlay
# - a "QR Find" toggle button (in the control row) that scans indefinitely until a QR is found
#   or the user cancels by pressing the button again. Found QR is stored in "QR Payload"
#   and displayed under "Exif Payload".
# - All UI/Texture creation happens on the main (GL) thread via Clock.schedule_once.
# - Defensive checks to avoid attribute races and missing-method errors.
#
# NOTE: adjust camera_ip default and Android SAF functions as needed for your environment.

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
    show_qr = BooleanProperty(False)

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
            # newer Kivy addition; safe to ignore if not present
            self.img.fit_mode = "contain"
        except Exception:
            pass

        self.add_widget(self.img)

        lw = 2
        lw_qr = 6

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

        self._ln_grid_list = []
        self._overlay_texture = None
        self._overlay_rect = None
        self._overlay_rect_color = None
        self._qr_points_px = None

        self.bind(pos=self._update_overlay_rect, size=self._update_overlay_rect)
        self.img.bind(pos=self._update_overlay_rect, size=self._update_overlay_rect)

        self.bind(
            pos=self._redraw, size=self._redraw,
            show_border=self._redraw, show_grid=self._redraw, show_57=self._redraw,
            show_810=self._redraw, show_oval=self._redraw, show_qr=self._redraw,
            grid_n=self._redraw,
            oval_cx=self._redraw, oval_cy=self._redraw, oval_w=self._redraw, oval_h=self._redraw
        )

        self._redraw()

    def set_overlay_texture(self, texture: Texture):
        self.clear_overlay_texture()
        if texture is None:
            return
        dx, dy, iw, ih = self._drawn_rect()
        self._overlay_texture = texture
        try:
            with self.img.canvas.after:
                self._overlay_rect_color = Color(1.0, 1.0, 1.0, 1.0)
                self._overlay_rect = Rectangle(texture=self._overlay_texture, pos=(dx, dy), size=(iw, ih))
        except Exception:
            try:
                with self.canvas.before:
                    self._overlay_rect_color = Color(1.0, 1.0, 1.0, 1.0)
                    self._overlay_rect = Rectangle(texture=self._overlay_texture, pos=(dx, dy), size=(iw, ih))
            except Exception:
                self._overlay_rect = None
                self._overlay_rect_color = None
                self._overlay_texture = None
                return

    def clear_overlay_texture(self):
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
        if self._overlay_rect is None:
            return
        try:
            dx, dy, iw, ih = self._drawn_rect()
            self._overlay_rect.pos = (dx, dy)
            self._overlay_rect.size = (iw, ih)
        except Exception:
            pass

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

    QR_FIND_POLL_S = 0.12  # poll interval while scanning

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # default config
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

        self._log_lines = []
        self._max_log_lines = 300
        self.show_log = False

        self._frame_texture = None
        self._frame_size = None

        self.dropdown = None

        # QR finder state (from-scratch)
        self._qr_detector = cv2.QRCodeDetector()
        self._qr_finder_thread = None
        self._qr_finder_stop = threading.Event()
        self._qr_found_text = ""
        self._qr_payload = ""
        self._csv_payload = ""

        self._latest_decoded_bgr = None
        self._latest_decoded_bgr_ts = 0.0

        self._decode_queue = queue.Queue(maxsize=2)
        self._decoder_thread = None
        self._decoder_stop = threading.Event()

        self._overlay_active = False
        self._overlay_thumb_index = None

        self._highlighted_thumb_index = None
        self._thumb_highlight_lines = {}

        self.author_max_chars = 60
        self._last_committed_author = None
        self._author_update_in_flight = False
        self.manual_payload = ""

        self.csv_headers = []
        self.csv_rows = []
        self.selected_headers = []
        self._headers_popup = None

        self._payload_source = "none"

        self._thumb_textures = []
        self._thumb_images = []
        self._thumb_paths = []
        self._thumb_saved_paths = []

        self._pending_full_fetches = {}

        self.download_dir = "downloads"
        self.thumb_dir = "thumbs"
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.thumb_dir, exist_ok=True)

        self._last_seen_image = None
        self._poll_thread = None
        self._poll_thread_stop = threading.Event()
        self.poll_interval_s = 2.0

        self.save_full_size = False

        self._session = requests.Session()
        self._session.verify = False

        self._android_activity_bound = False
        self._csv_req_code = 4242

        self.header = None
        self.preview_holder = None

        self._column_filters = {}
        self._column_sorts = {}
        self._selected_csv_row = None
        self._selected_author_payload = None

        self.autofetch_running = False

    # ---------- logging ----------
    def _log_internal(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]
        if getattr(self, "show_log", False):
            self._refresh_log_view()
        try:
            print(line, flush=True)
        except Exception:
            pass

    def _refresh_log_view(self):
        metrics_line = self._get_metrics_text()
        try:
            self.log_label.text = metrics_line + "\n\n" + "\n".join(self._log_lines)
        except Exception:
            pass

    def _get_metrics_text(self):
        return f"Delay: -- ms | Fetch: {self._fetch_count} | Decode: {self._decode_count} | Display: {self._display_count}"

    # ---------- Exif payload ----------
    def _set_exif_payload(self, payload: str, source: str):
        try:
            payload = (payload or "").strip()
            if not payload:
                source = "none"

            if source == "QR":
                self._qr_found_text = payload
                self._selected_csv_row = None
                self._selected_author_payload = None
            elif source == "CSV":
                self._selected_author_payload = payload
            elif source == "MANUAL":
                self.manual_payload = payload
            else:
                self._qr_found_text = ""
                self._selected_author_payload = None
                self.manual_payload = ""

            self._payload_source = source

            def _update_on_main(_dt):
                try:
                    if payload:
                        label_text = f"Exif Payload ({self._payload_source}): {payload}"
                        status_text = f"Exif Payload ({self._payload_source}): {payload[:80]}"
                    else:
                        label_text = "Exif Payload (none): none"
                        status_text = ("Exif Payload: on" if getattr(self, "live_running", False) else "Exif Payload: none")
                    self.csv_payload_label.text = f"CSV Payload: {payload}"
                    self.qr_status.text = status_text
                except Exception:
                    pass

            Clock.schedule_once(_update_on_main, 0)
        except Exception as e:
            try:
                self._log_internal(f"_set_exif_payload error: {e}")
            except Exception:
                pass

    # ---------- QR Find (toggle until found or canceled) ----------
    def _on_qr_find_pressed(self):
        if self._qr_finder_thread is not None and self._qr_finder_thread.is_alive():
            # cancel
            self._log_internal("QR Finder: user requested cancel")
            self._qr_finder_stop.set()
            try:
                self.qr_find_btn.text = "QR Find"
            except Exception:
                pass
            return

        self._log_internal("QR Finder: starting (will run until found or cancelled)")
        self._set_qr_payload("", source="none")
        try:
            self.qr_find_btn.text = "Stop Find"
            self.qr_find_btn.disabled = False
        except Exception:
            pass

        self._qr_finder_stop.clear()
        self._qr_finder_thread = threading.Thread(target=self._qr_finder_worker, daemon=True)
        self._qr_finder_thread.start()

    def _set_qr_payload(self, payload: str, source: str = "QR"):
        self._qr_payload = (payload or "").strip()
        def _update(_dt):
            try:
                if self._qr_payload:
                    self.qr_payload_label.text = f"QR Payload: {self._qr_payload}"
                else:
                    self.qr_payload_label.text = "QR Payload: none"
                if source == "QR" and self._qr_payload:
                    self._payload_source = "QR"
            except Exception:
                pass
        Clock.schedule_once(_update, 0)

    def _qr_finder_worker(self):
        found_text = None
        found_points = None

        while not self._qr_finder_stop.is_set():
            with self._lock:
                bgr = None
                if self._latest_decoded_bgr is not None:
                    bgr = self._latest_decoded_bgr.copy()

            if bgr is None:
                if self._qr_finder_stop.wait(self.QR_FIND_POLL_S):
                    break
                continue

            try:
                texts = []
                pts = None

                if hasattr(self._qr_detector, "detectAndDecodeMulti"):
                    try:
                        res = self._qr_detector.detectAndDecodeMulti(bgr)
                        if res:
                            decoded_texts, pts_arr, _ = res
                            if decoded_texts:
                                texts = [t for t in decoded_texts if isinstance(t, str) and t.strip()]
                            if pts_arr is not None and isinstance(pts_arr, np.ndarray) and pts_arr.size:
                                arr0 = pts_arr[0].astype(int).reshape(-1, 2)
                                if len(arr0) >= 4:
                                    pts = [(int(arr0[i][0]), int(arr0[i][1])) for i in range(4)]
                    except Exception:
                        texts = []
                        pts = None

                if not texts:
                    decoded, pts_single, _ = self._qr_detector.detectAndDecode(bgr)
                    if isinstance(decoded, str) and decoded.strip():
                        texts = [decoded.strip()]
                    if pts_single is not None and pts is None:
                        try:
                            arr = np.array(pts_single).astype(int).reshape(-1, 2)
                            if len(arr) >= 4:
                                pts = [(int(arr[i][0]), int(arr[i][1])) for i in range(4)]
                        except Exception:
                            pts = None

                if texts:
                    found_text = texts[0]
                    found_points = pts
                    break

            except Exception as e:
                self._log_internal(f"QR Finder exception: {e}")

            if self._qr_finder_stop.wait(self.QR_FIND_POLL_S):
                break

        def _finish_ui(_dt):
            try:
                if found_text:
                    self._log_internal(f"QR Finder: found -> '{found_text}'")
                    self._set_qr_payload(found_text, source="QR")
                    if found_points:
                        try:
                            self.preview.show_qr = True
                            self.preview.set_qr(found_points)
                            Clock.schedule_once(lambda *_: self.preview.set_qr(None), 3.0)
                        except Exception:
                            pass
                else:
                    if self._qr_finder_stop.is_set():
                        self._log_internal("QR Finder: canceled by user")
                    else:
                        self._log_internal("QR Finder: finished without finding a QR")
                    self._set_qr_payload("", source="none")

                try:
                    self.qr_find_btn.text = "QR Find"
                    self.qr_find_btn.disabled = False
                except Exception:
                    pass
            except Exception:
                pass

        self._qr_finder_stop.clear()
        Clock.schedule_once(_finish_ui, 0)

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

        # Control row - equally sized buttons so they fit on small screens
        row2 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.connect_btn = Button(text="Connect", font_size=sp(16), size_hint=(1, 1))
        self._style_connect_button(initial=True)
        self.start_btn = Button(text="Start", disabled=True, font_size=sp(16), size_hint=(1, 1))
        self._style_start_button(stopped=True)
        self.autofetch_btn = Button(text="Autofetch Off", font_size=sp(14), size_hint=(1, 1))
        self._style_autofetch_button(running=False)
        self.qr_find_btn = Button(text="QR Find", font_size=sp(14), size_hint=(1, 1))
        row2.add_widget(self.connect_btn)
        row2.add_widget(self.start_btn)
        row2.add_widget(self.autofetch_btn)
        row2.add_widget(self.qr_find_btn)
        root.add_widget(row2)

        # Exif / QR payload labels
        self.csv_payload_label = Label(text="CSV Payload: none", size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.qr_last_label)
        self.qr_payload_label = Label(text="QR Payload: none", size_hint=(1, None), height=dp(20), font_size=sp(12))
        root.add_widget(self.qr_payload_label)
        self.status = Label(text="Status: not connected", size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.status)
        self.qr_status = Label(text="", size_hint=(1, None), height=dp(18), font_size=sp(11))
        root.add_widget(self.qr_status)

        # Main area
        main_area = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint=(1, 0.6))
        self.preview_holder = AnchorLayout(anchor_x="center", anchor_y="center", size_hint=(0.80, 1))
        self.preview_scatter = Scatter(do_translation=False, do_scale=False, do_rotation=False, size_hint=(None, None))
        self.preview = PreviewOverlay(size_hint=(None, None))
        self.preview_scatter.add_widget(self.preview)
        self.preview_holder.add_widget(self.preview_scatter)
        main_area.add_widget(self.preview_holder)

        # Thumbnail sidebar
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

        # Footer
        footer = BoxLayout(orientation="horizontal", size_hint=(1, None), height=dp(48), spacing=dp(6))
        footer.add_widget(Label())
        self.csv_btn = Button(text="CSV", size_hint=(None, 1), width=dp(140))
        self.push_update_btn = Button(text="Push Update", size_hint=(None, 1), width=dp(140))
        footer.add_widget(self.csv_btn)
        footer.add_widget(self.push_update_btn)
        root.add_widget(footer)

        # Fit preview helper
        def fit_preview_to_holder(*_):
            w = max(dp(220), self.preview_holder.width * 0.98)
            h = max(dp(220), self.preview_holder.height * 0.98)
            self.preview_scatter.size = (w, h)
            self.preview.size = (w, h)
            self.preview_scatter.pos = (
                self.preview_holder.x + (self.preview_holder.width - w) / 2.0,
                self.preview_holder.y + (self.preview_holder.height - h) / 2.0
            )
            self.preview._update_overlay_rect()
            for idx in range(len(self._thumb_images)):
                self._update_thumb_highlight_pos(idx)

        self._fit_preview_to_holder = fit_preview_to_holder
        self.preview_holder.bind(pos=fit_preview_to_holder, size=fit_preview_to_holder)

        # Log area
        self.log_holder = BoxLayout(orientation="vertical", size_hint=(1, None), height=0)
        log_sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.log_label = Label(text="", size_hint_y=None, halign="left", valign="top", font_size=sp(11))
        self.log_label.bind(width=lambda *_: setattr(self.log_label, "text_size", (self.log_label.width, None)))
        self.log_label.bind(texture_size=lambda *_: setattr(self.log_label, "height", self.log_label.texture_size[1]))
        log_sv.add_widget(self.log_label)
        self.log_holder.add_widget(log_sv)
        root.add_widget(self.log_holder)

        # menu and bindings
        self.dropdown = self._build_dropdown()
        self.menu_btn.bind(on_release=lambda *_: self.dropdown.open(self.menu_btn))

        self.connect_btn.bind(on_press=lambda *_: self.connect_camera())
        self.start_btn.bind(on_press=lambda *_: self._on_start_pressed())
        self.autofetch_btn.bind(on_press=lambda *_: (self._on_autofetch_pressed(), self._style_autofetch_button(self.autofetch_running)))
        self.qr_find_btn.bind(on_press=lambda *_: self._on_qr_find_pressed())
        self.csv_btn.bind(on_release=lambda *_: self._open_csv_filechooser())
        self.push_update_btn.bind(on_release=lambda *_: self._push_update())

        # start decoder thread
        if self._decoder_thread is None:
            self._decoder_thread = threading.Thread(target=self._decoder_loop, daemon=True)
            self._decoder_thread.start()

        # display ticker
        self._reschedule_display_loop(12)

        # window resize
        Window.bind(on_resize=self._on_window_resize)

        self._set_controls_idle()
        self._log_internal("UI ready")
        return root

    # ---------- convenience methods ----------
    def _make_thumb_pos_updater(self, idx):
        def _updater(instance, value):
            try:
                self._update_thumb_highlight_pos(idx)
            except Exception:
                pass
        return _updater

    
    def _push_update(self):
        qr = (self._qr_payload or "").strip()
        csv = (self._csv_payload or "").strip()

        if qr and csv:
            self._open_payload_choice_popup(qr, csv)
        elif qr:
            self._maybe_commit_author(qr, source="QR")
        elif csv:
            self._maybe_commit_author(csv, source="CSV")
        else:
            try:
                self.qr_status.text = "No payload to push"
            except Exception:
                pass

    def _open_payload_choice_popup(self, qr_value, csv_value):
        content = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(8))
        content.add_widget(Label(text="Which value do you want to push?"))

        btn_qr = Button(text=f"QR: {qr_value[:40]}")
        btn_csv = Button(text=f"CSV: {csv_value[:40]}")
        btn_cancel = Button(text="Cancel")

        popup = Popup(
            title="Choose Payload",
            content=content,
            size_hint=(0.85, 0.4)
        )

        btn_qr.bind(on_release=lambda *_: (
            popup.dismiss(),
            self._maybe_commit_author(qr_value, source="QR")
        ))
        btn_csv.bind(on_release=lambda *_: (
            popup.dismiss(),
            self._maybe_commit_author(csv_value, source="CSV")
        ))
        btn_cancel.bind(on_release=lambda *_: popup.dismiss())

        content.add_widget(btn_qr)
        content.add_widget(btn_csv)
        content.add_widget(btn_cancel)
        popup.open()

    # ----------
 thumbnail highlight helpers ----------
    def _highlight_thumb(self, idx):
        self._clear_thumb_highlight()
        if idx is None or idx >= len(self._thumb_images):
            return
        img = self._thumb_images[idx]
        try:
            with img.canvas.after:
                col = Color(1.0, 0.8, 0.0, 1.0)
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

    # ---------- window/rotation ----------
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
        try:
            self.connect_btn.background_normal = ""
            self.connect_btn.background_down = ""
            self.connect_btn.background_color = (0.06, 0.45, 0.75, 1.0)
            if initial or not self.connected:
                self.connect_btn.color = (1, 1, 1, 1)
            else:
                self.connect_btn.color = (1, 1, 0, 1)
        except Exception:
            pass

    def _style_start_button(self, stopped=True):
        try:
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
        except Exception:
            pass

    def _style_autofetch_button(self, running=False):
        try:
            self.autofetch_btn.background_normal = ""
            self.autofetch_btn.background_down = ""
            if running:
                self.autofetch_btn.background_color = (0.05, 0.6, 0.05, 1.0)
                self.autofetch_btn.text = "Autofetch On"
            else:
                self.autofetch_btn.background_color = (0.45, 0.45, 0.45, 1.0)
                self.autofetch_btn.text = "Autofetch Off"
            self.autofetch_btn.color = (1, 1, 1, 1)
        except Exception:
            pass

    def _style_menu_button(self, b):
        try:
            b.background_normal = ""
            b.background_down = ""
            b.background_color = (0.10, 0.10, 0.10, 0.80)
            b.color = (1, 1, 1, 1)
        except Exception:
            pass
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
        add_toggle("QR polygon overlay", False, lambda v: setattr(self.preview, "show_qr", v))

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

    # ---------- Android CSV helpers ----------
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

    def _open_csv_filechooser(self):
        if platform != 'android':
            self._log_internal("CSV load is Android-only (SAF). Please run on-device to load CSV.")
            return
        return self._open_csv_saf()

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
                    self._style_connect_button()
                    self.start_btn.disabled = False
                else:
                    self.connected = False
                    self.status.text = f"Status: connect failed ({status})"
                    self._log_internal(f"Connect failed: {status}")
                    self._style_connect_button()
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

    def _maybe_commit_author(self, payload: str, source="manual"):
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

    # ---------- liveview / decoder ----------
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
        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        self._fetch_thread = threading.Thread(target=self._liveview_fetch_loop, daemon=True)
        self._fetch_thread.start()

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

    # ---------- thumbnails, overlay, full-res (unchanged) ----------
    def _download_thumb_for_path(self, ccapi_path: str):
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

        if self._overlay_active and (self._overlay_thumb_index == idx):
            self._log_internal(f"Thumbnail {idx} tapped while overlay active: closing overlay")
            self._clear_thumb_highlight()
            self.preview.clear_overlay_texture()
            self._overlay_active = False
            self._overlay_thumb_index = None
            if self._frame_texture is not None:
                try:
                    self.preview.set_texture(self._frame_texture)
                except Exception:
                    pass
            return True

        self._highlight_thumb(idx)

        if saved_path and os.path.exists(saved_path):
            tex = self._create_texture_from_jpeg_file(saved_path, rotate=rot, flip_vertical=True)
            if tex:
                Clock.schedule_once(lambda *_: self._show_overlay_with_texture(tex, idx), 0)
                threading.Thread(target=self._fetch_full_and_replace, args=(ccapi_path, idx), daemon=True).start()
                return True

        if idx < len(self._thumb_textures):
            thumb_tex = self._thumb_textures[idx]
            Clock.schedule_once(lambda *_: self._show_overlay_with_texture(thumb_tex, idx), 0)
            threading.Thread(target=self._fetch_full_and_replace, args=(ccapi_path, idx), daemon=True).start()
            return True

        threading.Thread(target=self._download_thumb_and_overlay, args=(ccapi_path, idx), daemon=True).start()
        return True

    def _download_thumb_and_overlay(self, ccapi_path, idx):
        self._download_thumb_for_path(ccapi_path)
        if idx < len(self._thumb_textures):
            tex = self._thumb_textures[idx]
            Clock.schedule_once(lambda *_: self._show_overlay_with_texture(tex, idx), 0)
            threading.Thread(target=self._fetch_full_and_replace, args=(ccapi_path, idx), daemon=True).start()

    def _show_overlay_with_texture(self, texture: Texture, thumb_index: int):
        if texture is None:
            return
        self._overlay_active = True
        self._overlay_thumb_index = thumb_index
        try:
            self.preview.set_overlay_texture(texture)
        except Exception as e:
            self._log_internal(f"Failed to set overlay texture: {e}")

    def _fetch_full_and_replace(self, ccapi_path: str, thumb_index: int):
        request_id = time.time()
        self._pending_full_fetches[thumb_index] = request_id
        full_url = f"https://{self.camera_ip}{ccapi_path}"
        self._log_internal(f"Fetching full-res: {full_url} (req={request_id})")
        try:
            resp = self._session.get(full_url, timeout=20.0, stream=True)
            if resp.status_code != 200 or not resp.content:
                self._log_internal(f"Full image download failed: {resp.status_code} (req={request_id})")
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

        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self._log_internal("cv2.imdecode returned None for full image")
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

        def _apply_full_on_main(_dt):
            cur_req = self._pending_full_fetches.get(thumb_index)
            if cur_req != request_id:
                try:
                    if thumb_index in self._pending_full_fetches and self._pending_full_fetches[thumb_index] == request_id:
                        del self._pending_full_fetches[thumb_index]
                except Exception:
                    pass
                return

            if (not getattr(self, "_overlay_active", False)) or (self._overlay_thumb_index != thumb_index):
                try:
                    if thumb_index in self._pending_full_fetches:
                        del self._pending_full_fetches[thumb_index]
                except Exception:
                    pass
                return

            tex = self._create_texture_from_rgb(rgb_bytes, w, h, flip_vertical=True)
            if not tex:
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
            try:
                if thumb_index < len(self._thumb_textures):
                    self._thumb_textures[thumb_index] = tex
                    if thumb_index < len(self._thumb_images):
                        self._thumb_images[thumb_index].texture = tex
            except Exception:
                pass
            self._log_internal("Full-res applied to overlay (overlay remains until user taps thumb again to close)")
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
        try:
            self._qr_finder_stop.set()
        except Exception:
            pass


if __name__ == "__main__":
    VolumeToolkitApp().run()
