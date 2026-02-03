# Volume Toolkit - main.py
# Reworked QR Find (from scratch) — updated to run indefinitely until a QR is found
# or the user presses the QR Find button again to cancel. When a QR is found its
# text is placed into the "QR Payload" placeholder (displayed under Exif Payload).
#
# Key behavior changes from prior version:
# - QR Find is a toggle: press to start scanning indefinitely, press again to stop.
# - The finder samples the latest decoded frame repeatedly (poll interval QR_FIND_POLL_S)
#   until it finds a QR or the user cancels.
# - UI updated on main thread via Clock.schedule_once; frames are shared under self._lock.
# - No auto-committing of QR to camera author; QR is stored in QR Payload for user action.

import os
import json
import threading
import time
from datetime import datetime
from io import BytesIO
import csv
import queue
import sys

import requests
import urllib3

# Set KIVY_HOME early on Android so Kivy won't try to copy icons into the bundle dir
if os.environ.get("ANDROID_ARGUMENT"):
    private_dir = os.environ.get("ANDROID_PRIVATE")
    if private_dir:
        kivy_home = os.path.join(private_dir, ".kivy")
        try:
            os.makedirs(kivy_home, exist_ok=True)
        except Exception:
            pass
        os.environ["KIVY_HOME"] = kivy_home

import kivy
kivy.require("2.3.0")

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

            # polygon line (kept, but used only if show_qr True)
            self._c_qr = Color(0.0, 1.0, 0.0, 0.95)
            self._ln_qr = Line(width=lw_qr, close=True)

        self._ln_grid_list = []

        # Overlay rectangle state (we will create it in img.canvas.after so it sits above texture)
        self._overlay_texture = None
        self._overlay_rect = None
        self._overlay_rect_color = None

        self._qr_points_px = None

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

        self._redraw()

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

    # QR finder configuration
    QR_FIND_POLL_S = 0.12      # how often to sample latest frame while searching

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

        # QR finder (from scratch)
        self._qr_detector = cv2.QRCodeDetector()
        self._qr_finder_thread = None
        self._qr_finder_stop = threading.Event()
        self._qr_found_text = ""          # last found QR text
        self._qr_payload = ""             # placeholder for "QR Payload"

        # latest decoded BGR for any background tool (protected by _lock)
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

        # author / CSV
        self.author_max_chars = 60
        self._last_committed_author = None
        self._author_update_in_flight = False
        self.manual_payload = ""

        self.csv_headers = []
        self.csv_rows = []
        self.selected_headers = []
        self._headers_popup = None

        # store which source is currently shown in the Exif Payload label:
        # "QR", "CSV", "MANUAL", or "none"
        self._payload_source = "none"

        # thumbnails
        self._thumb_textures = []
        self._thumb_images = []
        self._thumb_paths = []
        self._thumb_saved_paths = []

        # track pending full-res fetches to avoid stale-applying results
        self._pending_full_fetches = {}

        # storage
        self.download_dir = "downloads"
        self.thumb_dir = "thumbs"
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.thumb_dir, exist_ok=True)

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
        self._column_filters = {}
        self._column_sorts = {}
        self._selected_csv_row = None
        self._selected_author_payload = None

        # Autofetch state
        self.autofetch_running = False

    # ---------- utility/log ----------
    def _log_internal(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]
        if getattr(self, "show_log", False):
            self._refresh_log_view()

        # also echo to stdout so adb logcat / console sees it
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

    # ---------- centralized Exif payload setter ----------
    def _set_exif_payload(self, payload: str, source: str):
        """
        Atomically set the authoritative Exif payload and source.
        source: "QR", "CSV", "MANUAL", or "none"
        This schedules UI updates on the main thread and updates internal state deterministically.
        """
        try:
            payload = (payload or "").strip()
            if not payload:
                source = "none"

            # update internal authoritative fields
            if source == "QR":
                self._qr_found_text = payload
                # clear CSV selection when QR becomes authoritative
                self._selected_csv_row = None
                self._selected_author_payload = None
            elif source == "CSV":
                self._selected_author_payload = payload
            elif source == "MANUAL":
                self.manual_payload = payload
            else:
                # none
                self._qr_found_text = ""
                self._selected_author_payload = None
                self.manual_payload = ""

            self._payload_source = source

            # schedule UI update on main thread
            def _update_on_main(_dt):
                if payload:
                    label_text = f"Exif Payload ({self._payload_source}): {payload}"
                    status_text = f"Exif Payload ({self._payload_source}): {payload[:80]}"
                else:
                    label_text = "Exif Payload (none): none"
                    status_text = ("Exif Payload: on" if getattr(self, "live_running", False) else "Exif Payload: none")
                try:
                    self.qr_last_label.text = label_text[:200]
                except Exception:
                    pass
                try:
                    self.qr_status.text = status_text
                except Exception:
                    pass

            Clock.schedule_once(_update_on_main, 0)
        except Exception as e:
            try:
                self._log_internal(f"_set_exif_payload error: {e}")
            except Exception:
                pass

    # ---------- QR Find (updated to run until found or canceled) ----------
    def _on_qr_find_pressed(self):
        """
        QR Find button is a toggle:
        - If no finder is running: start finder thread that scans indefinitely until it finds
          a QR or until the user presses the QR Find button again to cancel.
        - If finder is running: signal it to stop (user canceled).
        """
        # If a finder is already running, signal it to stop (user pressed again)
        if self._qr_finder_thread is not None and self._qr_finder_thread.is_alive():
            self._log_internal("QR Finder: user requested cancel")
            self._qr_finder_stop.set()
            # UI revert handled by worker finish callback; but also update button immediately
            try:
                self.qr_find_btn.text = "QR Find"
            except Exception:
                pass
            return

        # Start a new finder run
        self._log_internal("QR Finder: starting (will run until found or cancelled)")
        # Clear previous QR payload
        self._set_qr_payload("", source="none")

        # Update button UI immediately (main thread)
        try:
            self.qr_find_btn.text = "Stop Find"
            self.qr_find_btn.disabled = False
        except Exception:
            pass

        # Start background worker
        self._qr_finder_stop.clear()
        self._qr_finder_thread = threading.Thread(target=self._qr_finder_worker, daemon=True)
        self._qr_finder_thread.start()

    def _set_qr_payload(self, payload: str, source: str = "QR"):
        """
        Atomic setter for the new QR Payload placeholder.
        Updates UI label on main thread under Exif Payload.
        """
        self._qr_payload = (payload or "").strip()
        def _update(_dt):
            try:
                if self._qr_payload:
                    self.qr_payload_label.text = f"QR Payload: {self._qr_payload}"
                else:
                    self.qr_payload_label.text = "QR Payload: none"
                # reflect the payload source if desired
                if source == "QR" and self._qr_payload:
                    self._payload_source = "QR"
            except Exception:
                pass
        Clock.schedule_once(_update, 0)

    def _qr_finder_worker(self):
        """
        Dedicated finder that runs until a QR is found or the user cancels.
        - samples self._latest_decoded_bgr under lock
        - tries detectAndDecodeMulti (if available) then detectAndDecode
        - stops immediately when a non-empty decoded string is found
        - returns (and updates UI) either on found or on cancellation
        """
        found_text = None
        found_points = None

        while not self._qr_finder_stop.is_set():
            with self._lock:
                bgr = None
                if self._latest_decoded_bgr is not None:
                    bgr = self._latest_decoded_bgr.copy()

            if bgr is None:
                # no frame yet; wait and check cancellation
                if self._qr_finder_stop.wait(self.QR_FIND_POLL_S):
                    break
                continue

            try:
                texts = []
                pts = None
                # Prefer multi-detect when available
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

                # Fallback single detect
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
                # log and continue scanning until canceled
                self._log_internal(f"QR Finder exception: {e}")

            # sleep/poll interval, but break early if canceled
            if self._qr_finder_stop.wait(self.QR_FIND_POLL_S):
                break

        # finished (either found or canceled). Update UI on main thread.
        def _finish_ui(_dt):
            try:
                if found_text:
                    self._log_internal(f"QR Finder: found -> '{found_text}'")
                    self._set_qr_payload(found_text, source="QR")
                    # show overlay if points available and user enabled polygon overlay
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
                    # keep QR Payload as-is (cleared at start), reflect none
                    self._set_qr_payload("", source="none")
                # restore button state
                try:
                    self.qr_find_btn.text = "QR Find"
                    self.qr_find_btn.disabled = False
                except Exception:
                    pass
            except Exception:
                pass

        # ensure stop flag is cleared for future runs
        self._qr_finder_stop.clear()
        Clock.schedule_once(_finish_ui, 0)

    # ---------- texture helpers ----------
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

        # Connect / Start / Autofetch / QR Find row
        row2 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.connect_btn = Button(text="Connect", font_size=sp(16), size_hint=(None, 1), width=dp(120))
        self._style_connect_button(initial=True)
        self.start_btn = Button(text="Start", disabled=True, font_size=sp(16), size_hint=(None, 1), width=dp(140))
        self._style_start_button(stopped=True)
        self.autofetch_btn = Button(text="Autofetch Off", font_size=sp(14), size_hint=(None, 1), width=dp(140))
        self._style_autofetch_button(running=False)
        self.qr_find_btn = Button(text="QR Find", font_size=sp(14), size_hint=(None, 1), width=dp(120))
        row2.add_widget(self.connect_btn)
        row2.add_widget(self.start_btn)
        row2.add_widget(self.autofetch_btn)
        row2.add_widget(self.qr_find_btn)
        root.add_widget(row2)

        # Exif Payload / QR Payload / status labels
        self.qr_last_label = Label(text="Exif Payload (none): none", size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.qr_last_label)

        # NEW: QR Payload placeholder displayed under Exif Payload
        self.qr_payload_label = Label(text="QR Payload: none", size_hint=(1, None), height=dp(20), font_size=sp(12))
        root.add_widget(self.qr_payload_label)

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

        # Footer: CSV + Push Update
        footer = BoxLayout(orientation="horizontal", size_hint=(1, None), height=dp(48), spacing=dp(6))
        footer.add_widget(Label())
        self.csv_btn = Button(text="CSV", size_hint=(None, 1), width=dp(140))
        self.push_update_btn = Button(text="Push Update", size_hint=(None, 1), width=dp(140))
        footer.add_widget(self.csv_btn)
        footer.add_widget(self.push_update_btn)
        root.add_widget(footer)

        # Fit preview
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
        self.autofetch_btn.bind(on_press=lambda *_: (self._on_autofetch_pressed(), self._style_autofetch_button(self.autofetch_running)))
        self.qr_find_btn.bind(on_press=lambda *_: self._on_qr_find_pressed())
        self.csv_btn.bind(on_release=lambda *_: self._open_csv_menu())
        self.push_update_btn.bind(on_release=lambda *_: self._push_update())

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

    # ---------- remaining methods unchanged (thumbnails, decoder, networking, menus, etc.) ----------
    # For brevity the rest of the file is unchanged from the prior reworked QR Find file:
    # - _create_texture_from_bgr_np, _create_texture_from_jpeg_bytes, _json_call, menu/popups,
    #   connect_camera/_connect_worker, _maybe_commit_author/_commit_author_worker,
    #   start_liveview/stop_liveview/_liveview_fetch_loop/_decoder_loop,
    #   _download_thumb_for_path/_fetch_full_and_replace, list_all_images,
    #   polling, and cleanup (on_stop).
    #
    # These functions are present in the repository file and remain identical except for the
    # QR Find changes above.
    #
    # (If you want the complete file reproduced here I can paste it again, but only the QR Find
    # behavior was modified to satisfy: "run indefinitely UNTIL it finds a QR or the User presses
    # the QR Find button".)
    #
    # End of file.
