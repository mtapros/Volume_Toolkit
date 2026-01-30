# main.py — Android-only Canon CCAPI Live View Tool

import os
import json
import csv
import time
import threading
from datetime import datetime
from io import BytesIO

import requests
import urllib3

import kivy
kivy.require("2.3.0")

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.graphics.texture import Texture
from kivy.utils import platform

# -------- ANDROID ONLY --------
if platform != "android":
    raise RuntimeError("This application is Android-only")

from jnius import autoclass
from android.activity import activity
from android.storage import app_storage_path

from PIL import Image as PILImage
import numpy as np
import cv2

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class CanonLiveViewApp(App):
    camera_ip = StringProperty("192.168.34.29")
    connected = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Networking
        self._session = requests.Session()
        self._session.verify = False

        # Live view
        self.live_running = False
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_jpeg_ts = 0
        self._frame_texture = None
        self._frame_size = None

        # QR
        self.qr_enabled = True
        self._qr_detector = cv2.QRCodeDetector()
        self._last_author = None
        self._author_busy = False

        # CSV
        self.csv_headers = []
        self.csv_rows = []
        self.selected_headers = []
        self._csv_picker_bound = False

        # Storage
        self.thumb_dir = os.path.join(app_storage_path(), "thumbs")
        os.makedirs(self.thumb_dir, exist_ok=True)

    # ---------- UI ----------

    def build(self):
        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))

        root.add_widget(Label(
            text="Canon Live View (Android)",
            font_size=sp(18),
            size_hint=(1, None),
            height=dp(40)
        ))

        self.ip_input = TextInput(
            text=self.camera_ip,
            multiline=False,
            size_hint=(1, None),
            height=dp(40)
        )
        root.add_widget(self.ip_input)

        btns = BoxLayout(size_hint=(1, None), height=dp(44), spacing=dp(6))
        self.connect_btn = Button(text="Connect")
        self.start_btn = Button(text="Start Live", disabled=True)
        self.csv_btn = Button(text="Load CSV")
        btns.add_widget(self.connect_btn)
        btns.add_widget(self.start_btn)
        btns.add_widget(self.csv_btn)
        root.add_widget(btns)

        self.status = Label(text="Status: idle", font_size=sp(13))
        root.add_widget(self.status)

        self.preview = Image(allow_stretch=True, keep_ratio=True)
        holder = AnchorLayout()
        holder.add_widget(self.preview)
        root.add_widget(holder)

        self.connect_btn.bind(on_release=lambda *_: self.connect_camera())
        self.start_btn.bind(on_release=lambda *_: self.start_liveview())
        self.csv_btn.bind(on_release=lambda *_: self.open_csv_picker())

        return root

    # ---------- Logging ----------

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    # ---------- CCAPI ----------

    def _json_call(self, method, path, payload=None):
        url = f"https://{self.camera_ip}{path}"
        try:
            if method == "GET":
                r = self._session.get(url, timeout=8)
            elif method == "PUT":
                r = self._session.put(url, json=payload, timeout=8)
            else:
                return False, None

            if r.status_code == 200:
                return True, r.json() if r.content else None
        except Exception as e:
            self.log(str(e))
        return False, None

    def connect_camera(self):
        self.camera_ip = self.ip_input.text.strip()
        ok, data = self._json_call("GET", "/ccapi/ver100/deviceinformation")
        if ok:
            self.connected = True
            self.status.text = "Status: connected"
            self.start_btn.disabled = False
        else:
            self.status.text = "Status: connection failed"

    # ---------- Live View ----------

    def start_liveview(self):
        if not self.connected or self.live_running:
            return

        ok, _ = self._json_call(
            "POST",
            "/ccapi/ver100/shooting/liveview",
            {"liveviewsize": "small"}
        )
        if not ok:
            self.status.text = "Live view failed"
            return

        self.live_running = True
        self.status.text = "Live view running"
        threading.Thread(target=self._liveview_loop, daemon=True).start()
        Clock.schedule_interval(self._display_frame, 1 / 15)

    def _liveview_loop(self):
        url = f"https://{self.camera_ip}/ccapi/ver100/shooting/liveview/flip"
        while self.live_running:
            try:
                r = self._session.get(url, timeout=5)
                if r.status_code == 200:
                    with self._lock:
                        self._latest_jpeg = r.content
                        self._latest_jpeg_ts = time.time()
            except Exception:
                pass

    def _display_frame(self, dt):
        with self._lock:
            jpeg = self._latest_jpeg

        if not jpeg:
            return

        try:
            pil = PILImage.open(BytesIO(jpeg)).convert("RGB")
            w, h = pil.size
            if self._frame_texture is None or self._frame_size != (w, h):
                self._frame_texture = Texture.create(size=(w, h), colorfmt="rgb")
                self._frame_texture.flip_vertical()
                self._frame_size = (w, h)

            self._frame_texture.blit_buffer(
                pil.tobytes(), colorfmt="rgb", bufferfmt="ubyte"
            )
            self.preview.texture = self._frame_texture

            self._qr_process(pil)

        except Exception:
            pass

    # ---------- QR + Author ----------

    def _qr_process(self, pil_img):
        if not self.qr_enabled or self._author_busy:
            return

        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        text, _, _ = self._qr_detector.detectAndDecode(bgr)
        text = text.strip()

        if text and text != self._last_author:
            self._last_author = text
            threading.Thread(
                target=self._commit_author,
                args=(text,),
                daemon=True
            ).start()

    def _commit_author(self, value):
        self._author_busy = True
        ok, _ = self._json_call(
            "PUT",
            "/ccapi/ver100/functions/registeredname/author",
            {"author": value[:60]}
        )
        self._author_busy = False
        if ok:
            self.log(f"Author set: {value}")

    # ---------- CSV (Android picker) ----------

    def open_csv_picker(self):
        Intent = autoclass("android.content.Intent")
        intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        intent.setType("text/*")

        REQUEST = 1001

        def on_result(req, res, data):
            if req != REQUEST or data is None:
                return
            uri = data.getData()
            try:
                resolver = activity.getContentResolver()
                stream = resolver.openInputStream(uri)
                raw = bytearray()
                buf = bytearray(4096)
                while True:
                    n = stream.read(buf)
                    if n == -1:
                        break
                    raw.extend(buf[:n])
                stream.close()
                self._parse_csv(bytes(raw))
            except Exception as e:
                self.log(f"CSV error: {e}")

        if not self._csv_picker_bound:
            activity.bind(on_activity_result=on_result)
            self._csv_picker_bound = True

        activity.startActivityForResult(intent, REQUEST)

    def _parse_csv(self, data):
        text = data.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(text.splitlines())
        self.csv_headers = reader.fieldnames or []
        self.csv_rows = list(reader)
        self.log(f"CSV loaded: {len(self.csv_rows)} rows")


if __name__ == "__main__":
    CanonLiveViewApp().run()
