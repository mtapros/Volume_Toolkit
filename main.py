import os
import threading
import time
from datetime import datetime
from io import BytesIO

import kivy
kivy.require("2.0.0")

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput


APP_VERSION = "1.0.9"


class VolumeToolkitApp(App):
    show_log = BooleanProperty(True)

    def build(self):
        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        # ---------------- Header ----------------
        header = BoxLayout(size_hint=(1, None), height=dp(40))
        header.add_widget(Label(
            text=f"Volume Toolkit {APP_VERSION}",
            font_size=sp(18)
        ))
        root.add_widget(header)

        # ---------------- Menu Button ----------------
        self.menu_btn = Button(
            text="OPEN MENU",
            size_hint=(1, None),
            height=dp(50),
            font_size=sp(18),
            background_color=(0.2, 0.6, 0.9, 1)
        )
        root.add_widget(self.menu_btn)

        # ---------------- Main Content Placeholder ----------------
        root.add_widget(Label(
            text="Main app content here",
            font_size=sp(16)
        ))

        # ---------------- Log ----------------
        self.log_label = Label(
            text="",
            size_hint=(1, None),
            height=dp(120),
            halign="left",
            valign="top",
            font_size=sp(11)
        )
        self.log_label.bind(
            size=lambda *_: setattr(self.log_label, "text_size", self.log_label.size)
        )
        root.add_widget(self.log_label)

        # ---------------- Menu Modal ----------------
        self.menu_modal = self._build_menu_modal()

        def open_menu(*_):
            self.log("[MENU] Button pressed")
            self.menu_modal.open()

        self.menu_btn.bind(on_press=open_menu)

        self.log("App ready")
        return root

    # ============================================================
    # Modal Menu
    # ============================================================

    def _build_menu_modal(self):
        modal = ModalView(
            size_hint=(0.95, 0.95),
            auto_dismiss=True
        )

        root = BoxLayout(
            orientation="vertical",
            padding=dp(10),
            spacing=dp(6)
        )

        # -------- Header --------
        header = BoxLayout(size_hint=(1, None), height=dp(44))
        header.add_widget(Label(
            text="Menu",
            font_size=sp(18),
            bold=True
        ))

        close_btn = Button(
            text="Close",
            size_hint=(None, 1),
            width=dp(90)
        )
        close_btn.bind(on_press=lambda *_: modal.dismiss())
        header.add_widget(close_btn)
        root.add_widget(header)

        # -------- Scrollable Content --------
        scroll = ScrollView()
        content = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=dp(6)
        )
        content.bind(minimum_height=content.setter("height"))
        scroll.add_widget(content)

        def section(title):
            content.add_widget(Label(
                text=title,
                size_hint_y=None,
                height=dp(30),
                font_size=sp(15)
            ))

        def action(text, fn):
            b = Button(
                text=text,
                size_hint_y=None,
                height=dp(42),
                font_size=sp(14)
            )
            b.bind(on_press=lambda *_: fn())
            content.add_widget(b)

        def toggle(text, initial, fn):
            row = BoxLayout(size_hint_y=None, height=dp(36))
            row.add_widget(Label(text=text))
            cb = CheckBox(active=initial)
            cb.bind(active=lambda _i, v: fn(v))
            row.add_widget(cb)
            content.add_widget(row)

        # -------- Menu Items --------
        section("Metrics")
        action("Toggle metrics drawer", lambda: self.log("Metrics toggled"))

        section("UI")
        toggle("Show log", True, self._set_log_visible)

        section("Debug")
        action("Test log entry", lambda: self.log("Menu test action"))

        root.add_widget(scroll)
        modal.add_widget(root)
        return modal

    # ============================================================
    # Logging
    # ============================================================

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self.log_label.text += line + "\n"

    def _set_log_visible(self, visible):
        self.show_log = bool(visible)
        self.log_label.opacity = 1 if visible else 0
        self.log_label.height = dp(120) if visible else 0


if __name__ == "__main__":
    VolumeToolkitApp().run()
