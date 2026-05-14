"""
SPT Viewer — Single Particle Tracking visualiser.

Loads a .tif video and an XML file containing particle tracks,
displays the video frame by frame with particle rectangles overlaid,
and allows interactive control of frame, playback speed, and rectangle size.

Usage
-----
    python scripts/spt_viewer.py
"""
import sys
import threading
import time
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageTk

from trajan.data import parse_particle_tree

HAND_CURSOR = "pointinghand" if sys.platform == "darwin" else "hand2"
SPACING = 0.036


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_video(path: str) -> tuple[np.ndarray, float]:
    """Load a .tif video and return (frames, pixel_spacing).

    The returned array is always 4D with shape (T, H, W, C) so downstream
    code can treat single- and multi-channel videos uniformly.

    Parameters
    ----------
    path : str
        Path to the .tif file.

    Returns
    -------
    tuple[np.ndarray, float]
        Array of shape (T, H, W, C) and pixel spacing in physical units.
        Spacing defaults to 1.0 if not found in metadata.
    """
    with tifffile.TiffFile(path) as tif:
        video = tif.asarray()
        spacing = SPACING
        if tif.is_imagej and tif.imagej_metadata:
            spacing = tif.imagej_metadata.get("spacing", SPACING)

    # Normalise to (T, H, W, C). Common input shapes:
    #   (H, W)            -> single frame, single channel
    #   (T, H, W)         -> multi-frame, single channel
    #   (T, C, H, W)      -> multi-frame, multi-channel (ImageJ-style)
    #   (T, H, W, C)      -> multi-frame, multi-channel (channels-last)
    if video.ndim == 2:
        video = video[np.newaxis, :, :, np.newaxis]
    elif video.ndim == 3:
        video = video[..., np.newaxis]
    elif video.ndim == 4:
        # Heuristic: a small leading non-time axis is the channel axis.
        # Real channel counts are tiny (typically 2–4); H and W are much larger.
        if video.shape[1] <= 4 and video.shape[1] < video.shape[-1]:
            video = np.moveaxis(video, 1, -1)  # (T, C, H, W) -> (T, H, W, C)
    else:
        raise ValueError(f"Unsupported video shape: {video.shape}")

    return video, float(spacing)


def load_tracks(path: str) -> list:
    """Load particle tracks from an XML file.

    Parameters
    ----------
    path : str
        Path to the XML file.

    Returns
    -------
    list
        List of particle trajectories as returned by parse_particle_tree.
    """
    tree = ET.parse(path)
    return parse_particle_tree(tree)


def normalise_frame(frame: np.ndarray) -> np.ndarray:
    """Normalise a single frame to uint8 for display."""
    f = frame.astype(np.float32)
    lo, hi = f.min(), f.max()
    if hi > lo:
        f = (f - lo) / (hi - lo) * 255
    return f.astype(np.uint8)


def build_frame_image(
    frame: np.ndarray,
    dets_by_frame: dict,
    frame_idx: int,
    spacing: float,
    rect_half: int,
    canvas_w: int,
    canvas_h: int,
) -> ImageTk.PhotoImage:
    """Render a video frame with particle rectangles overlaid.

    Parameters
    ----------
    frame : np.ndarray
        Frame array of shape (H, W, C). Each channel is rendered as a
        grayscale panel; multiple channels are placed side by side.
    tracks : list
        Particle trajectories from parse_particle_tree.
    frame_idx : int
        Current frame index, used to look up particle positions.
    spacing : float
        Pixel spacing to convert physical coordinates to pixels.
    rect_half : int
        Half-width of the rectangle in pixels.
    canvas_w, canvas_h : int
        Canvas dimensions for scaling.

    Returns
    -------
    ImageTk.PhotoImage
        Rendered image ready for display in a Tk canvas.
    """
    img_h, img_w, n_channels = frame.shape

    # Lay channels out horizontally with a small gap between panels.
    gap = 4 if n_channels > 1 else 0
    total_w = img_w * n_channels + gap * (n_channels - 1)
    scale = min(canvas_w / total_w, canvas_h / img_h)
    panel_w = int(img_w * scale)
    panel_h = int(img_h * scale)
    disp_w = panel_w * n_channels + gap * (n_channels - 1)

    # Build the composite image one channel at a time.
    composite = Image.new("RGB", (disp_w, panel_h), (20, 20, 20))
    for c in range(n_channels):
        chan_img = Image.fromarray(normalise_frame(frame[..., c])).convert("RGB")
        chan_img = chan_img.resize((panel_w, panel_h), Image.NEAREST)
        draw = ImageDraw.Draw(chan_img)

        # Draw bounding boxes on every panel (alignment check across channels).
        r = int(rect_half * scale)
        line_w = max(1, int(scale))
        for x, y in dets_by_frame.get(frame_idx, []):
            px = int((x / spacing) * scale)
            py = int((y / spacing) * scale)
            draw.rectangle(
                [px - r, py - r, px + r, py + r],
                outline=(255, 50, 50),
                width=line_w,
            )

        composite.paste(chan_img, (c * (panel_w + gap), 0))

    # Pad to canvas size and centre.
    padded = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    offset_x = (canvas_w - disp_w) // 2
    offset_y = (canvas_h - panel_h) // 2
    padded.paste(composite, (offset_x, offset_y))

    return ImageTk.PhotoImage(padded)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class SPTViewer(tk.Tk):
    """Single Particle Tracking viewer application."""

    CANVAS_W = 800
    CANVAS_H = 600

    def __init__(self):
        super().__init__()
        self.title("SPT Viewer")
        self.resizable(False, False)
        self.configure(bg="#141414")

        # State
        self.video: np.ndarray | None = None
        self.tracks: list = []
        self._dets_by_frame: dict[int, list[tuple[float, float]]] = {}
        self.spacing: float = 1.0
        self.frame_idx: int = 0
        self.playing: bool = False
        self._play_thread: threading.Thread | None = None
        self._photo: ImageTk.PhotoImage | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top bar — file loading
        top = tk.Frame(self, bg="#1e1e1e", pady=6)
        top.pack(fill="x", padx=0, pady=0)

        self.option_add("*Font", ("Courier", 14))

        btn_style = {"bg": "#2d2d2d", "fg": "#e0e0e0", "relief": "flat",
                     "padx": 12, "pady": 4, "cursor": "hand2",
                     "activebackground": "#3a3a3a", "activeforeground": "#ffffff",
                     "font": ("Courier", 10)}

        self._make_button(top, "Load .tif", self._load_video).pack(side="left", padx=(10, 4))
        self._make_button(top, "Load XML", self._load_xml).pack(side="left", padx=4)

        # Toggle: ask the user to load a new XML each time a video is loaded.
        self._ask_xml_var = tk.BooleanVar(value=True)
        ask_chk = tk.Checkbutton(
            top,
            text="Ask for XML on video load",
            variable=self._ask_xml_var,
            bg="#1e1e1e",
            fg="#bbb",
            activebackground="#1e1e1e",
            activeforeground="#fff",
            selectcolor="#1e1e1e",
            highlightthickness=0,
            bd=0,
            font=self.option_add,
            cursor=HAND_CURSOR,
        )
        ask_chk.pack(side="left", padx=(8, 0))

        self._file_label = tk.Label(top, text="No files loaded", bg="#1e1e1e",
                                    fg="#e0e0e0", font=self.option_add)
        self._file_label.pack(side="left", padx=12)

        self._spacing_label = tk.Label(top, text="spacing: —", bg="#1e1e1e",
                                       fg="#888", font=self.option_add)
        self._spacing_label.pack(side="right", padx=10)

        # Canvas
        self._canvas = tk.Canvas(self, width=self.CANVAS_W, height=self.CANVAS_H,
                                 bg="#141414", highlightthickness=0)
        self._canvas.pack(padx=0, pady=0)

        # Controls bar
        ctrl = tk.Frame(self, bg="#1e1e1e", pady=8)
        ctrl.pack(fill="x")

        # Frame slider
        tk.Label(ctrl, text="Frame", bg="#1e1e1e", fg="#9F9F9F",
                 font=self.option_add).grid(row=0, column=0, padx=(12, 4))

        self._frame_var = tk.IntVar(value=0)
        self._frame_slider = ttk.Scale(ctrl, from_=0, to=0, orient="horizontal",
                                       variable=self._frame_var, length=340,
                                       command=self._on_frame_slide)
        self._frame_slider.grid(row=0, column=1, padx=4)

        self._frame_label = tk.Label(ctrl, text="0 / 0", bg="#1e1e1e", fg="#ccc",
                                     font=self.option_add, width=8)
        self._frame_label.grid(row=0, column=2, padx=4)

        # Playback controls
        self._play_btn = tk.Button(ctrl, text="▶ Play", command=self._toggle_play,
                                   bg="#2d2d2d", fg="#000000", relief="flat",
                                   padx=12, pady=3, cursor="hand2",
                                   activebackground="#3a3a3a", font=self.option_add)
        self._play_btn.grid(row=0, column=3, padx=(16, 4))

        # FPS
        tk.Label(ctrl, text="FPS", bg="#1e1e1e", fg="#9F9F9F",
                 font=self.option_add).grid(row=0, column=4, padx=(12, 2))
        self._fps_var = tk.IntVar(value=10)
        ttk.Spinbox(ctrl, from_=1, to=60, textvariable=self._fps_var,
                    width=4, font=self.option_add).grid(row=0, column=5, padx=4)

        # Rectangle size
        tk.Label(ctrl, text="Rect px", bg="#1e1e1e", fg="#9F9F9F",
                 font=self.option_add).grid(row=0, column=6, padx=(16, 2))
        self._rect_var = tk.IntVar(value=18)
        rect_spin = ttk.Spinbox(ctrl, from_=1, to=100, textvariable=self._rect_var,
                                width=4, font=self.option_add,
                                command=self._refresh_frame)
        rect_spin.grid(row=0, column=7, padx=4)
        rect_spin.bind("<Return>", lambda e: self._refresh_frame())

        # Status bar
        self._status = tk.Label(self, text="Load a .tif and an XML file to begin.",
                                bg="#0e0e0e", fg="#555", font=self.option_add,
                                anchor="w", padx=10)
        self._status.pack(fill="x", side="bottom")

        self._style_ttk()

    def _style_ttk(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TScale", background="#1e1e1e", troughcolor="#2d2d2d",
                        slidercolor="#555")
        style.configure("TSpinbox", fieldbackground="#2d2d2d", background="#2d2d2d",
                        foreground="#e0e0e0", arrowcolor="#888",
                        bordercolor="#333", lightcolor="#333")

    def _make_button(self, parent, text, command):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg="#2a2a2a",
            fg="#030303",
            relief="flat",
            padx=16,
            pady=8,
            font=self.option_add,
            cursor=HAND_CURSOR,
            activebackground="#444444",
            activeforeground="#ffffff",
            borderwidth=0,
        )
        btn.bind("<Enter>", lambda e: btn.config(bg="#3a3a3a"))
        btn.bind("<Leave>", lambda e: btn.config(bg="#2a2a2a"))
        return btn

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def _load_video(self):
        path = filedialog.askopenfilename(
            title="Select .tif video",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._set_status(f"Loading {Path(path).name}…")
            self.video, self.spacing = load_video(path)
            n_frames = len(self.video)
            self._frame_slider.configure(to=n_frames - 1)
            self._frame_var.set(0)
            self.frame_idx = 0
            self._spacing_label.config(text=f"spacing: {self.spacing:.4f}")

            # Clear any previously loaded tracks — they belong to the old video.
            self._update_file_label(tif=Path(path).name)

            n_channels = self.video.shape[-1]
            ch_str = f", {n_channels} ch" if n_channels > 1 else ""
            self._set_status(f"Loaded {Path(path).name} — {n_frames} frames, "
                             f"{self.video.shape[1]}×{self.video.shape[2]} px"
                             f"{ch_str}")
            self._refresh_frame()

            # Prompt the user to load a matching XML, defaulting to the
            # directory the video was loaded from. Skipped if the toggle
            # is unchecked, but tracks are still cleared above either way.
            if self._ask_xml_var.get():
                video_dir = str(Path(path).parent)
                should_load = messagebox.askyesno(
                    title="Load matching tracks?",
                    message=(
                        "The video has changed. The currently loaded tracks "
                        "no longer apply.\n\nLoad a new XML track file now?"
                    ),
                )
                if should_load:
                    self._load_xml(initial_dir=video_dir)
        except Exception as e:
            messagebox.showerror("Error loading video", str(e))

    def _load_xml(self, initial_dir: str | None = None):
        kwargs = {
            "title": "Select XML track file",
            "filetypes": [("XML files", "*.xml"), ("All files", "*.*")],
        }
        if initial_dir:
            kwargs["initialdir"] = initial_dir
        path = filedialog.askopenfilename(**kwargs)
        if not path:
            return
        try:
            new_tracks = load_tracks(path)
            self.tracks = new_tracks
            # Index detections by frame for O(1) per-frame lookup.
            self._dets_by_frame: dict[int, list[tuple[float, float]]] = {}
            for particle in new_tracks:
                for det in particle:
                    t, x, y = det
                    self._dets_by_frame.setdefault(int(t), []).append((x, y))
            self._update_file_label(xml=Path(path).name)
            self._set_status(f"Loaded {Path(path).name} — {len(self.tracks)} tracks")
            self._refresh_frame()
        except Exception as e:
            messagebox.showerror("Error loading XML", str(e))

    def _update_file_label(self, tif: str = None, xml: str = None):
        current = self._file_label.cget("text")
        parts = {"tif": "", "xml": ""}
        if "tif:" in current:
            for part in current.split(" | "):
                if part.startswith("tif:"):
                    parts["tif"] = part.split("tif:")[1].strip()
                elif part.startswith("xml:"):
                    parts["xml"] = part.split("xml:")[1].strip()
        # `None` means "leave unchanged"; an empty string means "clear".
        if tif is not None:
            parts["tif"] = tif
        if xml is not None:
            parts["xml"] = xml
        label = " | ".join(f"{k}: {v}" for k, v in parts.items() if v)
        self._file_label.config(text=label, fg="#aaa")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh_frame(self, *_):
        if self.video is None:
            return
        n_frames = len(self.video)
        self.frame_idx = max(0, min(int(self._frame_var.get()), n_frames - 1))
        self._frame_label.config(text=f"{self.frame_idx} / {n_frames - 1}")
        photo = build_frame_image(
            frame=self.video[self.frame_idx],
            dets_by_frame=self._dets_by_frame,
            frame_idx=self.frame_idx,
            spacing=self.spacing,
            rect_half=self._rect_var.get(),
            canvas_w=self.CANVAS_W,
            canvas_h=self.CANVAS_H,
        )
        self._photo = photo
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=photo)

    def _on_frame_slide(self, val):
        self._refresh_frame()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            self._play_btn.config(text="▶ Play")
        else:
            self.playing = True
            self._play_btn.config(text="⏸ Pause")
            self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self._play_thread.start()

    def _play_loop(self):
        while self.playing:
            if self.video is None:
                break
            next_frame = (self.frame_idx + 1) % len(self.video)
            self._frame_var.set(next_frame)
            self.after(0, self._refresh_frame)
            fps = max(1, self._fps_var.get())
            time.sleep(1.0 / fps)
        self.playing = False
        self.after(0, lambda: self._play_btn.config(text="▶ Play"))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self._status.config(text=msg)
        self.update_idletasks()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _filter_macos_button_warnings():
    """Suppress the cosmetic 'min height of view' Cocoa warnings on macOS.

    These warnings come from native buttons in system-provided dialogs and
    don't affect functionality. Other stderr output is preserved.
    """
    import os
    import re
    import threading

    pattern = re.compile(r"Expected min height of view.*NSButton")
    read_fd, write_fd = os.pipe()
    real_stderr_fd = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _pump():
        with os.fdopen(read_fd, "rb") as r, os.fdopen(real_stderr_fd, "wb") as w:
            for line in r:
                if not pattern.search(line.decode("utf-8", errors="replace")):
                    w.write(line)
                    w.flush()

    threading.Thread(target=_pump, daemon=True).start()


if __name__ == "__main__":
    if sys.platform == "darwin":
        _filter_macos_button_warnings()
    app = SPTViewer()
    app.mainloop()