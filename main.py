import tkinter as tk
from tkinter import Label, Button, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import time

from encoder import Encoder
from decoder import Decoder

# -------------------- DEVICE --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- MODEL WRAPPER --------------------
class VideoEnhancer:
    def __init__(self):
        self.encoder = Encoder().to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.load_weights()

    def load_weights(self):
        self.encoder.load_state_dict(
            torch.load("./checkpoints/encoder.pth", map_location=DEVICE)
        )
        self.decoder.load_state_dict(
            torch.load("./checkpoints/decoder.pth", map_location=DEVICE)
        )
        self.encoder.eval()
        self.decoder.eval()

    def enhance_frame(self, frame):
        """
        Model → Auto-normalization → LAB color preservation → CLAHE
        """

        # ---------- INPUT ----------
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ---------- MODEL ----------
        with torch.no_grad():
            features = self.encoder(tensor)
            enhanced = self.decoder(features)

        enhanced = (
            enhanced.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        # ---------- AUTO NORMALIZATION ----------
        if enhanced.min() < 0:
            enhanced = (enhanced + 1.0) / 2.0

        enhanced = np.clip(enhanced, 0.0, 1.0)
        enhanced = (enhanced * 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        # ---------- COLOR PRESERVATION ----------
        orig_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        enh_lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

        l_enh, _, _ = cv2.split(enh_lab)
        _, a_orig, b_orig = cv2.split(orig_lab)

        enhanced = cv2.merge((l_enh, a_orig, b_orig))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # ---------- LOCAL CONTRAST ----------
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        return enhanced


# -------------------- TKINTER APP --------------------
class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Live Video Enhancement System")
        self.root.geometry("1280x720")
        self.root.configure(bg="#0f172a")

        self.model = VideoEnhancer()
        self.cap = None
        self.fps = 0.0

        # ---------- GRID ----------
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # ---------- HEADER ----------
        header = tk.Label(
            root,
            text="AI-Powered Real-Time Video Enhancement",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#0f172a"
        )
        header.grid(row=0, column=0, pady=20)

        # ---------- MAIN ----------
        main = tk.Frame(root, bg="#0f172a")
        main.grid(row=1, column=0, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)

        # ---------- ORIGINAL PANEL ----------
        left = tk.Frame(main, bg="#020617", padx=15, pady=15)
        left.grid(row=0, column=0, padx=20, pady=10)

        Label(
            left,
            text="Original Video",
            font=("Segoe UI", 14, "bold"),
            fg="#e5e7eb",
            bg="#020617"
        ).pack(pady=(0, 10))

        self.input_frame = Label(left, bg="black")
        self.input_frame.pack()

        # ---------- ENHANCED PANEL ----------
        right = tk.Frame(main, bg="#020617", padx=15, pady=15)
        right.grid(row=0, column=1, padx=20, pady=10)

        Label(
            right,
            text="Enhanced Output",
            font=("Segoe UI", 14, "bold"),
            fg="#e5e7eb",
            bg="#020617"
        ).pack(pady=(0, 10))

        self.output_frame = Label(right, bg="black")
        self.output_frame.pack()

        # ---------- FOOTER ----------
        footer = tk.Frame(root, bg="#0f172a")
        footer.grid(row=2, column=0, pady=15)

        self.upload_btn = Button(
            footer,
            text="Upload Video",
            command=self.upload_video,
            font=("Segoe UI", 12, "bold"),
            bg="#2563eb",
            fg="white",
            padx=20,
            pady=8,
            relief="flat",
            cursor="hand2"
        )
        self.upload_btn.grid(row=0, column=0, padx=20)

        self.fps_label = Label(
            footer,
            text="FPS: 0.00",
            font=("Segoe UI", 12),
            fg="#22c55e",
            bg="#0f172a"
        )
        self.fps_label.grid(row=0, column=1)

    def upload_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if path:
            self.cap = cv2.VideoCapture(path)
            self.update_frames()

    def update_frames(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        frame = cv2.resize(frame, (480, 360))

        # ---------- ORIGINAL ----------
        input_img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        )
        self.input_frame.configure(image=input_img)
        self.input_frame.image = input_img

        # ---------- ENHANCED ----------
        start = time.time()
        enhanced = self.model.enhance_frame(frame)
        end = time.time()

        inst_fps = 1.0 / max(end - start, 1e-6)
        self.fps = 0.9 * self.fps + 0.1 * inst_fps
        self.fps_label.config(text=f"FPS: {self.fps:.2f}")

        output_img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        )
        self.output_frame.configure(image=output_img)
        self.output_frame.image = output_img

        self.root.after(1, self.update_frames)

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


# -------------------- MAIN --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
