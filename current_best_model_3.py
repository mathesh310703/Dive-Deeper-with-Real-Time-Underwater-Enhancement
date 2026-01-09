import tkinter as tk
from tkinter import Label, Button, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import time

from encoder import Encoder
from decoder import Decoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== MODEL ====================
class VideoEnhancer:
    def __init__(self):
        self.encoder = Encoder().to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.load_weights()
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

    def load_weights(self):
        self.encoder.load_state_dict(torch.load("./checkpoints/encoder.pth", map_location=DEVICE))
        self.decoder.load_state_dict(torch.load("./checkpoints/decoder.pth", map_location=DEVICE))
        self.encoder.eval()
        self.decoder.eval()

    def enhance_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)\
            .permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            enhanced_t = self.decoder(self.encoder(img_t))

        enhanced_t = torch.clamp(
            (enhanced_t + 1.0) / 2.0 if enhanced_t.min() < 0 else enhanced_t,
            0, 1
        )

        enh_np = (enhanced_t.squeeze(0).permute(1, 2, 0)
                  .cpu().numpy() * 255).astype(np.uint8)

        orig_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        enh_lab = cv2.cvtColor(cv2.cvtColor(enh_np, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)

        l_enh, _, _ = cv2.split(enh_lab)
        _, a_orig, b_orig = cv2.split(orig_lab)

        p1, p99 = np.percentile(l_enh, (1, 99))
        if p99 > p1:
            l_enh = np.clip((l_enh - p1) * 255.0 / (p99 - p1), 0, 255).astype(np.uint8)

        l_enh = self.clahe.apply(l_enh)

        return cv2.cvtColor(cv2.merge((l_enh, a_orig, b_orig)), cv2.COLOR_LAB2BGR)


# ==================== APP ====================
class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep-Sea Vision")
        self.root.geometry("1280x820")
        self.root.configure(bg="white")

        self.model = VideoEnhancer()
        self.cap = None
        self.fps = 0.0

        # ---------------- WEBSITE CONTAINER ----------------
        container = tk.Frame(root, bg="white")
        container.pack(fill="both", expand=True)

        # ---------------- HERO ----------------
        tk.Label(
            container,
            text="Deep-Sea Vision",
            font=("Segoe UI", 30, "bold"),
            bg="white",
            fg="#0f172a"
        ).pack(pady=(30, 10))

        tk.Label(
            container,
            text=(
                "A real-time underwater video enhancement system powered by a deep "
                "multi-patch hierarchical neural network. Each frame is processed "
                "across multiple spatial scales to restore visibility, correct color "
                "distortion, and enhance fine details while maintaining smooth real-time "
                "performance for autonomous underwater systems."
            ),
            font=("Segoe UI", 11),
            bg="white",
            fg="#334155",
            wraplength=900,
            justify="center"
        ).pack(pady=(0, 30))

        # ---------------- VIDEO WRAPPER ----------------
        video_wrapper = tk.Frame(container, bg="white")
        video_wrapper.pack()

        # Fixed video box size
        BOX_W, BOX_H = 500, 360

        # ORIGINAL BOX
        original_box = tk.Frame(
            video_wrapper,
            bg="black",
            width=BOX_W,
            height=BOX_H,
            highlightthickness=1,
            highlightbackground="#cbd5f5"
        )
        original_box.grid(row=0, column=0, padx=25)
        original_box.pack_propagate(False)

        self.canvas_in = Label(original_box, bg="black")
        self.canvas_in.pack(expand=True)

        # ENHANCED BOX
        enhanced_box = tk.Frame(
            video_wrapper,
            bg="black",
            width=BOX_W,
            height=BOX_H,
            highlightthickness=1,
            highlightbackground="#cbd5f5"
        )
        enhanced_box.grid(row=0, column=1, padx=25)
        enhanced_box.pack_propagate(False)

        self.canvas_out = Label(enhanced_box, bg="black")
        self.canvas_out.pack(expand=True)

        # ---------------- LABEL ROW ----------------
        label_row = tk.Frame(container, bg="white")
        label_row.pack(pady=8)

        tk.Label(
            label_row,
            text="Original Video",
            font=("Segoe UI", 11, "bold"),
            bg="white"
        ).grid(row=0, column=0, padx=230)

        tk.Label(
            label_row,
            text="Enhanced Output",
            font=("Segoe UI", 11, "bold"),
            bg="white"
        ).grid(row=0, column=1, padx=230)

        # ---------------- FOOTER ----------------
        footer = tk.Frame(container, bg="white")
        footer.pack(pady=30)

        Button(
            footer,
            text="Upload Video",
            command=self.load,
            font=("Segoe UI", 12, "bold"),
            bg="#2563eb",
            fg="white",
            padx=30,
            pady=12,
            relief="flat",
            cursor="hand2"
        ).grid(row=0, column=0, padx=20)

        self.fps_lbl = Label(
            footer,
            text="FPS: 0.00",
            font=("Segoe UI", 12),
            bg="white",
            fg="#16a34a"
        )
        self.fps_lbl.grid(row=0, column=1)

    def load(self):
        path = filedialog.askopenfilename()
        if path:
            self.cap = cv2.VideoCapture(path)
            self.run()

    def run(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (500, 360))

        t0 = time.time()
        enhanced = self.model.enhance_frame(frame)
        t1 = time.time()

        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(t1 - t0, 1e-6))
        self.fps_lbl.config(text=f"FPS: {self.fps:.2f}")

        img_i = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        img_o = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)))

        self.canvas_in.config(image=img_i)
        self.canvas_in.image = img_i

        self.canvas_out.config(image=img_o)
        self.canvas_out.image = img_o

        self.root.after(1, self.run)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
