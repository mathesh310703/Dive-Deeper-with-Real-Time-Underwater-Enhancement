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


# -------------------- DEHAZING UTILS --------------------
def dark_channel(img, size=15):
    min_img = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_img, kernel)


def estimate_atmospheric_light(img, dark):
    h, w = dark.shape
    num_pixels = h * w
    num_bright = max(int(num_pixels * 0.001), 1)

    dark_vec = dark.reshape(num_pixels)
    img_vec = img.reshape(num_pixels, 3)

    indices = np.argsort(dark_vec)[-num_bright:]
    return np.mean(img_vec[indices], axis=0)


def estimate_transmission(img, A, omega=0.95, size=15):
    norm_img = img / (A + 1e-6)
    return 1 - omega * dark_channel(norm_img, size)


def recover_image(img, t, A, t0=0.1):
    t = np.clip(t, t0, 1)
    return (img - A) / t[..., None] + A


def dehaze(img):
    img = img.astype(np.float32) / 255.0
    dark = dark_channel(img)
    A = estimate_atmospheric_light(img, dark)
    t = estimate_transmission(img, A)
    recovered = recover_image(img, t, A)
    return np.clip(recovered * 255, 0, 255).astype(np.uint8)


# -------------------- MODEL --------------------
class VideoEnhancer:
    def __init__(self):
        self.encoder = Encoder().to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.load_weights()

    def load_weights(self):
        self.encoder.load_state_dict(torch.load("./checkpoints/encoder.pth", map_location=DEVICE))
        self.decoder.load_state_dict(torch.load("./checkpoints/decoder.pth", map_location=DEVICE))
        self.encoder.eval()
        self.decoder.eval()

    def enhance_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            enhanced = self.decoder(self.encoder(tensor))

        enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if enhanced.min() < 0:
            enhanced = (enhanced + 1.0) / 2.0

        enhanced = np.clip(enhanced, 0, 1)
        enhanced = (enhanced * 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        # LAB preservation
        orig_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        enh_lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

        l_enh, _, _ = cv2.split(enh_lab)
        _, a, b = cv2.split(orig_lab)

        enhanced = cv2.merge((l_enh, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Dehazing
        enhanced = dehaze(enhanced)

        # Final contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(1.5, (8, 8)).apply(l)

        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# -------------------- WEBSITE-STYLE APP --------------------
class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Harsh Environment Vision Enhancement")
        self.root.geometry("1280x820")
        self.root.configure(bg="white")

        self.model = VideoEnhancer()
        self.cap = None
        self.fps = 0.0

        container = tk.Frame(root, bg="white")
        container.pack(fill="both", expand=True)

        # -------- HERO --------
        tk.Label(
            container,
            text="DEEP SEA VISION",
            font=("Segoe UI", 30, "bold"),
            bg="white",
            fg="#0f172a"
        ).pack(pady=(30, 10))

        tk.Label(
            container,
            text=(
                "A unified deep learning system for underwater, foggy, and smoke-filled "
                "environments. The model combines multi-scale neural enhancement with "
                "physics-based dehazing to restore visibility, suppress color distortion, "
                "and improve scene clarity in real time."
            ),
            font=("Segoe UI", 11),
            bg="white",
            fg="#334155",
            wraplength=900,
            justify="center"
        ).pack(pady=(0, 30))

        # -------- VIDEO SECTION --------
        video_row = tk.Frame(container, bg="white")
        video_row.pack()

        BOX_W, BOX_H = 500, 360

        def create_video_box(parent):
            box = tk.Frame(
                parent,
                bg="black",
                width=BOX_W,
                height=BOX_H,
                highlightthickness=1,
                highlightbackground="#cbd5f5"
            )
            box.pack_propagate(False)
            label = Label(box, bg="black")
            label.pack(expand=True)
            return label, box

        self.input_frame, box1 = create_video_box(video_row)
        box1.grid(row=0, column=0, padx=25)

        self.output_frame, box2 = create_video_box(video_row)
        box2.grid(row=0, column=1, padx=25)

        # -------- LABELS --------
        labels = tk.Frame(container, bg="white")
        labels.pack(pady=8)

        tk.Label(labels, text="Original Video", font=("Segoe UI", 11, "bold"), bg="white")\
            .grid(row=0, column=0, padx=230)

        tk.Label(labels, text="Enhanced Output", font=("Segoe UI", 11, "bold"), bg="white")\
            .grid(row=0, column=1, padx=230)

        # -------- FOOTER --------
        footer = tk.Frame(container, bg="white")
        footer.pack(pady=30)

        Button(
            footer,
            text="Upload Video",
            command=self.upload_video,
            font=("Segoe UI", 12, "bold"),
            bg="#2563eb",
            fg="white",
            padx=30,
            pady=12,
            relief="flat",
            cursor="hand2"
        ).grid(row=0, column=0, padx=20)

        self.fps_label = Label(
            footer,
            text="FPS: 0.00",
            font=("Segoe UI", 12),
            bg="white",
            fg="#16a34a"
        )
        self.fps_label.grid(row=0, column=1)

    def upload_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if path:
            self.cap = cv2.VideoCapture(path)
            self.update_frames()

    def update_frames(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (500, 360))

        t0 = time.time()
        enhanced = self.model.enhance_frame(frame)
        t1 = time.time()

        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(t1 - t0, 1e-6))
        self.fps_label.config(text=f"FPS: {self.fps:.2f}")

        img_in = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        img_out = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)))

        self.input_frame.config(image=img_in)
        self.input_frame.image = img_in

        self.output_frame.config(image=img_out)
        self.output_frame.image = img_out

        self.root.after(1, self.update_frames)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
