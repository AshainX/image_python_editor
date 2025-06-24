import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
from PIL import Image, ImageTk
import cv2
import numpy as np
from backgroundRemoval import load_u2net_model, remove_background, convert_rgba_to_pil_image


class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Image Editor")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        self.init_vars()
        self.setup_ui()

    def init_vars(self):
        self.original_image = None
        self.edited_image = None
        self.filtered_image = None
        self.display_image_obj = None
        self.ratio = 1
        self.draw_color = (255, 0, 0)
        self.last_draw_point = None
        self.history = []
        self.future = []

    def setup_ui(self):
        ttk.Label(self.root, text="Image Editor", font=("Helvetica", 18, "bold")).pack(pady=10)

        self.canvas = tk.Canvas(self.root, bg="#ddd", width=600, height=400)
        self.canvas.pack(pady=10)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=5)

        ttk.Button(btn_frame, text="Upload", command=self.upload_image).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Save", command=self.save_image).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Undo", command=self.undo).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="Redo", command=self.redo).grid(row=0, column=3, padx=5)
        ttk.Button(btn_frame, text="Revert", command=self.revert_changes).grid(row=0, column=4, padx=5)

        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(pady=5)

        ttk.Button(filter_frame, text="Negative", command=self.apply_negative).grid(row=0, column=0, padx=5)
        ttk.Button(filter_frame, text="BW", command=self.apply_bw).grid(row=0, column=1, padx=5)
        ttk.Button(filter_frame, text="Sketch", command=self.apply_sketch).grid(row=0, column=2, padx=5)
        ttk.Button(filter_frame, text="Sepia", command=self.apply_sepia).grid(row=0, column=3, padx=5)

        edit_frame = ttk.Frame(self.root)
        edit_frame.pack(pady=5)

        ttk.Button(edit_frame, text="Draw", command=self.enable_drawing).grid(row=0, column=0, padx=5)
        ttk.Button(edit_frame, text="Add Text", command=self.add_text).grid(row=0, column=1, padx=5)
        ttk.Button(edit_frame, text="Crop", command=self.start_crop).grid(row=0, column=2, padx=5)
        ttk.Button(edit_frame, text="Pick Color", command=self.pick_color).grid(row=0, column=3, padx=5)

        adj_frame = ttk.Frame(self.root)
        adj_frame.pack(pady=5)

        ttk.Label(adj_frame, text="Brightness").grid(row=0, column=0, padx=5)
        self.brightness_slider = tk.Scale(adj_frame, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.adjust_brightness)
        self.brightness_slider.grid(row=0, column=1, padx=5)

        ttk.Label(adj_frame, text="Blur").grid(row=1, column=0, padx=5)
        self.blur_slider = tk.Scale(adj_frame, from_=0, to=25, orient=tk.HORIZONTAL, command=self.apply_blur)
        self.blur_slider.grid(row=1, column=1, padx=5)

    def save_history(self):
        if self.filtered_image is not None:
            self.history.append(self.filtered_image.copy())
            self.future.clear()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        self.original_image = cv2.imread(file_path)
        self.edited_image = self.original_image.copy()
        self.filtered_image = self.original_image.copy()
        self.display_image(self.filtered_image)
        self.save_history()

    def save_image(self):
        if self.filtered_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"),
                                                                ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.filtered_image)

    def undo(self):
        if self.history:
            self.future.append(self.filtered_image.copy())
            self.filtered_image = self.history.pop()
            self.display_image(self.filtered_image)

    def redo(self):
        if self.future:
            self.history.append(self.filtered_image.copy())
            self.filtered_image = self.future.pop()
            self.display_image(self.filtered_image)

    def revert_changes(self):
        self.filtered_image = self.original_image.copy()
        self.display_image(self.filtered_image)
        self.save_history()

    def apply_negative(self):
        self.filtered_image = cv2.bitwise_not(self.filtered_image)
        self.display_image(self.filtered_image)
        self.save_history()

    def apply_bw(self):
        gray = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY)
        self.filtered_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image(self.filtered_image)
        self.save_history()

    def apply_sketch(self):
        _, self.filtered_image = cv2.pencilSketch(self.filtered_image, sigma_s=60, sigma_r=0.07, shade_factor=0.03)
        self.display_image(self.filtered_image)
        self.save_history()

    def apply_sepia(self):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(self.filtered_image, kernel)
        self.filtered_image = np.clip(sepia_img, 0, 255).astype(np.uint8)
        self.display_image(self.filtered_image)
        self.save_history()

    def display_image(self, image):
        self.canvas.delete("all")
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        scale = min(600/w, 400/h) if w > 600 or h > 400 else 1
        new_w, new_h = int(w * scale), int(h * scale)
        self.ratio = image.shape[0] / new_h
        resized_img = cv2.resize(img_rgb, (new_w, new_h))
        img_pil = Image.fromarray(resized_img)
        self.display_image_obj = ImageTk.PhotoImage(img_pil)
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(new_w//2, new_h//2, image=self.display_image_obj)

    def pick_color(self):
        color = colorchooser.askcolor(title="Choose draw color")
        if color:
            self.draw_color = tuple(int(c) for c in color[0])

    def enable_drawing(self):
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)

    def start_draw(self, event):
        self.last_draw_point = (event.x, event.y)

    def draw_line(self, event):
        x1, y1 = self.last_draw_point
        x2, y2 = event.x, event.y
        self.canvas.create_line(x1, y1, x2, y2, fill=self.rgb_to_hex(self.draw_color), width=2)
        pt1 = int(x1 * self.ratio), int(y1 * self.ratio)
        pt2 = int(x2 * self.ratio), int(y2 * self.ratio)
        cv2.line(self.filtered_image, pt1, pt2, self.draw_color[::-1], thickness=2)
        self.last_draw_point = (x2, y2)

    def add_text(self):
        top = tk.Toplevel(self.root)
        top.title("Add Text")
        tk.Label(top, text="Enter text:").pack(padx=10, pady=5)
        entry = ttk.Entry(top)
        entry.pack(padx=10, pady=5)
        ttk.Button(top, text="Place Text", command=lambda: self.place_text(entry.get())).pack(pady=10)

    def place_text(self, text):
        self.canvas.bind("<Button-1>", lambda e: self.draw_text(e, text))

    def draw_text(self, event, text):
        pos = int(event.x * self.ratio), int(event.y * self.ratio)
        cv2.putText(self.filtered_image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, self.draw_color[::-1], 2)
        self.display_image(self.filtered_image)
        self.canvas.unbind("<Button-1>")
        self.save_history()

    def start_crop(self):
        self.crop_rect = None
        self.start_x = self.start_y = 0
        self.canvas.bind("<Button-1>", self.crop_start)
        self.canvas.bind("<B1-Motion>", self.crop_draw)
        self.canvas.bind("<ButtonRelease-1>", self.crop_end)

    def crop_start(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)

    def crop_draw(self, event):
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        self.crop_rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red")

    def crop_end(self, event):
        x1, y1 = sorted([self.start_x, event.x])
        x2, y2 = sorted([self.start_y, event.y])
        x1, x2 = int(x1 * self.ratio), int(x2 * self.ratio)
        y1, y2 = int(y1 * self.ratio), int(y2 * self.ratio)
        self.filtered_image = self.filtered_image[y2:y1, x1:x2] if y2 > y1 else self.filtered_image[y1:y2, x1:x2]
        self.display_image(self.filtered_image)
        self.save_history()
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def adjust_brightness(self, value):
        if self.edited_image is None:
            return
        value = int(value)
        temp_img = cv2.convertScaleAbs(self.filtered_image, alpha=1, beta=value)
        self.display_image(temp_img)

    def apply_blur(self, value):
        if self.filtered_image is None:
            return
        k = int(value)
        if k % 2 == 0:
            k += 1
        if k < 3:
            self.display_image(self.filtered_image)
            return
        blurred = cv2.GaussianBlur(self.filtered_image, (k, k), 0)
        self.display_image(blurred)

    def rgb_to_hex(self, rgb):
        return "#%02x%02x%02x" % rgb

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
