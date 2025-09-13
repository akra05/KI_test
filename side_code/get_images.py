import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import math 

# Bildgröße größer als 28x28, um sauber zeichnen zu können
CANVAS_SIZE = 280
IMG_SIZE = 28

# Bild vorbereiten
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # Weißer Hintergrund
draw = ImageDraw.Draw(image)

def draw_line(event):
    x, y = event.x, event.y
    r = 2  # Pinselgröße
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
    draw.ellipse([x - r, y - r, x + r, y + r], fill=255)  # Schwarz auf weißem Bild

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)

def get_drawing_matrix(img):
    matrix = []
    for i in range(28):
        
        for j in range(28):
            value = img.getpixel((i,j))
            matrix.append(value)
        

    return matrix
        


from scipy.ndimage import center_of_mass

def process_and_show():
    # 1. Runterskalieren auf 28x28
    img_small = image.resize((28, 28), Image.LANCZOS)
    img_gray = img_small.convert("L")  # 0-255 Graustufen
    img_arr = np.array(img_gray, dtype=np.float32)

    # 2. Bounding Box
    coords = np.argwhere(img_arr > 0)
    if coords.size == 0:
        return  # leeres Bild
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img_cropped = img_arr[y0:y1+1, x0:x1+1]

    # 3. Skalieren auf max 20x20
    max_size = 20
    h, w = img_cropped.shape
    scale = max(h, w) / max_size
    new_w = int(w / scale)
    new_h = int(h / scale)
    img_resized = Image.fromarray(img_cropped).resize((new_w, new_h), Image.LANCZOS)

    # 4. Neues 28x28 Bild und zentrieren
    img_final = np.zeros((28,28), dtype=np.float32)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    img_final[start_y:start_y+new_h, start_x:start_x+new_w] = np.array(img_resized, dtype=np.float32)

    # 5. Optional: Verschiebung nach Schwerpunkt (wie MNIST)
    cy, cx = center_of_mass(img_final)
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))
    img_final = np.roll(img_final, shift_y, axis=0)
    img_final = np.roll(img_final, shift_x, axis=1)

    # 6. Normalisieren auf 0-1
    arr = img_final / 255.0
    arr = arr.flatten().tolist()

    # 7. Vorhersage
    pred, probs = predict(arr)
    print(f"Vorhersage: {pred}")
    print(f"Wahrscheinlichkeiten: {probs}")


def predict(array):
    result_arr = [0]*10
    for i in range(10):
        for j in range(784):
            result_arr[i] += class_matrix[i][j] * array[j]
    # Softmax
    exp_vals = [math.exp(x) for x in result_arr]
    total = sum(exp_vals)
    result_arr = [v/total for v in exp_vals]
    predicted_class = result_arr.index(max(result_arr))
    return predicted_class, result_arr


class_matrix = np.load(r"E:\Eigene Projekte\Zahlerkennung\class_matrix.npy").tolist()
learn_rate = 0

# Tkinter-Setup
root = tk.Tk()
root.title("Zeichne eine Zahl")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.pack()

canvas.bind("<B1-Motion>", draw_line)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Löschen", command=clear_canvas).pack(side=tk.LEFT)
tk.Button(btn_frame, text="Verarbeiten", command=process_and_show).pack(side=tk.LEFT)

root.mainloop()
