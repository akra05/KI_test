import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np

# Bildgröße größer als 28x28, um sauber zeichnen zu können
CANVAS_SIZE = 280
IMG_SIZE = 28

# Bild vorbereiten
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # Weißer Hintergrund
draw = ImageDraw.Draw(image)

def draw_line(event):
    x, y = event.x, event.y
    r = 5  # Pinselgröße
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
        


def process_and_show():
    # Verkleinern auf 28x28
    img_small = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    
    # In reines Schwarz-Weiß umwandeln (0 oder 1)
    img_bw = img_small.point(lambda x: 0 if x < 128 else 1, '1')  # Schwellenwert 128

    # Umwandeln in NumPy-Array
    arr = get_drawing_matrix(img_bw)
    print(arr)  # Nur 0 und 1

    # Optional: abspeichern
    img_bw.save("gezeichnete_zahl.png")

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
