import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageOps, ImageGrab
import tensorflow 
import keras

# 加载模型
model = keras.models.load_model("D:\python3.x\mychat\model.h5")

window = tk.Tk()
window.title("手写数字识别（MNIST）")
window.geometry("560x560")

canvas = tk.Canvas(window, width=400, height=400, bg='white')
canvas.pack()

lastx, lasty = None, None

def xy(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y), width=25, fill='black', capstyle=ROUND, smooth=True)
    lastx, lasty = event.x, event.y

canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", addLine)


def predict_digit():
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
  
    img = ImageGrab.grab().crop((x, y, x1, y1))

    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28),Image.Resampling.LANCZOS)
    img = np.array(img)
    Image.fromarray(img).show()
    img = 255 - img

    img = img / 255.0
    img = img.reshape(1,28, 28)

    pred = model.predict(img)
    digit = np.argmax(pred)

    result_label.config(text=f"预测结果：{digit}")


def clear_canvas(): 
    canvas.delete("all")
    result_label.config(text="预测结果： ")

predict_button = tk.Button(window, text="识别", command=predict_digit)
predict_button.pack(pady=10)

clear_button = tk.Button(window, text="清空", command=clear_canvas)
clear_button.pack(pady=5)

result_label = tk.Label(window, text="预测结果：", font=("Arial", 16))
result_label.pack()

window.mainloop()
