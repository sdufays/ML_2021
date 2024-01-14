from tkinter import *
from PIL import ImageGrab
import tkinter.font as font
import tensorflow
import numpy as np
from PIL import ImageOps
from PIL import Image, ImageFilter
from PIL import ImageEnhance
import matplotlib.patches as patches
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

model = tensorflow.keras.models.load_model("model_final_midterm.h5")


class Paint(object):
   def __init__(self):
       self.root = Tk()
       self.root.title('Number recognizer')
       self.root.wm_iconbitmap('44143.ico')
       self.root.configure(background='red')
       self.c = Canvas(self.root, bg='white', height=330, width=400)
       self.label = Label(self.root, text='Draw any one-digit number!', font=20, bg='red')
       self.label.grid(row=0, column=3)
       self.c.grid(row=1, columnspan=9)
       self.c.create_line(0, 0, 400, 0, width=20, fill='midnight blue')
       self.c.create_line(0, 0, 0, 330, width=20, fill='midnight blue')
       self.c.create_line(400, 0, 400, 330, width=20, fill='midnight blue')
       self.c.create_line(0, 330, 400, 330, width=20, fill='midnight blue')
       self.myfont = font.Font(size=20, weight='bold')
       self.predicting_button = Button(self.root, text='Predict', fg='black', bg='steel blue', height=2, width=6,
                                       font=self.myfont, command=lambda: self.classify(self.c))
       self.predicting_button.grid(row=2, column=1)
       self.clear = Button(self.root, text='Clear', fg='blue', bg='red', height=2, width=6, font=self.myfont,
                           command=self.clear)
       self.clear.grid(row=2, column=5)
       self.prediction_text = Text(self.root, height=5, width=5)
       self.prediction_text.grid(row=4, column=3)
       self.label = Label(self.root, text="Predicted Number is", fg="black", font=30, bg='light salmon')
       self.label.grid(row=3, column=3)
       self.model = model
       self.setup()
       self.root.mainloop()

   def setup(self):
       self.old_x = None
       self.old_y = None
       self.color = 'black'
       self.linewidth = 15
       self.c.bind('<B1-Motion>', self.paint)
       self.c.bind('<ButtonRelease-1>', self.reset)

   def paint(self, event):
       paint_color = self.color
       if self.old_x and self.old_y:
           self.c.create_line(self.old_x, self.old_y, event.x, event.y, fill=paint_color, width=self.linewidth,
                              capstyle=ROUND,
                              smooth=TRUE, splinesteps=48)
       self.old_x = event.x
       self.old_y = event.y

   def clear(self):
       """Clear drawing area"""
       self.c.delete("all")

   def reset(self, event):
       """reset old_x and old_y if the left mouse button is released"""
       self.old_x, self.old_y = None, None

   def classify(self, widget):
       x = self.root.winfo_rootx() + widget.winfo_x()
       y = self.root.winfo_rooty() + widget.winfo_y()
       x1 = widget.winfo_width()
       y1 = widget.winfo_height()
       ImageGrab.grab().crop((50, 150, 800, 800)).save('classify.png')
       def widandheight(height1, width1):
           if height1 > width1:
               divboi = height1 / 50
               height2 = height1 / divboi
               width2 = width1 / divboi
           else:
               divboi = width1 / 50
               height2 = height1 / divboi
               width2 = width1 / divboi
           return height1, width1, height2, width2
       def processthisboi(location):
           image = Image.open(location)
           # image_names = input_folder + file
           image = image.filter(ImageFilter.BLUR)
           mooo = image.convert(mode='L')
           mooo = mooo.convert(mode='1')
           thresh = 140
           fn = lambda x: 255 if x > thresh else 0
           mooo = image.convert('L').point(fn, mode='1')
           # plt.figure(figsize=(8, 4))
           # plt.imshow(image)
           # plt.show()
           return mooo
       def finalproccess(namename):
           img = load_img(namename, color_mode = "grayscale", target_size=(28, 28))
           img = img_to_array(img)
           img = img.reshape(1, 28, 28, 1)
           img = img.astype('float32')
           img = img / 255.0
           return img
       def fixthesize(mooo):
           width, height = mooo.size
           left = 20
           top = 0
           right = width - 200
           bottom = height - 20
           mooo = mooo.crop((left, top, right, bottom))
           mooo.save("mooo.jpg")
       mooo = processthisboi("classify.png")
       fixthesize(mooo)
       save_folder = '../input/'
       filename = save_folder + 'digits.png'
       image_bw = Image.open("./mooo.jpg")
       mooo = image_bw.convert(mode='L')
       mooo = mooo.convert(mode='1')
       thresh = 140  # 140
       fn = lambda x: 255 if x > thresh else 0
       mooo = image_bw.convert('L').point(fn, mode='1')
       image_bw = image_bw.convert(mode='L')
       www, hhh = image_bw.size
       height1, width1, height2, width2 = widandheight(hhh, www)
       width2 = round(width2)
       height2 = round(height2)
       image_bw = image_bw.resize((width2, height2), resample=Image.BILINEAR)
       image_bw = image_bw.resize((width1, height1), resample=Image.BILINEAR)
       image_bw = ImageEnhance.Contrast(image_bw).enhance(2)
       mooo = image_bw.convert(mode='L')
       mooo = ImageEnhance.Contrast(mooo).enhance(1.5)
       sample = mooo
       inv_sample = ImageOps.invert(sample)
       bbox = inv_sample.getbbox()
       rect = patches.Rectangle(
           (bbox[0], bbox[3]), bbox[2] - bbox[0], -bbox[3] + bbox[1] - 1,
           fill=False, alpha=1, edgecolor='w')
       width, height = sample.size
       inv_sample = ImageOps.invert(sample)
       bbox = inv_sample.getbbox()
       rect = patches.Rectangle(
           (bbox[0], bbox[3]), bbox[2] - bbox[0], -bbox[3] + bbox[1] - 1,
           fill=False, alpha=1, edgecolor='w')
       crop = inv_sample.crop(bbox)
       img = crop
       width, height = img.size
       if height > width:
           divboi = height / 17
           height = height / divboi
           width = width / divboi
       else:
           divboi = width / 17
           height = height / divboi
           width = width / divboi
       width = round(width)
       height = round(height)
       crop = img.resize((width, height), resample=Image.BILINEAR)
       new_size = 28
       delta_w = new_size - crop.size[0]
       delta_h = new_size - crop.size[1]
       padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
       new_im = ImageOps.expand(crop, padding)
       new_im = ImageEnhance.Contrast(new_im).enhance(2)
       new_im.save("new.png")
       ofname = "./new.png"
       img = finalproccess(ofname)
       # Predict digit
       pred = self.model.predict(img)
       # Get index with highest probability
       pred = np.argmax(pred)
       # print(pred)
       self.prediction_text.delete("1.0", END)
       self.prediction_text.insert(END, pred)
       labelfont = ('times', 30, 'bold')
       self.prediction_text.config(font=labelfont)


if __name__ == '__main__':
   Paint()


