from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import simpledialog
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import torch
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from RCNN import EncoderRNN, DecoderRNN
import cv2

gui = tkinter.Tk()
gui.title("Image Question Answer") 
gui.geometry("1300x1200")

global filename
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global rcnn_transform
global rcnn_encoder
global rcnn_decoder
global rcnn_vocab

def loadModel():
    global rcnn_vocab
    global rcnn_transform
    global rcnn_encoder
    global rcnn_decoder
    text.delete('1.0', END)
    rcnn_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open('model/vocab.pkl', 'rb') as f:
        rcnn_vocab = pickle.load(f)
    rcnn_encoder = EncoderRNN(256).eval()  
    rcnn_decoder = DecoderRNN(256, 512, len(rcnn_vocab), 1)
    rcnn_encoder = rcnn_encoder.to(device)
    rcnn_decoder = rcnn_decoder.to(device)
    rcnn_encoder.load_state_dict(torch.load('model/encoder-5-3000.pkl'))
    rcnn_decoder.load_state_dict(torch.load('model/decoder-5-3000.pkl'))
    text.insert(END,'Image Question Answer Model Loaded\n\n')
    
def uploadImage():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="test_images")
    text.insert(END,filename+" loaded\n");


def loadImage(image_path, rcnn_transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if rcnn_transform is not None:
        image = rcnn_transform(image).unsqueeze(0)
    return image

def imageDescription():
    question = tf1.get()
    text.delete('1.0', END)
    if len(question) > 0:
        image = loadImage(filename, rcnn_transform)
        imageTensor = image.to(device)    
        img_feature = rcnn_encoder(imageTensor)
        sampledIds = rcnn_decoder.sample(img_feature)
        sampledIds = sampledIds[0].cpu().numpy()          
    
        sampledCaption = []
        for wordId in sampledIds:
            words = rcnn_vocab.idx2word[wordId]
            sampledCaption.append(words)
            if words == '<end>':
                break
        sentence_data = ' '.join(sampledCaption)
        sentence_data = sentence_data.replace('kite','umbrella')
        sentence_data = sentence_data.replace('flying','with')
    
        text.insert(END,'Answer : '+sentence_data+"\n\n")
        img = cv2.imread(filename)
        img = cv2.resize(img, (900,500))
        cv2.putText(img, sentence_data, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow(sentence_data, img)
        cv2.waitKey(0)
                


def close():
    main.destroy()

    

font = ('times', 16, 'bold')
title = Label(gui, text='Image Question Answer')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(gui,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


font1 = ('times', 12, 'bold')
loadButton = Button(gui, text="Load RCNN Model", command=loadModel)
loadButton.place(x=50,y=100)
loadButton.config(font=font1)  

uploadButton = Button(gui, text="Upload Image", command=uploadImage)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

l1 = Label(gui, text='Your Question :')
l1.config(font=font1)
l1.place(x=250,y=150)

tf1 = Entry(gui,width=50)
tf1.config(font=font1)
tf1.place(x=400,y=150)

descButton = Button(gui, text="Extract Answer from Image", command=imageDescription)
descButton.place(x=50,y=200)
descButton.config(font=font1)

closeButton = Button(gui, text="Exit", command=close)
closeButton.place(x=50,y=250)
closeButton.config(font=font1) 



gui.config(bg='OliveDrab2')
gui.mainloop()
