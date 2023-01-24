import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Combobox
import pyttsx3
import os

root = Tk()
root.title("Text to speech")
root.geometry('1200x650')
root.resizable(False, False)
root.configure(bg="#305065")

engine=pyttsx3.init()
def speaknow():
    text = text_area.get(1.0,END)
    gender=gender_combobox.get()
    speed=speed_combobox.get()
    voices=engine.getProperty("voices")
    def setvoice():
        if (gender == "Male"):
            engine.setProperty("voice",voices[0].id)
            engine.say(text)
            engine.runAndWait()
        else:
            engine.setProperty("voice", voices[1].id)
            engine.say(text)
            engine.runAndWait()
    if (text):
        if (speed=="Fast"):
            engine.setProperty("rate",250)
            setvoice()
        elif (speed=="Normal"):
            engine.setProperty("rate", 150)
            setvoice()
        else:
            engine.setProperty("rate", 60)
            setvoice()



def download():
    text = text_area.get(1.0, END)
    gender = gender_combobox.get()
    speed = speed_combobox.get()
    voices = engine.getProperty("voices")

    def setvoice():
        if (gender == "Male"):
            engine.setProperty("voice", voices[0].id)
            path=filedialog.askdirectory()
            os.chdir(path)
            engine.save_to_file(text,'text.mp3')
            engine.say(text)
            engine.runAndWait()
        else:
            engine.setProperty("voice", voices[1].id)
            path = filedialog.askdirectory()
            os.chdir(path)
            engine.save_to_file(text,'text.mp3')
            engine.runAndWait()

    if (text):
        if (speed == "Fast"):
            engine.setProperty("rate", 250)
            setvoice()
        elif (speed == "Normal"):
            engine.setProperty("rate", 150)
            setvoice()
        else:
            engine.setProperty("rate", 60)
            setvoice()







#icon
image_icon=PhotoImage(file="D://speak.png")
root.iconphoto(False,image_icon)
#Top Frame
Top_frame=Frame(root,bg="white",width=1200,height=200)
Top_frame.place(x=0,y=0)
Logo=PhotoImage(file="D://speaker logo.png")
Label(Top_frame,image=Logo,bg="white").place(x=10,y=5)
Label(Top_frame,text="TEXT TO SPEECH",font="arial 20 bold",bg="white",fg="black").place(x=300,y=50)


#################
text_area=Text(root,font="Robote 20",bg="white",relief=GROOVE,wrap=WORD)
text_area.place(x=10,y=250,width=500,height=250)
Label(root,text="VOICE",font="arial 15 bold",bg="#305065",fg="white").place(x=580,y=260)
Label(root,text="SPEED",font="arial 15 bold",bg="#305065",fg="white").place(x=750,y=260)
gender_combobox=Combobox(root,values=["Male","Female"],font="arial 14",state="r",width=10)
gender_combobox.place(x=550,y=300)
gender_combobox.set("Male")
speed_combobox=Combobox(root,values=["Fast","Normal","Slow"],font="arial 14",state="r",width=10)
speed_combobox.place(x=750,y=300)
speed_combobox.set('Normal')
imageicon=PhotoImage(file="D://speak.png")
btn=Button(root,text="Speak",compound=LEFT,image=imageicon,width=150,font="arial 14 bold",command=speaknow)
btn.place(x=550,y=380)
imageicon2=PhotoImage(file="D://download.png")
save=Button(root,text="Download",compound=LEFT,image=imageicon2,width=150,bg="#39c790",font="arial 14 bold",command=download)
save.place(x=750,y=380)


root.mainloop()
