import os
import speech_recognition as sr
import openai
from playsound import playsound
from gtts import gTTS
import cv2
import pytesseract
import gradio as gr
import re

openai.api_key = "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"

#start_sequence = "\nAI:"
#restart_sequence = "\nHuman: "

#prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: "

def openai_create(prompt):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )

    return response.choices[0].text



def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    if('draw' in input[0:4]):
        input = input.replace('draw','',1)
        response = openai.Image.create(
        prompt=input,
        n=1
        )
        image_url = response['data'][0]['url']
        output= image_url
    else:
        output = openai_create(inp)
    history.append((input, output))
    return history, history, ''

def voice():
    recognizer = sr.Recognizer()
    text=''
    with sr.Microphone() as source:
        print("Adjusting noise ")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Recording for 4 seconds")
        recorded_audio = recognizer.listen(source, timeout=5)
        print("Done recording")

    try:
        print("Recognizing the text")
        text = recognizer.recognize_google(recorded_audio, language="en-US")
    except Exception as ex:
        print(ex)
    return text

def outvoice(history):
    history = history or []
    if history:
        language = 'en'
        regex = re.compile(r'<[^>]+>')
        def remove_html(string):
            return regex.sub('', string)
        new_text=remove_html(history[-1][1])
        myobj = gTTS(text=new_text, lang=language, slow=False)
        myobj.save("welcome.mp3")
        playsound('welcome.mp3')
        os.remove('welcome.mp3')
    else:
        language = 'en'
        myobj = gTTS(text="No data present", lang=language, slow=False)
        myobj.save("welcome.mp3")
        playsound('welcome.mp3')
        os.remove('welcome.mp3')
    


def imagetotext(inp):
    img = cv2.imread(inp)
    config = ('-l eng --oem 1 --psm 3')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    maintext = pytesseract.image_to_string(gray) 
    return maintext

block = gr.Blocks()


with block:
    gr.Markdown("""<h1><font color='orange'><center>AlphaBot</center></font></h1>
    """)
    chatbot = gr.Chatbot()
    
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1, min_width=850):
            message = gr.Textbox(placeholder="Enter your query here")
        with gr.Column(min_width=1):
            submit = gr.Button("➡️")
            submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state,message])  
            voicef = gr.Button("🎙️") 
            voicef.click(voice, outputs=message)
        with gr.Column(min_width=1):
            btnf = gr.Button("📸")
            btnp = gr.Button("🔊")
            btnp.click(outvoice,inputs=state)

    imagef = gr.Image(type="filepath")
    btnf.click(imagetotext, inputs=gr.Image(type="filepath"),outputs=[message])
    
block.launch(debug = True)