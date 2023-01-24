import speech_recognition as sr
import openai
from playsound import playsound
from gtts import gTTS
import cv2
import pytesseract
import gradio as gr


openai.api_key = "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"

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: "

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
    output = openai_create(inp)
    history.append((input, output))
    mytext = output
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("welcome.mp3")
    playsound('C:\\Users\\abboj\\Downloads\\Hackathon-main\\Hackathon-main\\welcome.mp3')
    return history, history

def voice():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting noise ")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Recording for 4 seconds")
        recorded_audio = recognizer.listen(source, timeout=4)
        print("Done recording")

    try:
        print("Recognizing the text")
        text = recognizer.recognize_google(
                recorded_audio, 
                language="en-US"
            )
    except Exception as ex:
        print(ex)
    return text

def imagetotext(inp):
    img = cv2.imread(inp)
    config = ('-l eng --oem 1 --psm 3')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, threshimg = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    dilation = cv2.dilate(threshimg, rect_kernel, iterations = 1) 
    img_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                        cv2.CHAIN_APPROX_NONE) 
    maintext = ''
    for cnt in img_contours: 
        x, y, w, h = cv2.boundingRect(cnt) 

        rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        cropped_img = img[y:y + h, x:x + w] 
 
        file = open("recognized.txt", "w") 
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        text = pytesseract.image_to_string(cropped_img) 

        maintext = maintext+text
    return maintext

block = gr.Blocks()


with block:
    gr.Markdown("""<h1><center>Alpha Q</center></h1>
    """)
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt)
    state = gr.State()
    imagef = gr.Image(type="filepath")
    btnf = gr.Button("Upload Query")
    btnf.click(imagetotext, inputs=imagef,outputs=message)
    voicef = gr.Button("VOICE Query")
    voicef.click(voice, outputs=message)
    submit = gr.Button("EXECUTE")
    submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])
    

block.launch(debug = True)
