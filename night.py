import os
import openai
import gradio as gr
from gtts import gTTS
import speech_recognition as sr


openai.api_key = "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"

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

prompt = "The following is a conversation with Alpha Q. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by Bug Busters. How can I help you today?\nHuman: "

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

def speech_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting noise ")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Recording for 4 seconds")
        recorded_audio = recognizer.listen(source, timeout=4)
        print("Done recording")

    try:
        print("Recognizing the text")
        texts = recognizer.recognize_google(
                recorded_audio, 
                language="en-US"
            )
        print(texts)

    except Exception as ex:

        print(ex)

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
    os.system("mpg321 welcome.mp3")
    return history, history


block = gr.Blocks()


with block:
    gr.Markdown("""<h1><center>Alpha Q</center></h1>
    """)
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt)
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])
    with gr.Row():
        voice=gr.Button("Voice input")
        voice.click(speech_text,inputs=[message, state], outputs=[chatbot, state])
        upload=gr.Button("Upload Image")

block.launch(debug = True)
