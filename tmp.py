import gradio as gr
import speech_recognition as sr

def my_input_func(text_input, voice_input):
    return f"Text Input: {text_input}, Voice Input: {voice_input}"

def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

inputs = [gr.inputs.Textbox(label="Text Input"), 
          gr.inputs.Func(label="Voice Input", func=voice_input)]
output = gr.outputs.Textbox()

gr.Interface(my_input_func, inputs, output, layout="column").launch()
