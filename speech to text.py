import speech_recognition as sr
r = sr.Recognizer()

mic = sr.Microphone()

print('start')
with mic as source:
    audio = r.listen(source)
print('end')
print(r.recognize_google(audio))
