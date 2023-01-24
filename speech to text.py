import speech_recognition as sr
import openai
from playsound import playsound
from gtts import gTTS
import cv2
import pytesseract

openai.api_key = "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"

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

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
while(True):
    choiceType = int(input("Enter the choice\n1: Entering text\n2: Voice Query\n3: Image Query\n4: Exit\n"))


    if(choiceType==1):
        inp = input("Enter the query\n")
        output = openai_create(inp)
        
    elif(choiceType==2):
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

        output = openai_create(text)

    elif(choiceType==3):
        imgpath = input("Enter the image path\n")
        img = cv2.imread(imgpath)
        # set configurations
        config = ('-l eng --oem 1 --psm 3')
        # Convert the image to gray scale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # OTSU threshold performing
        ret, threshimg = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
        
        # Specifying kernel size and structure shape.  
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
        
        # Appplying dilation on the threshold image 
        dilation = cv2.dilate(threshimg, rect_kernel, iterations = 1) 
        
        # getting contours 
        img_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                        cv2.CHAIN_APPROX_NONE) 
        
        # Loop over contours and crop and extract the text file
        for cnt in img_contours: 
            x, y, w, h = cv2.boundingRect(cnt) 
            
            # Drawing a rectangle
            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            
            # Cropping the text block  
            cropped_img = img[y:y + h, x:x + w] 
            
            # Open the text file in write mode 
            file = open("recognized.txt", "w") 
            
            # Applying tesseract OCR on the cropped image 
            pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
            text = pytesseract.image_to_string(cropped_img) 
            
            # Appending the text into file 
            file.write(text) 
            file.write("\n") 
            file.close 
            file = open("recognized.txt", "r") 
            data = file.read()
            output = openai_create(data)
            # Close the file 
            file.close 
    else:
        break
    print("Answer: ",output)
    print()
    choiceOut = input("Do you want our app to speak it out Y/N\n")
    if(choiceOut=='Y'):
        myobj = gTTS(text=output, lang='en', slow=False)
        myobj.save("C:\\Users\\abboj\\Downloads\\Hackathon-main\\Hackathon-main\\welcome.mp3")
        playsound('C:\\Users\\abboj\\Downloads\\Hackathon-main\\Hackathon-main\\welcome.mp3')
