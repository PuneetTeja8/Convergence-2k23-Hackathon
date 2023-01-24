with block:
    gr.Markdown("""<h1><center>Alpha Q</center></h1>
    """)
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt)
    state = gr.State()
    with gr.Row():
        submit = gr.Button("EXECUTE")
        submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])
        voicef = gr.Button("VOICE Query")
        voicef.click(voice, outputs=message)
    imagef = gr.Image(type="filepath")
    btnf = gr.Button("Upload Query")
    btnf.click(imagetotext, inputs=imagef,outputs=message)
