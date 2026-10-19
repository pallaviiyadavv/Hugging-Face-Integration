import gradio as gr
from transformers import pipeline
classifier = pipeline("sentiment-analysis")


def function(emotion):
    result = classifier(emotion)
    fb_type = result[0]["label"]
    score = result[0]["score"]
    return f"Feedback Type: {fb_type}, Score: {score}"
    
idfc = gr.Interface(fn=function, inputs="text", outputs="text")
idfc.launch(share = True)

