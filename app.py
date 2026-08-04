import os 

import gradio as gr
import torch

from v1.master_model import MasterModel
from v1.master_tokenizer import MasterTokenizer

model, tokenizer, model_status = None, None, "Not loaded"

def load_model(custom_model_path = None):
    try:
        master_tokenizer = MasterTokenizer("v1/tokenizer.json")
        print(f"Tokenizer loaded succesfully, vocab size: {len(master_tokenizer.vocab)}")
        context_length = 32
        vocab_size = len(master_tokenizer.vocab)
        embedding_dimension = 12
        number_layers = 8
        number_heads = 4

        model = MasterModel(
            context_length = context_length,
            vocab_size = vocab_size,
            embedding_dim = embedding_dimension,
            num_heads = number_heads,
            num_layers = number_layers)
        if custom_model_path and os.path.exists(custom_model_path):
            model.load_state_dict(torch.load(custom_model_path))
        else:
            model.load_state_dict(torch.load("v1/master_model.pth"))
        model.eval()
        print(f"Model loaded succesfully, vocab size: {len(master_tokenizer.vocab)}")
        return model, master_tokenizer, "Model loaded succesfully"

    except Exception as e: 
        print(f"Error loading model: {e}")
        return None, None, "Error loading model"



try:
    model, tokenizer, model_Status = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model, tokenizer, model_status = None, None, "Error loading model"

print(f"Model status: {model_status}")

if model is not None:
    print("Model loaded succesfully")


with gr.Blocks(title = "Master Model") as demo:
    gr.Markdown("## Master Model")
    gr.Markdown("Chat with a custom model transformer language model built from scratch! This model specializes in greographical knowledge.")

    chatbot = gr.Chatbot(height = 400)
    msg = gr.Textbox(placeholder = "Enter your message here...", label = "Message")

    with gr.Row():
        send_button = gr.Button("Send",variant = "primary")
        clear_button = gr.Button("Clear",variant = "secondary")
    
    max_new_tokens = gr.Slider(minimum = 1, maximum = 30, value = 20, step = 1, label = "Max new tokens", info = "The maximum number of new tokens to generate")


    gr.Markdown("## Load a custom model")
    with gr.Row():
        custom_model_path = gr.Textbox(placeholder = "Enter the path to the custom model...", label = "Custom model path")
        load_button = gr.Button("Load Model",variant = "primary")


if __name__ == "__main__":
    demo.launch(share = True)