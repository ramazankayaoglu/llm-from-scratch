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

def chat_with_model(message, chat_history, max_new_tokens=20):
    try:
        tokens = tokenizer.encode(message)

        if len(tokens) > 25:
            tokens = tokens[-25:]

        with torch.no_grad():
            actual_max_new_tokens = min(
                max_new_tokens,
                32 - len(tokens)
            )

            generated_tokens = model.generate(
                tokens,
                max_new_tokens=actual_max_new_tokens
            )

        response = tokenizer.decode(generated_tokens)

        original_message = tokenizer.decode(tokens.tolist())

        if response.startswith(original_message):
            response = response[len(original_message):]

        response = (
            response
            .replace("<pad>", "")
            .replace("<unk>", "")
            .strip()
        )

        if not response:
            response = (
                "I am sorry, I don't know the answer "
                "to that question."
            )

        # Yeni Gradio messages formatı
        chat_history.append({
            "role": "user",
            "content": message
        })

        chat_history.append({
            "role": "assistant",
            "content": response
        })

        return chat_history, ""

    except Exception as error:
        print(f"Error : {error}")
        return chat_history, ""


def load_model_from_url(custom_model_url):
    global model, tokenizer, model_status
    try:        
        import requests
        headers = {
            "Accept" : "application/octet-stream",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        response = requests.get(custom_model_url, headers = headers)
        response.raise_for_status()

        temp_file = "temp_model.pth"
        with open(temp_file, "wb") as f:
            f.write(response.content)

        model, tokenizer, model_status = load_model(temp_file)
        os.remove(temp_file)
        return "Model loaded succesfully from URL"
    except Exception as e:
        print(f"Error loading model from URL: {e}")
        return "Error loading model from URL"

def load_model_from_file(model_file):
    global model, tokenizer, model_status
    try:
        print(f"Loading model from file {model_file.name}")
        model, tokenizer, model_status = load_model(model_file)
        return "Model loaded succesfully from file"
    except Exception as e:
        print(f"Error loading model from file: {e}")
        return "Error loading model from file"


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
        custom_model_url = gr.Textbox(placeholder = "Enter the path to the custom model...", label = "Custom model path", scale = 4)
        load_url_button = gr.Button("Load Model",variant = "primary", scale = 1)


    with gr.Row():
        model_file = gr.File(label = "Model file", file_types = [".pth", ".pt", ".bin"])
        load_file_button = gr.Button("Load Model", variant = "primary")

    status = gr.Textbox(label = "Model Status", value = model_status, interactive = False)

    def send_message(message, chat_history, max_new_tokens):
        if not message.strip():
            return chat_history, ""

        return chat_with_model(message, chat_history, max_new_tokens)        


    send_button.click(
        send_message,
        inputs = [msg, chatbot, max_new_tokens],
        outputs = [chatbot, msg]
    )

    msg.submit(
        send_message,
        inputs= [msg, chatbot, max_new_tokens],
        outputs = [chatbot, msg]
    )

    clear_button.click(lambda : None, None, chatbot, status)


    load_url_button.click(
        load_model_from_url,
        inputs = [custom_model_url],
        outputs = [status]
    )

    load_file_button.click(
        load_model_from_file,
        inputs = [model_file],
        outputs = [status]
    )


if __name__ == "__main__":
    demo.launch(share = True)