# Following pip install needed - nltk, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
import math
import gradio as gr
import nltk
import os
nltk.download('punkt_tab')

# Base model
hf_token = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt"
headers =  { "Authorization": f"Bearer {hf_token}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


# Fine tuned model - Spanish
model_path_es = "agodbole/fine_tuned_model_es"

fine_tuned_tokenizer_es = AutoTokenizer.from_pretrained(model_path_es)
fine_tuned_model_es = AutoModelForSeq2SeqLM.from_pretrained(model_path_es)

# Fine tuned model - Hindi
model_path_hi = "agodbole/fine_tuned_model_hi"

fine_tuned_tokenizer_hi = AutoTokenizer.from_pretrained(model_path_hi)
fine_tuned_model_hi = AutoModelForSeq2SeqLM.from_pretrained(model_path_hi)


# Function to evaluate predicted labels
def evaluate(sentence):
    predicted_gender = ''
   
    tokens = nltk.word_tokenize(sentence.lower())

    if 'he' in tokens and 'she' in tokens:
        label = str(input("Enter whether - male or female or neutral : ")) #happens rarely
        predicted_gender = label

    elif 'she' in tokens:
        predicted_gender = 'female'

    elif 'he' in tokens:
        predicted_gender = 'male'

    elif 'his' in tokens or 'him' in tokens:
        if 'her' in tokens:
            label = str(input("Enter whether - male or female : ")) #happens rarely
            predicted_gender = label

        else:
            predicted_gender = 'male'

    elif 'her' in tokens:
        predicted_gender = 'female'

    else:
        predicted_gender = 'neutral'

    return predicted_gender


# Define a gender prediction function - Base model output
def gender_prediction_base(sentence, src_lang):
    try:
        output = query({
	        "inputs": sentence,
            "parameters": {
                "src_lang": src_lang,
                "tgt_lang": "en_XX"
            }
        })
    
        translation = output[0]['translation_text']
        predicted_gender = evaluate(translation)

        result = {"translation": translation, "predicted_gender": predicted_gender}
        return result
    
    except:
        return 'Error occurred. Please try again'


# Define a gender prediction function - Fine tuned output - Spanish
def gender_prediction_ft_es(sentence):
    try:
        inputs = fine_tuned_tokenizer_es(sentence, return_tensors="pt", padding=True, truncation=True)
    
        with torch.no_grad():
            generated_ids = fine_tuned_model_es.generate(
                inputs["input_ids"],
                forced_bos_token_id=fine_tuned_tokenizer_es.lang_code_to_id["en_XX"],  # Specify the target language
            )

        translation = fine_tuned_tokenizer_es.batch_decode(generated_ids, skip_special_tokens=True)
        predicted_gender = evaluate(translation[0])

        result = {"translation": translation[0], "predicted_gender": predicted_gender}
        return result

    except:
        return 'Error occurred. Please try again'

# Define a gender prediction function - Fine tuned output - Hindi
def gender_prediction_ft_hi(sentence):
    try:
        inputs = fine_tuned_tokenizer_hi(sentence, return_tensors="pt", padding=True, truncation=True)
    
        with torch.no_grad():
            generated_ids = fine_tuned_model_hi.generate(
                inputs["input_ids"],
                forced_bos_token_id=fine_tuned_tokenizer_hi.lang_code_to_id["en_XX"],  # Specify the target language
            )

        translation = fine_tuned_tokenizer_hi.batch_decode(generated_ids, skip_special_tokens=True)
        predicted_gender = evaluate(translation[0])

        result = {"translation": translation[0], "predicted_gender": predicted_gender}
        return result

    except:
        return 'Error occurred. Please try again'

def compare_models_es(sentence_es):
    try:
        # Get predictions from both models
        output_base = gender_prediction_base(sentence_es, 'es_XX')
        output_ft = gender_prediction_ft_es(sentence_es)
    
        # Return results in a format that can be displayed side by side
        return output_base["translation"] + f" [{output_base['predicted_gender']}]", output_ft["translation"] + f" [{output_ft['predicted_gender']}]"

    except:
        return 'Error occurred. Please try again', 'Error occurred. Please try again'

def compare_models_hi(sentence_hi):
    try:
        # Get predictions from both models
        output_base = gender_prediction_base(sentence_hi, 'hi_IN')
        output_ft = gender_prediction_ft_hi(sentence_hi)
    
        # Return results in a format that can be displayed side by side
        return output_base["translation"] + f" [{output_base['predicted_gender']}]", output_ft["translation"] + f" [{output_ft['predicted_gender']}]"

    except:
        return 'Error occurred. Please try again', 'Error occurred. Please try again'

# Create a Blocks layout to add the interfaces to the same app
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            title = gr.HTML("<h1>Spanish</h1>")
            inputs = gr.Textbox(label="Input Sentence")
            btn = gr.Button(value="Submit")
            output_base = gr.Textbox(label="Base Model Translation [Gender Prediction Accuracy 71%]") 
            output_ft = gr.Textbox(label="Fine tuned Model Translation [Gender Prediction Accuracy 84%]")
            
            btn.click(compare_models_es, inputs=[inputs], outputs=[output_base, output_ft])
    
        with gr.Column(scale=1, min_width=300):
            title = gr.HTML("<h1>Hindi</h1>")
            inputs = gr.Textbox(label="Input Sentence")
            btn = gr.Button(value="Submit")
            output_base = gr.Textbox(label="Base Model Translation [Gender Prediction Accuracy 45%]") 
            output_ft = gr.Textbox(label="Fine tuned Model Translation [Gender Prediction Accuracy 53%]")
            
            btn.click(compare_models_hi, inputs=[inputs], outputs=[output_base, output_ft])

# Launch the app
demo.launch(share=False)