import torch
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
import gradio as gr
from gtts import gTTS
import tempfile
import os

# Load the fine-tuned model
processor = BlipProcessor.from_pretrained("./fine_tuned_blip2")
model = Blip2ForConditionalGeneration.from_pretrained("./fine_tuned_blip2", device_map="auto", torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

class TextToSpeech:
    def __init__(self):
        self.temp_file = "temp_tts.mp3"

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(self.temp_file)
            return self.temp_file
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None

    def play_audio(self, audio_path):
        if audio_path and os.path.exists(audio_path):
            return audio_path
        return None

class ImageQA:
    def __init__(self):
        self.processor = processor
        self.model = model

    def answer_question(self, image_path, question):
        try:
            raw_image = Image.open(image_path).convert('RGB')

            inputs = self.processor(raw_image, question, return_tensors="pt").to(
                "cuda" if torch.cuda.is_available() else "cpu",
                torch.float16
            )

            out = self.model.generate(**inputs)
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            print(f"Error in QA: {e}")
            return "Sorry, I couldn't process that question."

# Initialize ImageQA and TextToSpeech
qa = ImageQA()
tts = TextToSpeech()

def process_query(audio, image, text_question):
    question = ""

    # Process audio if provided
    if audio is not None:
        question = stt.transcribe(audio)  # Assume `stt` is your Speech-to-Text processing

    if not question and text_question:
        question = text_question

    if not question:
        return "Please ask a question (voice or text)", None

    if image is None:
        return "Please upload an image", None

    # Save image to temp file
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_img.name)

    # Get answer from fine-tuned VLM
    answer = qa.answer_question(temp_img.name, question)
    os.unlink(temp_img.name)  # Clean up

    # Generate TTS
    tts_path = tts.speak(answer)
    return answer, tts_path

# Setup Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### 🖼️ Ask-the-Image (Fine-tuned Model)")
    gr.Markdown("Upload an image and ask questions about it (voice or text)")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Record your question (10 sec max)", type="numpy")
            text_input = gr.Textbox(label="Or type your question here")
            image_input = gr.Image(label="Upload an image", type="pil")
            submit_btn = gr.Button("Ask")

        with gr.Column():
            answer_output = gr.Textbox(label="Answer", interactive=False)
            audio_output = gr.Audio(label="Spoken Answer", visible=True)

    submit_btn.click(
        fn=process_query,
        inputs=[audio_input, image_input, text_input],
        outputs=[answer_output, audio_output]
    )

# Launch Gradio interface
demo.launch(debug=True, share=True)
