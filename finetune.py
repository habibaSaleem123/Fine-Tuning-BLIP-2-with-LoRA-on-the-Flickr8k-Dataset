import torch
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from PIL import Image

# Path to your custom dataset
images_folder = "/Users/habibasaleem/Desktop/Q2_A6/flickr8k/images"  # Update this path
captions_file = "/Users/habibasaleem/Desktop/Q2_A6/flickr8k/captions.txt"  # Update this path

# Custom dataset class
class Flickr8kDataset(Dataset):
    def __init__(self, images_folder, captions_file, processor):
        self.images_folder = images_folder
        self.processor = processor
        self.image_caption_pairs = []

        # Read captions from the file
        with open(captions_file, "r") as f:
            for line in f:
                if line.strip():
                    filename, caption = line.strip().split(",", 1)
                    image_path = os.path.join(images_folder, filename)
                    if os.path.exists(image_path):
                        self.image_caption_pairs.append({
                            "image_path": image_path,
                            "caption": caption
                        })

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        example = self.image_caption_pairs[idx]
        image = Image.open(example["image_path"]).convert('RGB')
        caption = example["caption"]

        # Preprocess image and caption
        inputs = self.processor(images=image, text=caption, return_tensors="pt")
        return inputs

# Initialize processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto", torch_dtype=torch.float16)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qformer.query_tokens", "language_model"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare dataset
train_dataset = Flickr8kDataset(images_folder, captions_file, processor)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3  # Adjust this based on your requirements

for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        # Move inputs to device (GPU or CPU)
        inputs = {key: value.to(device) for key, value in batch.items()}

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
    print(f"Epoch {epoch+1}/{epochs} finished with loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_blip2")
