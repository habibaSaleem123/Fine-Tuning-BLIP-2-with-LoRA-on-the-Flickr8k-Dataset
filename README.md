# Fine-Tuning BLIP-2 with LoRA on the Flickr8k Dataset

## Project Overview

This project demonstrates the process of fine-tuning the **BLIP-2** model, a state-of-the-art vision-language model, using the **Flickr8k dataset** for **image captioning**. The fine-tuning is made more efficient by utilizing **LoRA** (Low-Rank Adaptation), a technique that reduces memory usage and accelerates training by modifying only a small portion of the model's parameters.

---

## Project Description

The goal of this project is to enable BLIP-2 to generate accurate and relevant captions for images from the **Flickr8k dataset**, which contains 8,000 images, each paired with five human-written captions. The project focuses on:
- Fine-tuning BLIP-2 for the specific task of image captioning.
- Using LoRA to optimize memory and computational efficiency during the fine-tuning process.
- Leveraging pre-trained BLIP-2 (via the `Salesforce/blip2-flan-t5-xl` checkpoint) for improved performance on the image captioning task.

This project is suitable for:
- Researchers or practitioners who want to fine-tune BLIP-2 for custom image captioning tasks.
- Developers seeking to optimize model training with LoRA to reduce hardware requirements.
- Anyone interested in leveraging advanced vision-language models for practical multimodal applications.

---

## Key Features

- **BLIP-2 for Image Captioning**: Fine-tuning BLIP-2 on the Flickr8k dataset to generate image captions based on visual input.
- **LoRA Integration**: Memory-efficient training with Low-Rank Adaptation (LoRA), which only fine-tunes small, low-rank adapters instead of the entire model.
- **Flickr8k Dataset**: The dataset consists of 8,000 images paired with five human-written captions per image, ideal for training image captioning models.
- **Evaluation Metrics**: Use of BLEU, METEOR, CIDEr, and ROUGE for evaluating the performance of the fine-tuned model.

---

## Installation

To run this project, you will need to set up the following dependencies:

### Clone the repository
```bash
git clone https://github.com/your-username/fine-tuning-blip2-lora.git
cd fine-tuning-blip2-lora
