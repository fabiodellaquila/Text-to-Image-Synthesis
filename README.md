# Generative Synthesis of Pokémon Sprites from Textual Descriptions

It is a deep learning model that generates 2D Pokémon sprites directly from Pokédex text descriptions. The model leverages a pre-trained **BERT-mini** text encoder and an **Image Decoder with Cross-Attention** to map natural language descriptions into 215×215 RGB Pokémon sprites.

---

## Architecture Overview

- **Text Encoder:** Pre-trained `bert-mini` (`prajjwal1/bert-mini`) converts textual descriptions into contextual embeddings.
- **Cross-Attention Mechanism:** Multi-head cross-attention layers at multiple decoder stages enable visual feature maps to selectively focus on relevant text tokens.
- **Image Decoder:** Transposed convolutional network progressively upsampling a combined latent noise vector and text representation into a 215×215 image.
- **Loss Function:** Combined **L1 Pixel Loss** and **Perceptual Loss (VGG19 Content & Style/Gram Matrix)**, evaluated with **SSIM (Structural Similarity Index Measure)**.

---

## Repository Structure

```
├── dataset/
│   ├── pokemon.csv           # Text descriptions and Pokémon numbers
│   └── small_images/         # Folder for Pokémon sprite images
├── model/
│   └── pokemon_generator.pth # Saved model weights
├── demo.py                   # Gradio web interface for inference
├── main.py                   # Dataset loader, model architecture, and training pipeline
├── Report.pdf                # Detailed technical report
└── README.md
```

---

## Dataset Setup

1. The metadata file `pokemon.csv` is already provided in the `dataset/` directory.
2. The sprite images must be downloaded separately from:
   - [https://github.com/cristobalmitchell/pokedex/](https://github.com/cristobalmitchell/pokedex/)
3. Place the sprite images into the `dataset/small_images/` folder, formatted as 3-digit PNG files (e.g., `001.png`, `025.png`).

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/fabiodellaquila/Text-to-Image-Synthesis.git](https://github.com/fabiodellaquila/Text-to-Image-Synthesis.git)
cd Text-to-Image-Synthesis
pip install torch torchvision transformers gradio pytorch-msssim pillow pandas
```

---

## Usage

### 1. Train the Model
To start or resume the training pipeline:

```bash
python main.py
```
- Training runs by default for 300 epochs using AdamW.
- Periodically saves generated sample images to `generated_images/`.
- Saves the trained model weights to `model/pokemon_generator_new.pth`.

### 2. Run the Interactive Web UI (Gradio)
To launch the user-friendly Gradio interface for generating Pokémon sprites from custom text descriptions:

```bash
python demo.py
```
Open the local URL displayed in your terminal to interact with the generator.
