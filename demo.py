import gradio as gr # Import the Gradio library for creating web UIs
import numpy as np
from main import *

# --- Model Configuration and Loading ---
MODEL_PATH = 'model/pokemon_generator.pth'  # Define the path to the pre-trained model file
MAX_LEN = 128  # Set the maximum sequence length for text encoding, must match training configuration
IMAGE_OUTPUT_SIZE = 215  # Define the desired output size for generated images, must match training configuration
ENCODER_HIDDEN_DIM = 256 # Define the hidden dimension for the encoder part of the model
LATENT_VECTOR_DIM = 100 # Define the dimension of the latent vector, which connects encoder and decoder
DECODER_BASE_FILTERS = 1024 # Define the base number of filters for the decoder's convolutional layers
IMG_CHANNELS = 3 # Define the number of channels for the output image (e.g., 3 for RGB)
NUM_ATTENTION_HEADS= 4 # Define the number of attention heads for multi-head attention mechanisms in the model

# Determine the device to run the model on (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained BERT tokenizer for text processing
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-mini')

# Initialize the PokemonGenerator model with specified architectural parameters
model = PokemonGenerator(
    bert_model_name='prajjwal1/bert-mini', encoder_hidden_dim=ENCODER_HIDDEN_DIM, latent_dim=LATENT_VECTOR_DIM,
    decoder_base_filters=DECODER_BASE_FILTERS, img_channels=IMG_CHANNELS, output_size=IMAGE_OUTPUT_SIZE, n_heads=NUM_ATTENTION_HEADS)

# Load the saved state dictionary into the model, mapping to the determined device
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Move the model to the specified device (CPU or GPU)
model.to(device)
# Set the model to evaluation mode
model.eval()


# --- Inference Function for Gradio ---
def generate_pokemon_sprite(description: str) -> Image.Image:
    """
    Generates a Pokémon sprite from a given text description.

    Args:
        description (str): A text description of the Pokémon.

    Returns:
        Image.Image: The generated Pokémon sprite as a PIL Image object.
    """
    # If the description is empty or only whitespace, return a blank white image
    if not description.strip():
        return Image.new('RGB', (IMAGE_OUTPUT_SIZE, IMAGE_OUTPUT_SIZE), color='white')

    # Encode the input description using the BERT tokenizer
    # It pads to MAX_LEN, truncates if longer, and returns PyTorch tensors
    encoded = tokenizer(description, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')

    # Extract input IDs (tokenized text) and attention mask from the encoded output
    tokens = encoded['input_ids'].to(device) # Move tokens to the appropriate device
    attention_mask = encoded['attention_mask'].to(device) # Move attention mask to the appropriate device

    # Disable gradient calculation for inference to save memory and speed up computation
    with torch.no_grad():
        # Pass the tokens and attention mask to the model to generate an image tensor
        generated_image_tensor = model(tokens, attention_mask)

    # Post-process the generated image tensor:
    # 1. Remove the batch dimension (squeeze(0))
    # 2. Move the tensor from GPU to CPU (.cpu())
    # 3. Convert the tensor to a NumPy array (.numpy())
    generated_image_tensor = generated_image_tensor.squeeze(0).cpu().numpy()
    # Rescale pixel values from [-1, 1] to [0, 1]
    generated_image_tensor = (generated_image_tensor + 1) / 2
    # Transpose the dimensions from (C, H, W) to (H, W, C) for image display
    generated_image_tensor = generated_image_tensor.transpose(1, 2, 0)
    # Scale pixel values from [0, 1] to [0, 255] and convert to unsigned 8-bit integers
    generated_image_numpy = (generated_image_tensor * 255).astype(np.uint8)

    # Convert the NumPy array to a PIL Image and return it
    return Image.fromarray(generated_image_numpy)


# --- Gradio Interface Creation ---
if __name__ == "__main__":
    # Create a Gradio interface
    interface = gr.Interface(
        fn=generate_pokemon_sprite, # Specify the function to be called when the user interacts
        inputs=gr.Textbox(lines=7, label="Pokémon Description"), # Define a multiline text input for the description
        outputs=gr.Image(label="Generated Pokémon Sprite", type="pil", width=IMAGE_OUTPUT_SIZE, height=IMAGE_OUTPUT_SIZE), # Define an image output, expecting a PIL image
        title="PikaPikaGen: Generator of Pokémon Sprites from Text Descriptions", # Set the title of the Gradio application
        description="Enter a Pokémon's description to generate its sprite", # Provide a descriptive text below the title
        flagging_mode="never" # Disable the flagging functionality
    )
    # Launch the Gradio interface
    interface.launch()