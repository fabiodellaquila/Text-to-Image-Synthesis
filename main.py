import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import ssim
from transformers import BertModel, BertTokenizer
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image
import os
import math
import random

# Dataset class for loading Pokemon data, including descriptions and images.
# Supports data augmentation for training.
class PokemonDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, max_len=128, augment=False):
        # Load the dataset from a TSV file.
        self.df = pd.read_csv(csv_path, sep='\t', quotechar='"', engine='python')
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

        # Base transformations applied to all images (resizing,ToTensor, normalization).
        base_transforms = [
            transforms.Resize((215, 215)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]

        # Apply horizontal flip for data augmentation if 'augment' is True.
        if self.augment:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)] + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        # Return the total number of samples in the dataset.
        return len(self.df)

    def __getitem__(self, idx):
        # Get description and image path for a given index.
        desc = self.df.iloc[idx]['description']
        number = str(self.df.iloc[idx]['national_number']).zfill(3)

        # Tokenize the description using the BERT tokenizer.
        encoded = self.tokenizer(desc, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        # Extract input IDs and attention mask.
        tokens = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Load and preprocess the image. Convert RGBA to RGB with a white background.
        img_path = os.path.join(self.image_dir, f"{number}.png")
        img = Image.open(img_path).convert("RGBA")

        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, (0, 0), img)

        # Apply defined image transformations.
        image = self.transform(background)

        return tokens, attention_mask, image


# Text Encoder using a pre-trained BERT model to generate textual embeddings.
class TextEncoder(nn.Module):
    def __init__(self, bert_model_name):
        super(TextEncoder, self).__init__()
        # Load a pre-trained BERT model.
        self.bert = BertModel.from_pretrained(bert_model_name)

    def forward(self, input_ids, attention_mask):
        # Pass input IDs and attention mask through BERT.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Return the last hidden states, which are contextualized embeddings.
        contextual_hidden_states = outputs.last_hidden_state
        return contextual_hidden_states


# Cross-Attention mechanism to allow interaction between different modalities (e.g., text and image features).
class CrossAttention(nn.Module):
    # n_heads: Number of attention heads. This allows the model to jointly attend to information
    #          from different representation subspaces at different positions.
    # d_embed: The dimensionality of the query (x) input.
    # d_cross: The dimensionality of the key/value (y) input. In this case, it's the dimension of the text embeddings.
    # in_proj_bias, out_proj_bias: Booleans to include bias in the linear projections.
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Query projection: Transforms the input 'x' (e.g., image features or a context vector) into a query representation.
        # Input shape: (Batch, Sequence_Length_Q, d_embed).
        # Output shape: (Batch, Sequence_Length_Q, d_embed).
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        # Key projection: Transforms the input 'y' (e.g., text embeddings) into a key representation.
        # Input shape: (Batch, Sequence_Length_KV, d_cross).
        # Output shape: (Batch, Sequence_Length_KV, d_embed).
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # Value projection: Transforms the input 'y' (e.g., text embeddings) into a value representation.
        # Input shape: (Batch, Sequence_Length_KV, d_cross).
        # Output shape: (Batch, Sequence_Length_KV, d_embed).
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # Output projection: Combines the outputs from all attention heads into a single, final output.
        # Input shape: (Batch, Sequence_Length_Q, d_embed).
        # Output shape: (Batch, Sequence_Length_Q, d_embed).
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        # d_head is the dimension of each attention head. The total embedding dimension (d_embed)
        # is split across n_heads.
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length_q, d_embed = input_shape
        # interim_shape is used to reshape the projected queries, keys, and values
        # for multi-head attention. It stacks the heads along a new dimension.
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Project input tensors to queries, keys, and values.
        # q (query): derived from 'x', used to query 'y'. Shape: (Batch, Sequence_Length_Q, d_embed)
        # k (key): derived from 'y', represents information to be queried. Shape: (Batch, Sequence_Length_KV, d_embed)
        # v (value): derived from 'y', contains the actual information to be retrieved. Shape: (Batch, Sequence_Length_KV, d_embed)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Reshape for multi-head attention:
        # The heads are separated into a distinct dimension.
        # For example, if q was (B, L_Q, D_E), it becomes (B, N_H, L_Q, D_H) after view and transpose.
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Calculate attention scores:
        # 'weight' represents how much each query token attends to each key token.
        # It's a dot product between query and key, divided by sqrt(d_head) for scaling.
        # Shape: (Batch, N_H, Sequence_Length_Q, Sequence_Length_KV)
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head) # Scale by square root of head dimension to prevent large values.
        weight = F.softmax(weight, dim=-1) # Apply softmax to get attention weights that sum to 1 along the KV dimension.

        # Compute the weighted sum of values:
        # The attention weights are multiplied with the values to get the context vector.
        # This is where the information from 'y' (e.g., text) is selectively aggregated based on 'x'.
        # Shape: (Batch, N_H, Sequence_Length_Q, D_H)
        output = weight @ v
        # Transpose and reshape back to original input shape for the output projection.
        # From (Batch, N_H, Sequence_Length_Q, D_H) to (Batch, Sequence_Length_Q, N_H * D_H) which is (Batch, Sequence_Length_Q, d_embed).
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape) # Reshape back to original input shape (Batch, Sequence_Length_Q, d_embed).
        output = self.out_proj(output) # Final linear projection to produce the refined output.
        return output

# Image Decoder (Generator) using Transposed Convolutional Layers and Cross-Attention.
class ImageDecoder(nn.Module):
    # latent_dim: Dimension of the random latent vector 'z' (noise input).
    # img_channels: Number of output image channels (e.g., 3 for RGB).
    # base_filters: The starting number of filters in the transposed convolutional layers.
    #               This determines the channel depth at the initial, smallest feature map.
    # output_size: The desired height and width of the output image (e.g., 215x215).
    # encoder_hidden_dim: The dimensionality of the hidden states from the text encoder (BERT).
    #                     This is the dimension of the text embeddings used for conditioning.
    # n_heads: Number of attention heads for the cross-attention modules.
    def __init__(self, latent_dim, img_channels=3, base_filters=1024, output_size=215, encoder_hidden_dim=256, n_heads=4):
        super(ImageDecoder, self).__init__()
        self.output_size = output_size
        self.encoder_hidden_dim = encoder_hidden_dim

        # Initial input dimension: latent vector z concatenated with CLS token (global context).
        self.initial_input_dim = latent_dim + encoder_hidden_dim

        # Fully connected layer: transforms (z + CLS) into a tensor suitable for a 4x4 feature map.
        # Input: (Batch, latent_dim + encoder_hidden_dim)
        # Output: (Batch, base_filters * 4 * 4) -> reshaped to (Batch, base_filters, 4, 4)
        self.fc = nn.Linear(self.initial_input_dim, base_filters * 4 * 4)

        # Transposed convolutional upsampling blocks.
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters, base_filters // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(True)
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters // 2, base_filters // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 4),
            nn.ReLU(True)
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters // 4, base_filters // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 8),
            nn.ReLU(True)
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(base_filters // 8, base_filters // 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 16),
            nn.ReLU(True)
        )
        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(base_filters // 16, base_filters // 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 32),
            nn.ReLU(True)
        )
        self.convT6 = nn.Sequential(
            nn.ConvTranspose2d(base_filters // 32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Cross-attention modules at different stages of the decoder.
        # These allow the decoder to incorporate semantic information from the text
        # embeddings at various resolutions during image generation.
        # The query's dimension (d_embed) is the channel count of the corresponding feature map.
        # The key/value dimension (d_cross) is the hidden dimension of the text encoder.
        self.cross_attn1 = CrossAttention(n_heads, base_filters // 2, encoder_hidden_dim)
        self.cross_attn2 = CrossAttention(n_heads, base_filters // 4, encoder_hidden_dim)
        self.cross_attn3 = CrossAttention(n_heads, base_filters // 8, encoder_hidden_dim)
        self.cross_attn4 = CrossAttention(n_heads, base_filters // 16, encoder_hidden_dim)
        self.cross_attn5 = CrossAttention(n_heads, base_filters // 32, encoder_hidden_dim)

    def forward(self, z, encoder_outputs):
        batch_size = z.shape[0]

        # Concatenate CLS token with random noise z as input for the decoder
        x = torch.cat([z, encoder_outputs[:, 0, :]], dim=1)

        # Pass through the fully connected layer and reshape to a 4x4 feature map.
        # This serves as the starting point for the transposed convolutions.
        # Shape of x: (Batch, base_filters, 4, 4)
        x = self.fc(x)
        x = x.view(batch_size, -1, 4, 4)

        # --- Decoder stages with cross-attention conditioning ---
        # At each stage, the feature maps are upsampled and then refined by adding
        # an attention output. The feature maps themselves act as queries to attend
        # over the full sequence of text embeddings.

        # Stage 1: Upsamples from 4x4 to 8x8.
        x = self.convT1(x)
        # Reshape the feature map into a sequence to act as queries for cross-attention.
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        q = x.flatten(2).transpose(1, 2)   # (B, Nq, Cq)
        # Compute attention output: how image features at this resolution attend to text embeddings.
        attn_out = self.cross_attn1(q, encoder_outputs)
        # Reshape the attention output back to a feature map and add it to the upsampled features.
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        x = x + attn_out.transpose(1, 2).view_as(x)

        # Stage 2: Upsamples from 8x8 to 16x16.
        x = self.convT2(x)
        q = x.flatten(2).transpose(1, 2)
        attn_out = self.cross_attn2(q, encoder_outputs)
        x = x + attn_out.transpose(1, 2).view_as(x)

        # Stage 3: Upsamples from 16x16 to 32x32.
        x = self.convT3(x)
        q = x.flatten(2).transpose(1, 2)
        attn_out = self.cross_attn3(q, encoder_outputs)
        x = x + attn_out.transpose(1, 2).view_as(x)

        # Stage 4: Upsamples from 32x32 to 64x64.
        x = self.convT4(x)
        q = x.flatten(2).transpose(1, 2)
        attn_out = self.cross_attn4(q, encoder_outputs)
        x = x + attn_out.transpose(1, 2).view_as(x)

        # Stage 5: Upsamples from 64x64 to 128x128.
        x = self.convT5(x)
        q = x.flatten(2).transpose(1, 2)
        attn_out = self.cross_attn5(q, encoder_outputs)
        x = x + attn_out.transpose(1, 2).view_as(x)

        # Final upsampling convolution to generate the full image.
        # Shape changes from (B, 32, 128, 128) to (B, 3, 256, 256).
        x = self.convT6(x)

        # Crop to the desired output size (215x215) from 256x256.
        start_idx = (x.shape[2] - self.output_size) // 2
        end_idx = start_idx + self.output_size
        generated_image = x[:, :, start_idx:end_idx, start_idx:end_idx]
        return generated_image



# Wrapper class combining the Text Encoder and Image Decoder to form the complete generator.
class PokemonGenerator(nn.Module):
    def __init__(self, bert_model_name, encoder_hidden_dim=256, latent_dim=100, decoder_base_filters=1024,
                 img_channels=3, output_size=215, n_heads=4):
        super(PokemonGenerator, self).__init__()
        self.text_encoder = TextEncoder(bert_model_name=bert_model_name)
        self.image_decoder = ImageDecoder(latent_dim=latent_dim, img_channels=img_channels, base_filters=decoder_base_filters,
                                          output_size=output_size, encoder_hidden_dim=encoder_hidden_dim, n_heads=n_heads)
        self.latent_dim = latent_dim
        self.encoder_hidden_dim = encoder_hidden_dim

    def forward(self, input_ids, attention_mask):
        # Encode the input text to get contextualized embeddings.
        encoder_outputs = self.text_encoder(input_ids, attention_mask)
        batch_size = input_ids.shape[0]

        # Generate a random latent vector `z` from a standard normal distribution.
        z = torch.randn(batch_size, self.latent_dim).to(input_ids.device)

        # Decode the latent vector and text embeddings into an image.
        generated_image = self.image_decoder(z, encoder_outputs)
        return generated_image


# Helper function to compute the Gram Matrix, used for style loss in Perceptual Loss.
def gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()  # a=batch size, b=feature maps, c,d=width,height
    features = input_tensor.view(a * b, c * d)  # Reshape to (features*batch, area)
    G = torch.mm(features, features.transpose(0, 1)) # Compute the outer product.
    return G.div(a * b * c * d)  # Normalize by the number of elements for stability.


# Perceptual Loss (VGG Loss) implementation, combining Content and Style losses.
class PerceptualLoss(nn.Module):
    def __init__(self, device, content_layers=None, style_layers=None):
        super(PerceptualLoss, self).__init__()
        # Load a pre-trained VGG19 model's features section.
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Define default layers for content and style loss if not provided.
        if content_layers is None:
            content_layers = [21]  # relu4_2
        if style_layers is None:
            style_layers = [
                2,  # relu1_2
                7,  # relu2_2
                12,  # relu3_2
                21,  # relu4_2
                30  # relu5_2
            ]

        self.content_layers = sorted(content_layers)
        self.style_layers = sorted(style_layers)

        # Create a sequential model containing only the necessary VGG layers up to the furthest required one.
        max_layer_idx = max(max(self.content_layers), max(self.style_layers))

        self.model = nn.Sequential()
        current_layer_idx = 0
        for i, layer in enumerate(vgg19):
            # Replace in-place ReLU with out-of-place for compatibility with feature extraction.
            if isinstance(layer, nn.ReLU):
                self.model.add_module(str(current_layer_idx), nn.ReLU(inplace=False))
            else:
                self.model.add_module(str(current_layer_idx), layer)

            if i >= max_layer_idx:
                break
            current_layer_idx += 1

        # Freeze VGG parameters as we only use it for feature extraction, not training.
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(device)  # Move the VGG model to the specified device.
        self.eval()  # Set the VGG model to evaluation mode.

        # Normalization transformation specific to VGG's expected input (mean and std for ImageNet).
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_features(self, image):
        # Denormalize image from [-1, 1] to [0, 1] before VGG-specific normalization.
        image_01 = (image + 1) / 2
        # Apply VGG normalization.
        image_vgg = self.normalize(image_01)

        features = {}
        x = image_vgg
        # Iterate through the VGG layers and extract features at specified points.
        for name, layer in self.model.named_children():
            x = layer(x)
            # Store features for content and style layers.
            if int(name) in self.content_layers:
                features[f'content_relu_{name}'] = x
            if int(name) in self.style_layers:
                features[f'style_relu_{name}'] = x
        return features

    def forward(self, generated_images, real_images, content_weight=1.0, style_weight=1.0):
        # Get feature maps for both generated and real images.
        gen_features = self.get_features(generated_images)
        real_features = self.get_features(real_images)

        content_loss = 0
        # Calculate content loss (L1 difference between feature maps).
        for layer_idx in self.content_layers:
            content_loss += F.l1_loss(gen_features[f'content_relu_{layer_idx}'],
                                      real_features[f'content_relu_{layer_idx}'])

        style_loss = 0
        # Calculate style loss (L1 difference between Gram matrices of feature maps).
        for layer_idx in self.style_layers:
            gen_gram = gram_matrix(gen_features[f'style_relu_{layer_idx}'])
            real_gram = gram_matrix(real_features[f'style_relu_{layer_idx}'])
            style_loss += F.l1_loss(gen_gram, real_gram)

        # Combine content and style losses with their respective weights.
        total_perceptual_loss = content_weight * content_loss + style_weight * style_loss
        return total_perceptual_loss


# Main execution block for training, validation, and testing.
if __name__ == "__main__":
    # Initialize BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-mini')

    # Define dataset paths.
    CSV_FILE = '../dataset/pokemon.csv'
    IMAGE_DIR = '../dataset/small_images'

    # Define model saving and image generation directories.
    MODEL_SAVE_DIR = './model'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'pokemon_generator.pth') # Path for saving/loading the model state.

    GENERATED_IMAGES_DIR = './generated_images'
    os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

    # Hyperparameters.
    BATCH_SIZE = 16
    NUM_EPOCHS = 300
    LEARNING_RATE = 0.0002

    encoder_hidden_dim = 256
    latent_vector_dim = 100
    num_attention_heads = 4

    # Determine the device to use (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("Loading and splitting dataset...")
        # Create a dataset instance to get total size for splitting. Data augmentation is not applied here.
        full_dataset_indices = PokemonDataset(csv_path=CSV_FILE, image_dir=IMAGE_DIR, tokenizer=tokenizer, max_len=128, augment=False)

        total_size = len(full_dataset_indices)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError("Dataset size too small for 80/10/10 split.")

        # Randomly shuffle indices for dataset splitting.
        indices = list(range(total_size))
        random.seed(42) # For reproducibility.
        random.shuffle(indices)

        # Split indices into training, validation, and test sets.
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create separate Dataset objects for train, validation, and test.
        # Data augmentation is enabled only for the training dataset.
        train_dataset = PokemonDataset(csv_path=CSV_FILE, image_dir=IMAGE_DIR, tokenizer=tokenizer, max_len=128, augment=True)
        val_dataset = PokemonDataset(csv_path=CSV_FILE, image_dir=IMAGE_DIR, tokenizer=tokenizer, max_len=128, augment=False)
        test_dataset = PokemonDataset(csv_path=CSV_FILE, image_dir=IMAGE_DIR, tokenizer=tokenizer, max_len=128, augment=False)

        # Create Subset objects using the generated indices.
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        test_subset = Subset(test_dataset, test_indices)

        # Create DataLoader instances for batching and shuffling data.
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        print(f"Dataset sizes: Training={len(train_subset)}, Validation={len(val_subset)}, Test={len(test_subset)}")

        print("\nStarting training...")
        # Initialize the PokemonGenerator model.
        model = PokemonGenerator(
            bert_model_name='prajjwal1/bert-mini', encoder_hidden_dim=encoder_hidden_dim, latent_dim=latent_vector_dim,
            decoder_base_filters=1024, img_channels=3, output_size=215, n_heads=num_attention_heads)
        model.to(device)

        # Load the previously saved model state if it exists, to resume training.
        if os.path.exists(MODEL_PATH):
            print(f"Loading previous model from {MODEL_PATH}...")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully. Training will continue from saved weights.")
        else:
            print("No previous model found. Starting training from scratch.")

        # Initialize the AdamW optimizer.
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        # L1 Loss for pixel-wise comparison.
        pixel_criterion = nn.L1Loss()

        # Perceptual Loss criterion using VGG19.
        perceptual_criterion = PerceptualLoss(device).to(device)

        # Weights for combining different loss components.
        PIXEL_LOSS_WEIGHT = 1
        PERCEPTUAL_CONTENT_WEIGHT = 0.5
        PERCEPTUAL_STYLE_WEIGHT = 100
        PERCEPTUAL_TOTAL_LOSS_WEIGHT= 0.05


        # Function to denormalize images from [-1, 1] to [0, 1] for saving.
        def denormalize(tensor):
            tensor = tensor * 0.5 + 0.5
            return tensor.clamp(0, 1)


        # Get a fixed batch from the validation and training loaders for consistent visualization of generated images
        # across epochs.
        fixed_val_batch = next(iter(val_loader))
        fixed_tokens, fixed_attention_mask, fixed_real_images = fixed_val_batch
        fixed_tokens = fixed_tokens.to(device)
        fixed_attention_mask = fixed_attention_mask.to(device)
        # Save a sample of real images from the validation set for comparison.
        utils.save_image(denormalize(fixed_real_images[:8]), os.path.join(GENERATED_IMAGES_DIR, 'real_images_sample_val.png'),
                         nrow=4, padding=2)

        fixed_train_batch = next(iter(train_loader))
        fixed_train_tokens, fixed_train_attention_mask, fixed_train_real_images = fixed_train_batch
        fixed_train_tokens = fixed_train_tokens.to(device)
        fixed_train_attention_mask = fixed_train_attention_mask.to(device)
        # Save a sample of real images from the training set for comparison.
        utils.save_image(denormalize(fixed_train_real_images[:8]), os.path.join(GENERATED_IMAGES_DIR, 'real_images_sample_train.png'),
                         nrow=4, padding=2)


        # Training loop.
        for epoch in range(NUM_EPOCHS):
            model.train() # Set model to training mode.
            running_train_total_loss = 0.0
            running_train_pixel_loss = 0.0
            running_train_perceptual_loss = 0.0
            running_train_ssim_score = 0.0 # Initialize SSIM score for training.

            for batch_idx, (tokens, attention_mask, real_images) in enumerate(train_loader):
                tokens, attention_mask, real_images = tokens.to(device), attention_mask.to(device), real_images.to(device)
                optimizer.zero_grad() # Clear gradients.
                generated_images = model(tokens, attention_mask) # Generate images.

                # Calculate L1 (pixel) loss.
                pixel_loss = pixel_criterion(generated_images, real_images)

                # Calculate perceptual loss.
                perceptual_loss_combined = perceptual_criterion(generated_images, real_images,
                    content_weight=PERCEPTUAL_CONTENT_WEIGHT, style_weight=PERCEPTUAL_STYLE_WEIGHT)

                # Calculate SSIM (Structural Similarity Index Measure).
                ssim_score = ssim(denormalize(generated_images), denormalize(real_images), data_range=1.0)

                # Combine all losses based on their weights.
                loss = PIXEL_LOSS_WEIGHT * pixel_loss + PERCEPTUAL_TOTAL_LOSS_WEIGHT * perceptual_loss_combined

                loss.backward() # Backpropagate the loss.

                # Apply gradient clipping to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step() # Update model parameters.
                # Accumulate loss and SSIM for reporting.
                running_train_total_loss += loss.item()
                running_train_pixel_loss += pixel_loss.item()
                running_train_perceptual_loss += perceptual_loss_combined.item()
                running_train_ssim_score += ssim_score.item()

            # Calculate average training metrics for the epoch.
            avg_train_total_loss = running_train_total_loss / len(train_loader)
            avg_train_pixel_loss = running_train_pixel_loss / len(train_loader)
            avg_train_perceptual_loss = running_train_perceptual_loss / len(train_loader)
            avg_train_ssim_score = running_train_ssim_score / len(train_loader)


            model.eval() # Set model to evaluation mode.
            running_val_total_loss = 0.0
            running_val_pixel_loss = 0.0
            running_val_perceptual_loss = 0.0
            running_val_ssim_score = 0.0 # Initialize SSIM score for validation.
            with torch.no_grad(): # Disable gradient calculations for validation.
                for tokens, attention_mask, real_images in val_loader:
                    tokens, attention_mask, real_images = tokens.to(device), attention_mask.to(device), real_images.to(device)
                    generated_images = model(tokens, attention_mask)

                    # Calculate validation losses and SSIM.
                    pixel_loss = pixel_criterion(generated_images, real_images)
                    perceptual_loss_combined = perceptual_criterion(generated_images, real_images,
                        content_weight=PERCEPTUAL_CONTENT_WEIGHT, style_weight=PERCEPTUAL_STYLE_WEIGHT)

                    ssim_score = ssim(denormalize(generated_images), denormalize(real_images), data_range=1.0)

                    loss = PIXEL_LOSS_WEIGHT * pixel_loss + PERCEPTUAL_TOTAL_LOSS_WEIGHT * perceptual_loss_combined

                    # Accumulate validation metrics.
                    running_val_total_loss += loss.item()
                    running_val_pixel_loss += pixel_loss.item()
                    running_val_perceptual_loss += perceptual_loss_combined.item()
                    running_val_ssim_score += ssim_score.item()

            # Calculate average validation metrics for the epoch.
            avg_val_total_loss = running_val_total_loss / len(val_loader)
            avg_val_pixel_loss = running_val_pixel_loss / len(val_loader)
            avg_val_perceptual_loss = running_val_perceptual_loss / len(val_loader)
            avg_val_ssim_score = running_val_ssim_score / len(val_loader)

            # Print epoch summary.
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
                  f"Train Total Loss: {avg_train_total_loss:.4f}, "
                  f"Train L1 Loss: {avg_train_pixel_loss:.4f}, "
                  f"Train Perceptual Loss: {avg_train_perceptual_loss:.4f}, "
                  f"Train SSIM: {avg_train_ssim_score:.4f}, "
                  f"Val Total Loss: {avg_val_total_loss:.4f}, "
                  f"Val L1 Loss: {avg_val_pixel_loss:.4f}, "
                  f"Val Perceptual Loss: {avg_val_perceptual_loss:.4f}, "
                  f"Val SSIM: {avg_val_ssim_score:.4f}")

            # Save generated images periodically for visual inspection of training progress.
            if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
                model.eval()
                with torch.no_grad():
                    # Generate and save images from the fixed training batch.
                    generated_images_for_viz_train = model(fixed_train_tokens, fixed_train_attention_mask)
                    save_path_train = os.path.join(GENERATED_IMAGES_DIR, f'generated_train_epoch_{epoch + 1:03d}.png')
                    utils.save_image(denormalize(generated_images_for_viz_train[:8]), save_path_train, nrow=4,padding=2)
                    print(f"Generated images (Training) saved to {save_path_train}")

                    # Generate and save images from the fixed validation batch.
                    generated_images_for_viz_val = model(fixed_tokens, fixed_attention_mask)
                    save_path_val = os.path.join(GENERATED_IMAGES_DIR, f'generated_val_epoch_{epoch + 1:03d}.png')
                    utils.save_image(denormalize(generated_images_for_viz_val[:8]), save_path_val, nrow=4, padding=2)
                    print(f"Generated images (Validation) saved to {save_path_val}")


        print("\nTraining completed.")
        # Save the final trained model.
        FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'pokemon_generator_new.pth')
        torch.save(model.state_dict(), FINAL_MODEL_PATH)

        # Evaluate the model on the test set.
        model.to(device)
        model.eval()

        test_total_loss = 0.0
        test_pixel_loss = 0.0
        test_perceptual_loss = 0.0
        test_ssim_score = 0.0 # Initialize SSIM for testing.
        with torch.no_grad():
            # Get a fixed batch from the test loader for visualization.
            final_test_batch = next(iter(test_loader))
            final_test_tokens, final_test_attention_mask, final_test_real_images = final_test_batch
            final_test_tokens = final_test_tokens.to(device)
            final_test_attention_mask = final_test_attention_mask.to(device)

            # Save a sample of real images from the test set.
            utils.save_image(denormalize(final_test_real_images[:8]), os.path.join(GENERATED_IMAGES_DIR, 'real_images_sample_test.png'),
                             nrow=4, padding=2)

            # Generate and save images from the test set using the final model.
            generated_images_test = model(final_test_tokens, final_test_attention_mask)
            save_path_final_test = os.path.join(GENERATED_IMAGES_DIR, 'generated_test.png')
            utils.save_image(denormalize(generated_images_test[:8]), save_path_final_test, nrow=4, padding=2)
            print(f"Generated images (Test) saved to {save_path_final_test}")

            # Iterate through the entire test set to calculate overall performance.
            for tokens, attention_mask, real_images in test_loader:
                tokens, attention_mask, real_images = tokens.to(device), attention_mask.to(device), real_images.to(device)
                generated_images = model(tokens, attention_mask)

                # Calculate losses and SSIM for the test set.
                pixel_loss = pixel_criterion(generated_images, real_images)
                perceptual_loss_combined = perceptual_criterion(generated_images, real_images,
                    content_weight=PERCEPTUAL_CONTENT_WEIGHT, style_weight=PERCEPTUAL_STYLE_WEIGHT)

                ssim_score = ssim(denormalize(generated_images), denormalize(real_images), data_range=1.0)

                loss = PIXEL_LOSS_WEIGHT * pixel_loss + PERCEPTUAL_TOTAL_LOSS_WEIGHT * perceptual_loss_combined

                test_total_loss += loss.item()
                test_pixel_loss += pixel_loss.item()
                test_perceptual_loss += perceptual_loss_combined.item()
                test_ssim_score += ssim_score.item()

        # Calculate average test metrics.
        avg_test_total_loss = test_total_loss / len(test_loader)
        avg_test_pixel_loss = test_pixel_loss / len(test_loader)
        avg_test_perceptual_loss = test_perceptual_loss / len(test_loader)
        avg_test_ssim_score = test_ssim_score / len(test_loader)

        # Print final test performance.
        print(f"\nPerformance on Test Set:")
        print(f"  Total Loss: {avg_test_total_loss:.4f}")
        print(f"  L1 Loss: {avg_test_pixel_loss:.4f}")
        print(f"  Perceptual Loss: {avg_test_perceptual_loss:.4f}")
        print(f"  SSIM Score: {avg_test_ssim_score:.4f}")

    # Error handling for file not found or dataset issues.
    except FileNotFoundError:
        print(f"Error: Make sure the paths {CSV_FILE} and {IMAGE_DIR} are correct.")
    except ValueError as ve:
        print(f"Error during dataset preparation: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")