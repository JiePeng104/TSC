import numpy as np
from PIL import Image
import os


def load_and_preprocess_image(image_path, label, save_path=None):
    # Load the image using PIL
    img = Image.open(image_path)

    # Resize the image to 224x224
    img = img.resize((224, 224))

    # Convert image to numpy array
    x = np.array(img)

    # Ensure the image has 3 channels (RGB)
    if len(x.shape) == 2:  # If grayscale, convert to RGB
        x = np.stack((x,) * 3, axis=-1)
    elif x.shape[2] == 4:  # If RGBA, remove alpha channel
        x = x[:, :, :3]

    # Ensure uint8 type with values in [0,255]
    x = x.astype('uint8')

    # Add batch dimension and ensure correct shape [1, 224, 224, 3]
    x = np.expand_dims(x, axis=0)

    # Store the label
    y = label

    # Save x and y to npz file if save_path is provided
    if save_path is not None:
        np.savez(save_path, x=x, y=[y])
        print(f"Saved data to {save_path}")

    return x, y


# Example usage
if __name__ == "__main__":
    # Example image path and label
    image_path = "F:\\forP\DECREE-master\data\\1103303.jpg"
    label = 100  # Your label (can be int, string, or any other format depending on your needs)
    save_path = "F:\\forP\DECREE-master\\reference\\CLIP\\waffles.npz"  # Path where to save the npz file

    # Load, preprocess, and save the image
    x, y = load_and_preprocess_image(image_path, label, save_path)

    # Print shapes and data type to verify
    print(f"Image shape: {x.shape}")  # Should be (1, 224, 224, 3)
    print(f"Image dtype: {x.dtype}")  # Should be uint8
    print(f"Value range: [{x.min()}, {x.max()}]")  # Should be in [0, 255]
    print(f"Label: {y}")

    # Verify the saved data
    loaded_data = np.load(save_path)
    print("\nVerifying saved data:")
    print(f"Loaded image shape: {loaded_data['x'].shape}")
    print(f"Loaded image dtype: {loaded_data['x'].dtype}")
    print(f"Loaded label: {loaded_data['y']}")