import torch
import numpy as np

def convert_pth_to_bin(pth_file, output_bin_file):
    # Load the PyTorch checkpoint
    checkpoint = torch.load(pth_file, map_location=torch.device('cpu'))
    
    # Assuming the checkpoint contains a state_dict with model weights
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Flatten all weights into a single array
    all_weights = []
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Convert BFloat16 to float32 for compatibility with NumPy
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(dtype=torch.float32)
            all_weights.append(tensor.flatten().cpu().numpy())
        else:
            print(f"Skipping non-tensor key: {key}")
    
    # Concatenate all weights
    flattened_weights = np.concatenate(all_weights, axis=0).astype(np.float32)

    # Save as a binary file
    with open(output_bin_file, "wb") as f:
        flattened_weights.tofile(f)

    print(f"Saved {len(flattened_weights)} weights to {output_bin_file}.")

# Example usage
convert_pth_to_bin("/work/hdd/bdof/nkanamarla/models/LLAMA3download/Llama3.2-3B/consolidated.00.pth", "/work/hdd/bdof/nkanamarla/models/LLAMA3download/Llama3.2-3B/model_checkpoint.bin")
