import safetensors
import struct
import os
import argparse
import safetensors.torch

def convert_safetensors_to_binary(input_path, output_path):
    # Load the tensors from the safetensors file
    tensors = safetensors.torch.load_file(input_path)

    # Open the output file in binary write mode
    with open(output_path, "wb") as f:
        # Iterate through the tensors
        for tensor_name, tensor in tensors.items():
            # Write metadata: tensor name and size
            f.write(struct.pack("I", len(tensor_name)))  # Length of tensor name
            f.write(tensor_name.encode("utf-8"))          # Tensor name bytes
            f.write(struct.pack("I", tensor.numel()))     # Number of elements in tensor
            # Write the tensor data
            f.write(tensor.numpy().tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a safetensors file to a binary file.")
    parser.add_argument("input_file", type=str, help="Path to the input safetensors file.")
    parser.add_argument("output_file", type=str, help="Path to the output binary file.")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
    else:
        # Convert safetensors file to binary file
        convert_safetensors_to_binary(args.input_file, args.output_file)
        print(f"Conversion complete. Binary file saved as '{args.output_file}'")
