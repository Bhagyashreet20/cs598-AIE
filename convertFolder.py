import os
import struct
import argparse

def pack_directory(input_dir, output_file):
    with open(output_file, 'wb') as bin_file:
        for root, _, files in os.walk(input_dir):
            for file in files:
                # Get the full and relative file paths
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, input_dir)
                # Read file content
                with open(full_path, 'rb') as f:
                    data = f.read()
                # Encode the relative file path
                relative_path_encoded = relative_path.encode('utf-8')
                # Pack the file name length and name
                bin_file.write(struct.pack('I', len(relative_path_encoded)))
                bin_file.write(relative_path_encoded)
                # Pack the file size and data
                bin_file.write(struct.pack('Q', len(data)))
                bin_file.write(data)

def unpack_directory(input_file, output_dir):
    with open(input_file, 'rb') as bin_file:
        while True:
            # Read the length of the file name
            name_length_data = bin_file.read(4)
            if not name_length_data:
                break  # EOF reached
            name_length = struct.unpack('I', name_length_data)[0]
            # Read the file name
            file_name = bin_file.read(name_length).decode('utf-8')
            # Read the file size
            file_size = struct.unpack('Q', bin_file.read(8))[0]
            # Read the file data
            data = bin_file.read(file_size)
            # Create the full output path
            output_path = os.path.join(output_dir, file_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Write the file data
            with open(output_path, 'wb') as f:
                f.write(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack or unpack a directory to/from a .bin file.")
    parser.add_argument('action', choices=['pack', 'unpack'], help="Action to perform: pack or unpack.")
    parser.add_argument('input', help="Input directory (for pack) or .bin file (for unpack).")
    parser.add_argument('output', help="Output .bin file (for pack) or directory (for unpack).")

    args = parser.parse_args()

    if args.action == 'pack':
        pack_directory(args.input, args.output)
        print(f"Packed '{args.input}' into '{args.output}'.")
    elif args.action == 'unpack':
        unpack_directory(args.input, args.output)
        print(f"Unpacked '{args.input}' into '{args.output}'.")
