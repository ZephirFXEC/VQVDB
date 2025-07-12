import argparse
import os

def convert_to_header(input_file, output_file, array_name="g_model_data"):
    """
    Converts a binary file into a C++ header file containing a byte array.
    """
    try:
        with open(input_file, 'rb') as f:
            binary_data = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Get the base name for include guard
    guard_name = os.path.basename(output_file).upper().replace('.', '_')

    with open(output_file, 'w') as f:
        f.write(f'#pragma once\n\n')
        f.write(f'#ifndef {guard_name}\n')
        f.write(f'#define {guard_name}\n\n')
        f.write('#include <cstddef> // For size_t\n')
        f.write('#include <cstdint> // For uint8_t\n\n')

        f.write(f'// Model: {os.path.basename(input_file)}\n')
        f.write(f'// Size:  {len(binary_data)} bytes\n\n')

        # Define the array size
        f.write(f'const size_t {array_name}_size = {len(binary_data)};\n\n')

        # Define the byte array
        f.write(f'const unsigned char {array_name}[{array_name}_size] = {{\n    ')

        # Write byte data, 12 bytes per line for readability
        for i, byte in enumerate(binary_data):
            f.write(f'0x{byte:02x}, ')
            if (i + 1) % 12 == 0 and (i + 1) != len(binary_data):
                f.write('\n    ')
        
        f.write('\n};\n\n')
        f.write(f'#endif // {guard_name}\n')

    print(f"Successfully converted '{input_file}' to '{output_file}'")
    print(f"C++ array name: '{array_name}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert binary file to a C++ header.")
    parser.add_argument('input', help="Path to the binary input file (e.g., model.pt).")
    parser.add_argument('output', help="Path for the generated C++ header file (e.g., model_data.h).")
    parser.add_argument('--name', default="g_model_data", help="Name of the C++ byte array variable.")
    
    args = parser.parse_args()
    convert_to_header(args.input, args.output, args.name)