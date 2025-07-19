#!/usr/bin/env python3
"""
Convert your specific VQ-VAE model to ONNX format.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


class EncoderWrapper(nn.Module):
    """Wrapper for the encoder to ensure proper ONNX export."""
    
    def __init__(self, vqvae_model):
        super().__init__()
        self.vqvae = vqvae_model
        
    def forward(self, x):
        # Your encode method returns int64 indices
        indices = self.vqvae.encode(x)
        # Convert to uint8 for smaller storage (0-255 range is sufficient)
        return indices.to(torch.uint8)


class DecoderWrapper(nn.Module):
    """Wrapper for the decoder to ensure proper ONNX export."""
    
    def __init__(self, vqvae_model):
        super().__init__()
        self.vqvae = vqvae_model
        
    def forward(self, indices_uint8):
        # Convert uint8 back to int64 for embedding lookup
        indices = indices_uint8.to(torch.int64)
        return self.vqvae.decode(indices)


def load_vqvae_model(model_path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    """Load your VQ-VAE model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading VQ-VAE model from: {model_path}")
    device_obj = torch.device(device)
    model = torch.jit.load(model_path, map_location=device_obj)
    model.eval()
    
    return model


def convert_vqvae_to_onnx(
    model_path: str,
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 1,
    opset_version: int = 11,
    validate: bool = True
) -> Tuple[str, str]:
    """Convert your VQ-VAE model to ONNX format."""
    
    # Load the model
    vqvae_model = load_vqvae_model(model_path, device)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    encoder_path = output_path / "encoder.onnx"
    decoder_path = output_path / "decoder.onnx"
    
    device_obj = torch.device(device)
    
    # Create test inputs
    encoder_input = torch.randn(batch_size, 1, 8, 8, 8, dtype=torch.float32, device=device_obj)
    
    # Test the original model to understand the shapes
    with torch.no_grad():
        original_indices = vqvae_model.encode(encoder_input)
        original_reconstruction = vqvae_model.decode(original_indices)
        
        print(f"Original model test:")
        print(f"  Input shape: {encoder_input.shape} ({encoder_input.dtype})")
        print(f"  Indices shape: {original_indices.shape} ({original_indices.dtype})")
        print(f"  Reconstruction shape: {original_reconstruction.shape} ({original_reconstruction.dtype})")
        
        # Create decoder test input (uint8 version of indices)
        decoder_input = original_indices.to(torch.uint8)
        print(f"  Decoder input shape: {decoder_input.shape} ({decoder_input.dtype})")
    
    # Create wrappers
    encoder_wrapper = EncoderWrapper(vqvae_model)
    encoder_wrapper = torch.jit.script(encoder_wrapper)
    decoder_wrapper = DecoderWrapper(vqvae_model)
    decoder_wrapper = torch.jit.script(decoder_wrapper)
    
    # Export encoder
    print("\n=== Exporting Encoder ===")
    print(f"Input: {encoder_input.shape} {encoder_input.dtype}")
    
    with torch.no_grad():
        encoder_output = encoder_wrapper(encoder_input)
        print(f"Output: {encoder_output.shape} {encoder_output.dtype}")
    
    torch.onnx.export(
        encoder_wrapper,
        encoder_input,
        str(encoder_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    # Export decoder
    print("\n=== Exporting Decoder ===")
    print(f"Input: {decoder_input.shape} {decoder_input.dtype}")
    
    with torch.no_grad():
        decoder_output = decoder_wrapper(decoder_input)
        print(f"Output: {decoder_output.shape} {decoder_output.dtype}")
    
    torch.onnx.export(
        decoder_wrapper,
        decoder_input,
        str(decoder_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    # Validate models
    if validate:
        print("\n=== Validation ===")
        
        # Validate encoder
        encoder_valid = validate_onnx_model(
            str(encoder_path), 
            encoder_wrapper, 
            encoder_input,
            "Encoder"
        )
        
        # Validate decoder
        decoder_valid = validate_onnx_model(
            str(decoder_path), 
            decoder_wrapper, 
            decoder_input,
            "Decoder"
        )
        
        if encoder_valid and decoder_valid:
            print("✓ All models validated successfully!")
        else:
            print("✗ Some models failed validation")
    
    print(f"\n=== Conversion Complete ===")
    print(f"Encoder saved to: {encoder_path}")
    print(f"Decoder saved to: {decoder_path}")
    
    return str(encoder_path), str(decoder_path)


def validate_onnx_model(onnx_path: str, pytorch_model: nn.Module, dummy_input: torch.Tensor, name: str) -> bool:
    """Validate ONNX model against PyTorch model."""
    
    print(f"Validating {name}: {onnx_path}")
    
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ {name} ONNX model is valid")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Test inference
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).cpu().numpy()
            
            # ONNX inference
            ort_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            onnx_outputs = session.run(None, ort_inputs)
            onnx_output = onnx_outputs[0]
            
            # Compare outputs
            if np.allclose(pytorch_output, onnx_output, atol=1e-5):
                print(f"  ✓ {name} outputs match")
                return True
            else:
                max_diff = np.max(np.abs(pytorch_output - onnx_output))
                print(f"  ✗ {name} outputs differ (max diff: {max_diff:.6f})")
                return False
                
    except Exception as e:
        print(f"  ✗ {name} validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert VQ-VAE model to ONNX")
    parser.add_argument("input_path", help="Path to your inference_vqvae_optimized.pt file")
    parser.add_argument("output_dir", help="Directory to save ONNX models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument("--no-validate", action="store_true")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        args.device = "cpu"
    
    try:
        encoder_path, decoder_path = convert_vqvae_to_onnx(
            args.input_path,
            args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            opset_version=args.opset_version,
            validate=not args.no_validate
        )
        
        print("\n🎉 Conversion successful!")
        print(f"Encoder: {encoder_path}")
        print(f"Decoder: {decoder_path}")
        
        # Print usage instructions
        print(f"\nTo use with your C++ backend:")
        print(f"config.source = OnnxModelPaths{{")
        print(f"    .encoder_path = \"{encoder_path.replace(chr(92), '/')}\",")
        print(f"    .decoder_path = \"{decoder_path.replace(chr(92), '/')}\",")
        print(f"}};")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()