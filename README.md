<h1 align="center">VQVDB: AI Compression for OpenVDB Grids </h1>


<div align="center">
  <a> <img src="https://github.com/user-attachments/assets/5285b3b3-fa68-4710-a29f-ecba1a6d8acf"> </a>
  <a> <img src="https://github.com/user-attachments/assets/2b410d95-019d-46eb-9970-0465d7deb7a7"> </a>
</div>
<br>

# Oveview 

VQVDB is a deep learning–powered compressor for volumetric data stored in OpenVDB. 
It uses Vector Quantized Variational Autoencoders (VQ-VAE) to learn a compact latent space and achieves up to 32× compression of float voxel grids, nearly lossless at the visual level.

VQVDB is designed for GPU-accelerated decoding via CUDA but also support CPU encoding / decoding, enabling real-time decompression of large volumes, with native support for integration into Houdini.

## 📂 File Format

Each `.vqvdb` file stores:

| Section         | Description                              |
|----------------|-------------------------------------------|
| Header         | Magic, version, codebook size, shape info |
| Codebook       | 256×128 float matrix                      |
| Index Tensors  | [B × 8 × 8 × 8] uint8 values              |
| Origins        | Per-leaf grid coordinates                 |

---


## 📈 Future Work

- Hierarchical VQ-VAE (multi-res compression)
- Residual VAE
- Transformer-based latent upsampling
- Real-time decompression via Vulkan compute
- VDB segmentation / semantic-aware encoding

---

## 📜 Citation

If you use VQVDB in academic work, please cite the project:

```bibtex
@misc{vqvdb2025,
  title={VQVDB: VDB Compression using Vector Quantized Autoencoders},
  author={Enzo Crema},
  year={2025},
  url={https://github.com/zephirfx/vqvdb}
}
```
