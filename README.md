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
| Index Tensors  | [B × 4 × 4 × 4] uint8 values              |
| Origins        | Per-leaf grid coordinates                 |

---

## 🧠 How it Works

### Training Pipeline
> I'm very bad at programming and couldn't compile pyopenvdb. so I had to extract vdb data to .npy files. 

1. **Leaf Extraction**  
   Extract all non-empty 8×8×8 voxel blocks as well as the coords of each leaf origins (for reconstruction) from a dataset of VDB volumes.

2. **VQ-VAE Model**  
   A PyTorch-based encoder compresses each leaf into a latent vector of size `D` in this case 128 dimmenions.  
   The quantizer maps this vector to the closest of `K` learned codebook entries, the codebook as 256, which makes it fit in a uint8_t.

3. **Loss Function**  
   Optimized using the following objective:
   ```math
   \mathcal{L} = \|x - \hat{x}\|^2 + \beta \cdot \| \text{sg}[z_e(x)] - e \|^2
   ```
   with **Exponential Moving Average (EMA)** updates to the embedding table.

### Runtime Decompression (C++/CUDA)

1. **Load `.vqvdb` file**, which contains:
   - Codebook (float32)
   - Per-leaf index tensors (`4×4×4×uint8_t = 64 bytes`)
   - Leaf origins (`openvdb::Coord`)

2. **GPU Decoding**
   - Launch CUDA kernel to decode latent codes into dense voxel blocks
   - Allocate leaf nodes in a new `openvdb::FloatGrid`
   - Write the reconstructed 8×8×8 voxel blocks

3. **Streaming-Friendly**
   - Decompression supports lazy loading in batches
   - Low VRAM footprint when streaming large scenes

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
