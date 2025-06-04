# VQVDB Compression Tools

This repository contains a prototype VQ-VAE based system for compressing OpenVDB
leaf nodes.  Training utilities are provided in `VQVAE.py` and the runtime
decoder is implemented in C++.

## Encoding

The `VQVAE.py` script can train a model and compress a dataset of extracted
leaf nodes.  When supplying an additional `--origins_file` pointing to a Numpy
array of shape `(N,3)` with integer leaf coordinates, the encoder embeds these
positions in the output `.vqvdb` file.

Example:

```bash
python VQVAE.py --input_dir leaves/ --output_dir out/ \
    --origins_file leaves_origins.npy
```

The resulting file `compressed_indices.vqvdb` contains both the codebook indices
and the leaf origins.

## Decoding

`vqvdb_decoder` reconstructs a new VDB grid using a trained model and the
compressed file:

```bash
./vqvdb_decoder model.pt compressed_indices.vqvdb template.vdb output.vdb
```

The decoder no longer needs to derive leaf positions from the template grid;
it reads them directly from the `.vqvdb` file.  The template VDB is only used
for metadata such as grid transform, class and name.
