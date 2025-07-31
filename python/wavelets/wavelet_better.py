from __future__ import annotations
import os

os.add_dll_directory("C:/Users/zphrfx/Desktop/hdk/VQVDB/.venv/Lib/site-packages")
os.add_dll_directory("C:/vcpkg/installed/x64-windows/bin")

import argparse
import json
import pathlib
import struct
import logging
from typing import List, Tuple

import numpy as np
import openvdb as vdb
import pywt
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.decomposition import IncrementalPCA

# Ensure predictable backend
tl.set_backend("numpy")

# ------------------------------------------------------------
# Wavelet helpers
# ------------------------------------------------------------

def wavedec3(block: np.ndarray, wavelet: str, level: int) -> Tuple[np.ndarray, List[slice], List[np.ndarray]]:
    coeffs = pywt.wavedecn(block, wavelet=wavelet, level=level, mode="periodization")
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr.astype(np.float32), coeff_slices, coeffs


def waverec3(arr: np.ndarray, coeff_slices, wavelet: str) -> np.ndarray:
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format="wavedecn")
    return pywt.waverecn(coeffs, wavelet=wavelet, mode="periodization")


# ------------------------------------------------------------
# Quantisation helpers
# ------------------------------------------------------------

def quantise(x: np.ndarray, n_bits: int) -> Tuple[np.ndarray, float]:
    maxval = np.max(np.abs(x))
    scale = ((2**(n_bits - 1)) - 1) / maxval if maxval else 1.0
    return np.round(x * scale).astype(np.int16), scale


def dequantise(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) / scale


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    return float(np.sqrt(np.mean((a_f - b_f) ** 2)))


# ------------------------------------------------------------
# CP helpers
# ------------------------------------------------------------

def _two_stage_cp_compress(q_tensor: np.ndarray, rank: int, batch_size: int = 256):
    """
    Memory-efficient two-stage approximate CP (Kruskal) decomposition with
    the PCA mean folded in as an extra constant temporal component.
    Returns: (weights, [A, B, C]) where rank returned is original rank+1.
    """
    n_frames, n_blocks, coeff_len = q_tensor.shape

    logging.debug("Reshaping tensor to (%d, %d) for PCA...", n_frames, n_blocks * coeff_len)
    flat_tensor = q_tensor.reshape(n_frames, -1).astype(np.float32)

    ipca = IncrementalPCA(n_components=rank, batch_size=batch_size)

    logging.info("Running Incremental PCA to extract temporal factor...")
    A = ipca.fit_transform(flat_tensor).astype(np.float32)  # (n_frames, rank)
    combined_factors = ipca.components_.T.astype(np.float32)  # (n_blocks*coeff_len, rank)
    mean_flat = ipca.mean_.astype(np.float32)  # (n_blocks*coeff_len,)

    logging.debug("Separating spatial and coefficient factors via SVD...")
    weights = np.zeros(rank, dtype=np.float32)
    B = np.zeros((n_blocks, rank), dtype=np.float32)
    C = np.zeros((coeff_len, rank), dtype=np.float32)

    for r in range(rank):
        comp_matrix = combined_factors[:, r].reshape(n_blocks, coeff_len)
        try:
            U, s, Vh = np.linalg.svd(comp_matrix, full_matrices=False)
            weights[r] = s[0]
            B[:, r] = U[:, 0]
            C[:, r] = Vh[0, :]
        except np.linalg.LinAlgError:
            logging.warning("SVD failed for component %d; zeroing.", r)
            weights[r] = 0.0
            B[:, r] = 0.0
            C[:, r] = 0.0

    # Incorporate the PCA mean as an extra component (constant over time)
    logging.debug("Processing PCA mean as extra Kruskal component...")
    mean_matrix = mean_flat.reshape(n_blocks, coeff_len)
    try:
        U_m, s_m, Vh_m = np.linalg.svd(mean_matrix, full_matrices=False)
        weight_mean = s_m[0]
        B_mean = U_m[:, 0]
        C_mean = Vh_m[0, :]
    except np.linalg.LinAlgError:
        logging.warning("SVD failed on mean matrix; using zeros for mean component.")
        weight_mean = 0.0
        B_mean = np.zeros(n_blocks, dtype=np.float32)
        C_mean = np.zeros(coeff_len, dtype=np.float32)

    A_mean = np.ones((n_frames, 1), dtype=np.float32)  # temporal factor: constant

    # Append mean component
    weights = np.concatenate([weights, np.array([weight_mean], dtype=np.float32)], axis=0)
    A = np.concatenate([A, A_mean], axis=1)  # now (n_frames, rank+1)
    B = np.concatenate([B, B_mean[:, None]], axis=1)  # (n_blocks, rank+1)
    C = np.concatenate([C, C_mean[:, None]], axis=1)  # (coeff_len, rank+1)

    logging.info("Two-stage CP compression produced rank %d (including mean).", weights.shape[0])
    return weights, [A, B, C]


def _refine_cp(q_tensor: np.ndarray, weights, factors, n_iter=10, tol=1e-6):
    """
    Attempt to refine the CP approximation using ALS starting from the provided factors.
    Falls back silently if the API variant doesn't apply.
    """
    tensor = q_tensor.astype(np.float32)
    rank = weights.shape[0]
    logging.info("Attempting CP refinement with ALS (rank=%d, max_iter=%d)...", rank, n_iter)
    try:
        # Try modern variant first
        cp_refined = parafac(tensor, rank=rank, init=(weights, factors), n_iter_max=n_iter, tol=tol)
    except TypeError:
        try:
            cp_refined = parafac(tensor, rank=rank, init="custom", init_factors=(weights, factors), n_iter_max=n_iter, tol=tol)
        except Exception as e:
            logging.warning("CP refinement failed (fallback API) with error: %s; using initial factors.", e)
            return weights, factors
    except Exception as e:
        logging.warning("CP refinement failed with error: %s; using initial factors.", e)
        return weights, factors

    if hasattr(cp_refined, "weights") and hasattr(cp_refined, "factors"):
        logging.info("CP refinement succeeded.")
        return cp_refined.weights, cp_refined.factors
    else:
        # assume tuple
        logging.info("CP refinement returned raw tuple.")
        return cp_refined  # type: ignore


def cp_decompress(weights: np.ndarray, factors):
    return tl.cp_to_tensor((weights, factors))


# ------------------------------------------------------------
# Leaf block insertion (vectorized)
# ------------------------------------------------------------

def insert_leaf_blocks(grid_template: vdb.FloatGrid, coords, blocks, block_size: int = 8):
    grid = grid_template.deepCopy()
    acc = grid.getAccessor()
    grid.clear()

    for origin, blk in zip(coords, blocks):
        nonzero = np.nonzero(blk)
        values = blk[nonzero]
        ox, oy, oz = origin
        for dx, dy, dz, val in zip(*nonzero, values):
            coord = (int(ox + dx), int(oy + dy), int(oz + dz))
            acc.setValueOn(coord, float(val))
    return grid


# ------------------------------------------------------------
# Container header
# ------------------------------------------------------------

MAGIC = b"VDBR\0"
HEADER_STRUCT = struct.Struct("<5sHHHHHI")  # magic, vmaj, vmin, blk, lvl, rank, n_frames
META_STRUCT = struct.Struct("<fIII")  # scale, coeff_len, n_blocks, json_bytes


# ------------------------------------------------------------
# Compressor class
# ------------------------------------------------------------

class VDBSequenceCompressor:
    def __init__(
        self,
        block=8,
        levels=2,
        rank=32,
        quant_bits=12,
        wavelet="haar",
    ):
        self.block = block
        self.levels = levels
        self.rank = rank
        self.quant_bits = quant_bits
        self.wavelet = wavelet

    def _analyse_frames_from_npy(self, npy_paths: List[str], origin_paths: List[str]):
        if not origin_paths:
            raise ValueError("Origin paths must be provided when loading from .npy")

        n_frames = len(npy_paths)
        if n_frames == 0:
            return np.array([]), [], None

        logging.info("Finding the union of active blocks across all frames...")
        all_origins = set()
        frame_origin_maps = []
        for origin_path in origin_paths:
            origins_np = np.load(origin_path, allow_pickle=False)
            origin_map = {tuple(o): i for i, o in enumerate(origins_np)}
            frame_origin_maps.append(origin_map)
            for o in origin_map.keys():
                all_origins.add(o)

        master_coords = sorted(list(all_origins))
        n_total_blocks = len(master_coords)
        coord_to_master_idx = {coord: i for i, coord in enumerate(master_coords)}

        logging.info("Found %d unique blocks across %d frames.", n_total_blocks, n_frames)

        logging.info("Determining wavelet coefficient length...")
        dummy_block = np.zeros((self.block, self.block, self.block), dtype=np.float32)
        dummy_coeffs, _, _ = wavedec3(dummy_block, self.wavelet, self.levels)
        coeff_len = dummy_coeffs.size
        logging.info("Wavelet coefficient length is %d.", coeff_len)

        final_coeff_tensor = np.zeros((n_frames, n_total_blocks, coeff_len), dtype=np.float32)

        for i, npy_path in enumerate(npy_paths):
            logging.debug("Processing frame %d/%d", i + 1, n_frames)
            dense_blocks_this_frame = np.load(npy_path, allow_pickle=False)
            origin_map_this_frame = frame_origin_maps[i]

            for origin_tuple, local_idx in origin_map_this_frame.items():
                master_idx = coord_to_master_idx[origin_tuple]

                blk = dense_blocks_this_frame[local_idx].astype(np.float32, copy=False)
                arr, _, _ = wavedec3(blk, self.wavelet, self.levels)

                final_coeff_tensor[i, master_idx, :] = arr.ravel()

        logging.info("Finished processing all frames.")
        return final_coeff_tensor, master_coords, None

    def compress_from_npy(
        self,
        npy_paths: List[str],
        origin_paths: List[str],
        out_path: str,
        grid_metadata: dict,
        refine_cp: bool = False,
    ):
        coeff_tensor, coords, _ = self._analyse_frames_from_npy(npy_paths, origin_paths)

        if coeff_tensor.size == 0:
            logging.warning("No data to compress; exiting.")
            return

        n_frames, n_blocks, coeff_len = coeff_tensor.shape

        if n_frames <= 1:
            raise ValueError(
                f"CP/Parafac decomposition requires a sequence of length > 1. Received {n_frames} frames."
            )

        # Quantisation
        q_tensor, scale = quantise(coeff_tensor, self.quant_bits)
        logging.info(
            "Quantisation complete. Bits=%d. Quantization RMSE (wavelet domain): %.6f",
            self.quant_bits,
            compute_rmse(coeff_tensor, dequantise(q_tensor, scale)),
        )

        logging.info("Performing two-stage CP compression on quantized tensor of shape %s...", q_tensor.shape)
        weights, factors = _two_stage_cp_compress(q_tensor, self.rank)
        A, B, C = factors

        # CP approximation error before refinement
        approx_q = cp_decompress(weights, [A, B, C])
        rmse_before = compute_rmse(q_tensor, approx_q)
        logging.info("CP approximation RMSE before refinement: %.6f (on quantized tensor)", rmse_before / scale)

        # Optional refinement
        if refine_cp:
            weights_refined, factors_refined = _refine_cp(q_tensor, weights, [A, B, C])
            A, B, C = factors_refined
            weights = weights_refined
            approx_q_refined = cp_decompress(weights, [A, B, C])
            rmse_after = compute_rmse(q_tensor, approx_q_refined)
            logging.info("CP approximation RMSE after refinement: %.6f (on quantized tensor)", rmse_after / scale)
        else:
            logging.debug("Skipping CP refinement (disabled).")

        logging.info("Final CP rank stored: %d", weights.shape[0])
        logging.info("Factor shapes: A=%s, B=%s, C=%s", A.shape, B.shape, C.shape)

        # Metadata augmentation
        if not all(k in grid_metadata for k in ["grid_name", "voxel_size", "background"]):
            raise ValueError("grid_metadata must contain 'grid_name', 'voxel_size', and 'background'")

        grid_metadata["wavelet"] = self.wavelet
        grid_metadata["quant_bits"] = self.quant_bits

        meta_bytes = json.dumps(grid_metadata).encode()

        # ---------------- write container ----------------
        with open(out_path, "wb") as fp:
            fp.write(
                HEADER_STRUCT.pack(
                    MAGIC, 0, 2, self.block, self.levels, weights.shape[0], n_frames
                )
            )
            fp.write(META_STRUCT.pack(scale, coeff_len, n_blocks, len(meta_bytes)))
            fp.write(meta_bytes)
            fp.write(np.asarray(coords, dtype=np.int32).tobytes())
            fp.write(weights.astype(np.float32).tobytes())
            fp.write(A.astype(np.float32).tobytes())
            fp.write(B.astype(np.float32).tobytes())
            fp.write(C.astype(np.float32).tobytes())

        compressed_size = pathlib.Path(out_path).stat().st_size
        uncompressed_size = coeff_tensor.nbytes  # approximate
        ratio = uncompressed_size / compressed_size if compressed_size else float("inf")
        logging.info(
            "Successfully compressed %d frames to %s. Compression ratio (wavelet coeffs raw / file): %.2f",
            n_frames,
            out_path,
            ratio,
        )

    def decompress(self, in_path: str, out_dir: str):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(in_path, "rb") as fp:
            magic, vmaj, vmin, blk, lvl, rank, n_frames = HEADER_STRUCT.unpack(
                fp.read(HEADER_STRUCT.size)
            )
            if magic != MAGIC:
                raise ValueError("Invalid VDBR file")
            scale, coeff_len, n_blocks, json_bytes = META_STRUCT.unpack(fp.read(META_STRUCT.size))
            meta = json.loads(fp.read(json_bytes))
            coords = np.frombuffer(fp.read(n_blocks * 3 * 4), dtype=np.int32).reshape(n_blocks, 3)
            weights = np.frombuffer(fp.read(rank * 4), dtype=np.float32)
            A = np.frombuffer(fp.read(n_frames * rank * 4), dtype=np.float32).reshape(n_frames, rank)
            B = np.frombuffer(fp.read(n_blocks * rank * 4), dtype=np.float32).reshape(n_blocks, rank)
            C = np.frombuffer(fp.read(coeff_len * rank * 4), dtype=np.float32).reshape(coeff_len, rank)

        # Reconstruct quantized tensor
        logging.info("Reconstructing quantized tensor via CP inverse...")
        q_matrix = cp_decompress(weights, [A, B, C])  # (n_frames, n_blocks, coeff_len)
        q_tensor = np.round(q_matrix).astype(np.int16).reshape(n_frames, n_blocks, coeff_len)
        coeff_tensor = dequantise(q_tensor, scale)

        # Wavelet synthesis prep
        wavelet = meta.get("wavelet", self.wavelet)
        dummy = np.zeros((blk, blk, blk), dtype=np.float32)
        _, coeff_slices, _ = wavedec3(dummy, wavelet, lvl)

        # Build template grid
        template_grid = vdb.FloatGrid(background=meta["background"])
        template_grid.name = meta["grid_name"]
        voxel_size = meta["voxel_size"]
        template_grid.transform = vdb.createLinearTransform(voxel_size[0])

        for fi in range(n_frames):
            dense_blocks = []
            for bi in range(n_blocks):
                arr = coeff_tensor[fi, bi].reshape(blk, blk, blk)
                dense_blocks.append(waverec3(arr, coeff_slices, wavelet))
            grid = insert_leaf_blocks(template_grid, coords, dense_blocks, blk)
            vdb.write(str(out_dir / f"frame_{fi:04d}.vdb"), [grid])

        logging.info("Decompression finished; wrote %d frames to %s.", n_frames, out_dir)

    def compress(self, paths: List[str], out_path: str):
        raise NotImplementedError(
            "Standard VDB-to-compressed path ('compress') is not implemented. Use 'compress-npy'."
        )

    @staticmethod
    def _load_first_grid(path: str) -> vdb.FloatGrid:
        grids = vdb.readAll(path)
        if not grids:
            raise ValueError(f"No grids in {path}")
        return grids[0][0]


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="OpenVDB sequence compressor (v0.3 improved)")
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (once=INFO, twice=DEBUG)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="(stub) Compress from VDB sequence; not implemented")
    c.add_argument("pattern")
    c.add_argument("out_file")
    c.add_argument("--start", type=int, default=0)
    c.add_argument("--end", type=int, required=True)
    c.add_argument("--block", type=int, default=8)
    c.add_argument("--levels", type=int, default=2)
    c.add_argument("--rank", type=int, default=32)

    c_npy = sub.add_parser("compress-npy", help="Compress from pre-processed .npy leaf blocks")
    c_npy.add_argument("npy_pattern", help="Path pattern for leaf data, e.g., 'frames/f%04d.npy'")
    c_npy.add_argument(
        "origin_pattern", help="Path pattern for origin data, e.g., 'frames/f%04d._origins.npy'"
    )
    c_npy.add_argument("out_file", help="Output compressed file path")
    c_npy.add_argument("--start", type=int, default=0, help="Start frame number")
    c_npy.add_argument("--end", type=int, required=True, help="End frame number")
    c_npy.add_argument("--block", type=int, default=8)
    c_npy.add_argument("--levels", type=int, default=2)
    c_npy.add_argument("--rank", type=int, default=32)
    c_npy.add_argument(
        "--grid-name", type=str, required=True, help="Name of the original grid (e.g., 'density')"
    )
    c_npy.add_argument(
        "--voxel-size", type=float, nargs=3, required=True, help="Voxel size (e.g., 0.1 0.1 0.1)"
    )
    c_npy.add_argument(
        "--background", type=float, required=True, help="Background value of the grid (e.g., 0.0)"
    )
    c_npy.add_argument(
        "--refine-cp",
        action="store_true",
        help="Run ALS-based refinement on the two-stage CP initialization to improve fit",
    )

    d = sub.add_parser("decompress", help="Decompress .vdbr file")
    d.add_argument("in_file")
    d.add_argument("out_dir")

    args = ap.parse_args()

    # Setup logging
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    comp = VDBSequenceCompressor(
        block=args.block if hasattr(args, "block") else 8,
        levels=args.levels if hasattr(args, "levels") else 2,
        rank=args.rank if hasattr(args, "rank") else 32,
    )

    if args.cmd == "compress":
        paths = [args.pattern % i for i in range(args.start, args.end + 1)]
        try:
            comp.compress(paths, args.out_file)
        except NotImplementedError as e:
            logging.error("%s", e)
    elif args.cmd == "compress-npy":
        npy_paths = [args.npy_pattern % i for i in range(args.start, args.end + 1)]
        origin_paths = [args.origin_pattern % i for i in range(args.start, args.end + 1)]
        metadata = {
            "grid_name": args.grid_name,
            "voxel_size": args.voxel_size,
            "background": args.background,
        }
        comp = VDBSequenceCompressor(
            block=args.block, levels=args.levels, rank=args.rank, quant_bits=12, wavelet="haar"
        )
        comp.compress_from_npy(
            npy_paths, origin_paths, args.out_file, metadata, refine_cp=args.refine_cp
        )
    else:  # decompress
        comp.decompress(args.in_file, args.out_dir)


if __name__ == "__main__":
    main()
