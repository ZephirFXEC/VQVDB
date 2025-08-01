from __future__ import annotations

import os

# --- (No changes to imports or DLL paths) ---
os.add_dll_directory("C:/Users/zphrfx/Desktop/hdk/VQVDB/.venv/Lib/site-packages")
os.add_dll_directory("C:/vcpkg/installed/x64-windows/bin")

import argparse
import json
import pathlib
import struct
import logging
from typing import List

import numpy as np
import openvdb as vdb
import pywt
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.decomposition import IncrementalPCA

# Ensure predictable backend
tl.set_backend("numpy")


# ------------------------------------------------------------
# Wavelet helpers (No changes)
# ------------------------------------------------------------


def wavedec3(block: np.ndarray, wavelet: str, level: int):
    coeffs = pywt.wavedecn(block, wavelet=wavelet, level=level, mode="periodization")
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr.astype(np.float32), coeff_slices, coeffs


def waverec3(arr: np.ndarray, coeff_slices, wavelet: str) -> np.ndarray:
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format="wavedecn")
    return pywt.waverecn(coeffs, wavelet=wavelet, mode="periodization")


def quantise(x: np.ndarray, n_bits: int):
    maxval = np.max(np.abs(x))
    scale = ((2 ** (n_bits - 1)) - 1) / maxval if maxval else 1.0
    return np.round(x * scale).astype(np.int16), scale


def dequantise(q: np.ndarray, scale: float):
    return q.astype(np.float32) / scale


def compute_rmse(a: np.ndarray, b: np.ndarray):
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    return float(np.sqrt(np.mean((a_f - b_f) ** 2)))


def _pad_init_cp(init_weights, init_factors, target_rank: int):
    """
    Pads a low-rank CP initial guess up to `target_rank` and re-normalises it.

    Returns
    -------
    (weights, factors) : tuple
        A properly sized CP decomposition with unit-norm columns.
    """
    init_rank = init_weights.shape[0]

    # If already large enough, truncate / normalise and return.
    if init_rank >= target_rank:
        trimmed = (
            init_weights[:target_rank],
            [f[:, :target_rank] for f in init_factors],
        )
        return cp_normalize(trimmed)

    padding_rank = target_rank - init_rank
    ctx = tl.context(init_weights)

    # ---- pad weights -------------------------------------------------
    final_weights = tl.zeros(target_rank, **ctx)
    final_weights[:init_rank] = init_weights
    final_weights[init_rank:] = tl.mean(init_weights) if init_rank else 1.0

    # ---- pad factors -------------------------------------------------
    final_factors = []
    for F in init_factors:
        newF = tl.zeros((F.shape[0], target_rank), **ctx)
        newF[:, :init_rank] = F

        rnd = np.random.random((F.shape[0], padding_rank)).astype(F.dtype)
        # scale random columns to average norm of existing ones
        avg_norm = tl.mean(tl.norm(F, axis=0)) if init_rank else 1.0
        newF[:, init_rank:] = rnd * avg_norm
        final_factors.append(newF)

    # Ensure unit-norm columns so ALS starts stable
    return cp_normalize((final_weights, final_factors))


# ---------------------------------------------------------------------
# 2.  Hybrid direct CP compression (now stable & overflow-safe)
# ---------------------------------------------------------------------
def _hybrid_direct_cp_compress(
    q_tensor: np.ndarray, rank: int, n_iter_max: int = 100, dtype=np.float32
):
    """
    Robust, memory-safe CP via two-stage init  ➜  full PARAFAC.

    * The quantised tensor is first **whitened** to max-abs ≤ 1 to
      avoid float32 overflow inside ALS.
    * Factor columns are re-normalised every iteration
      (`normalize_factors=True`) so growth is pushed into `weights`.
    """
    # ---------- 0 · Rescale tensor to [-1, 1] -------------------------
    scale_back = np.max(np.abs(q_tensor))
    scale_back = 1.0 if scale_back == 0 else scale_back
    tensor = (q_tensor / scale_back).astype(dtype)

    n_frames = tensor.shape[0]
    init_rank = min(rank - 1, n_frames - 2, 32)

    # ---------- 1 · Obtain low-rank seed -----------------------------
    if init_rank <= 0:
        init_method = "random"
    else:
        try:
            w0, f0 = _two_stage_cp_compress(q_tensor, rank=init_rank)
            init_method = _pad_init_cp(w0, f0, rank)
        except Exception:
            init_method = "random"

    # ---------- 2 · Full PARAFAC with safe settings ------------------
    weights, factors = parafac(
        tensor,
        rank=rank,
        init=init_method,
        n_iter_max=n_iter_max,
        tol=1e-7,
        normalize_factors=True,  # ← keeps column norms ≈ 1
        verbose=False,
    )

    # ---------- 3 · Undo whitening ----------------------------------
    weights = weights * scale_back
    return weights, factors


# ---------------------------------------------------------------------
# 3.  Optional CP refinement (just passes extra kwarg)
# ---------------------------------------------------------------------
def _refine_cp(
    q_tensor: np.ndarray, weights, factors, n_iter: int = 10, tol: float = 1e-6
):
    """
    Light ALS sweep to polish an existing CP solution.
    """
    tensor = q_tensor.astype(np.float32)
    rank = weights.shape[0]

    try:
        weights, factors = parafac(
            tensor,
            rank=rank,
            init=(weights, factors),
            n_iter_max=n_iter,
            tol=tol,
            normalize_factors=True,  # ← same safety guard
            verbose=False,
        )
    except Exception:
        # keep original on failure
        pass

    return weights, factors


def _apply_coefficient_thresholding(
    tensor: np.ndarray, keep_percent: float
) -> np.ndarray:
    """
    Keeps only the largest `keep_percent` of wavelet coefficients, setting
    the rest to zero. This is a key step for controlling loss vs. compression.
    """
    if keep_percent >= 100.0:
        logging.debug("keep_percent is 100; skipping thresholding.")
        return tensor

    logging.info(
        "Applying coefficient thresholding to keep top %.2f%%...", keep_percent
    )

    # Get absolute values of all coefficients
    abs_coeffs = np.abs(tensor.ravel())

    # We only want to consider non-zero coefficients for the percentile calculation
    # to avoid skewing by the existing zeros.
    abs_coeffs_nonzero = abs_coeffs[abs_coeffs > 1e-9]
    if abs_coeffs_nonzero.size == 0:
        logging.warning("Tensor is all zeros; skipping thresholding.")
        return tensor

    # Calculate the threshold value.
    # e.g., if keep_percent is 80, we want to find the 20th percentile.
    percentile_to_cut = 100.0 - keep_percent
    threshold = np.percentile(abs_coeffs_nonzero, percentile_to_cut)

    logging.info(
        f"Calculated threshold: {threshold:.6f}. Setting coefficients smaller than this to zero."
    )

    # Apply the threshold
    tensor_thresholded = tensor.copy()
    tensor_thresholded[np.abs(tensor_thresholded) < threshold] = 0.0

    sparsity = 100.0 * np.count_nonzero(tensor_thresholded) / tensor_thresholded.size
    logging.info(
        f"Thresholding complete. Tensor sparsity is now {sparsity:.2f}% (non-zero elements)."
    )

    return tensor_thresholded


# ------------------------------------------------------------
# CP helpers
# ------------------------------------------------------------


def _two_stage_cp_compress(q_tensor: np.ndarray, rank: int, batch_size: int = 256):
    """
    Memory-efficient two-stage approximate CP (Kruskal) decomposition.
    Limited by rank < n_frames.
    Returns: (weights, [A, B, C]) where rank returned is original rank+1.
    """
    n_frames, n_blocks, coeff_len = q_tensor.shape

    logging.debug(
        "Reshaping tensor to (%d, %d) for PCA...", n_frames, n_blocks * coeff_len
    )
    flat_tensor = q_tensor.reshape(n_frames, -1).astype(np.float32)

    # --- (rest of this function is unchanged) ---
    ipca = IncrementalPCA(n_components=rank, batch_size=batch_size)

    logging.info("Running Incremental PCA to extract temporal factor...")
    A = ipca.fit_transform(flat_tensor).astype(np.float32)  # (n_frames, rank)
    combined_factors = ipca.components_.T.astype(
        np.float32
    )  # (n_blocks*coeff_len, rank)
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
    weights = np.concatenate(
        [weights, np.array([weight_mean], dtype=np.float32)], axis=0
    )
    A = np.concatenate([A, A_mean], axis=1)  # now (n_frames, rank+1)
    B = np.concatenate([B, B_mean[:, None]], axis=1)  # (n_blocks, rank+1)
    C = np.concatenate([C, C_mean[:, None]], axis=1)  # (coeff_len, rank+1)

    logging.info(
        "Two-stage CP compression produced rank %d (including mean).", weights.shape[0]
    )
    return weights, [A, B, C]


def _direct_cp_compress(q_tensor: np.ndarray, rank: int, n_iter_max: int = 100):
    """
    Performs a direct CP decomposition using Tensorly's PARAFAC.
    More memory intensive but not limited by n_frames for the rank.
    """
    logging.info("Performing direct CP compression (PARAFAC) with rank=%d...", rank)
    tensor = q_tensor.astype(np.float32)
    try:
        weights, factors = parafac(
            tensor, rank=rank, n_iter_max=n_iter_max, tol=1e-7, verbose=0
        )
        logging.info("Direct CP compression succeeded.")
        return weights, factors
    except Exception as e:
        logging.error("Direct CP compression failed: %s", e)
        raise


def cp_decompress(weights: np.ndarray, factors):
    return tl.cp_to_tensor((weights, factors))


# ------------------------------------------------------------
# Leaf block insertion & Header (No changes)
# ------------------------------------------------------------
def insert_leaf_blocks(
    grid_template: vdb.FloatGrid, coords, blocks, block_size: int = 8
):
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


MAGIC = b"VDBR\0"
HEADER_STRUCT = struct.Struct(
    "<5sHHHHHI"
)  # magic, vmaj, vmin, blk, lvl, rank, n_frames
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
        # --- NEW parameters with defaults ---
        keep_coeffs_percent=100.0,
        cp_method="direct",
    ):
        self.block = block
        self.levels = levels
        self.rank = rank
        self.quant_bits = quant_bits
        self.wavelet = wavelet
        # --- NEW parameters ---
        self.keep_coeffs_percent = keep_coeffs_percent
        self.cp_method = cp_method

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

        logging.info(
            "Found %d unique blocks across %d frames.", n_total_blocks, n_frames
        )

        logging.info("Determining wavelet coefficient length...")
        logging.debug(
            "Shape of dummy block: (%d, %d, %d)", self.block, self.block, self.block
        )
        dummy_block = np.zeros((self.block, self.block, self.block), dtype=np.float32)
        dummy_coeffs, _, _ = wavedec3(dummy_block, self.wavelet, self.levels)
        coeff_len = dummy_coeffs.size
        logging.info("Wavelet coefficient length is %d.", coeff_len)

        final_coeff_tensor = np.zeros(
            (n_frames, n_total_blocks, coeff_len), dtype=np.float32
        )

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

        coeff_tensor_thresholded = _apply_coefficient_thresholding(
            coeff_tensor, self.keep_coeffs_percent
        )

        # 2. Quantisation
        q_tensor, scale = quantise(coeff_tensor_thresholded, self.quant_bits)
        logging.info(
            "Quantisation complete. Bits=%d. Quantization RMSE (wavelet domain): %.6f",
            self.quant_bits,
            compute_rmse(coeff_tensor_thresholded, dequantise(q_tensor, scale)),
        )

        # 3. CP Decomposition (with choice of method)
        if self.cp_method == "two-stage":
            if self.rank >= n_frames:
                logging.warning(
                    f"Rank ({self.rank}) must be less than n_frames ({n_frames}) for 'two-stage' method. "
                    f"Consider using '--cp-method direct' or reducing the rank."
                )
                self.rank = n_frames - 1
                logging.warning(f"Clamping rank to {self.rank}.")

            logging.info(
                "Performing two-stage CP compression on quantized tensor of shape %s...",
                q_tensor.shape,
            )
            weights, factors = _two_stage_cp_compress(q_tensor, self.rank)
            A, B, C = factors
        elif self.cp_method == "direct":
            logging.info(
                "Performing direct CP compression on quantized tensor of shape %s...",
                q_tensor.shape,
            )
            weights, factors = _hybrid_direct_cp_compress(q_tensor, self.rank)
            A, B, C = factors
        else:
            raise ValueError(f"Unknown cp_method: {self.cp_method}")

        # CP approximation error before refinement
        approx_q = cp_decompress(weights, [A, B, C])
        rmse_before = compute_rmse(q_tensor, approx_q)
        logging.info(
            "CP approximation RMSE before refinement: %.6f (on quantized tensor)",
            rmse_before,
        )

        # 4. Optional refinement
        if refine_cp:
            weights_refined, factors_refined = _refine_cp(q_tensor, weights, [A, B, C])
            A, B, C = factors_refined
            weights = weights_refined
            approx_q_refined = cp_decompress(weights, [A, B, C])
            rmse_after = compute_rmse(q_tensor, approx_q_refined)
            logging.info(
                "CP approximation RMSE after refinement: %.6f (on quantized tensor)",
                rmse_after,
            )
        else:
            logging.debug("Skipping CP refinement (disabled).")

        logging.info("Final CP rank stored: %d", weights.shape[0])
        logging.info("Factor shapes: A=%s, B=%s, C=%s", A.shape, B.shape, C.shape)

        # --- (Rest of the function is the same, just writes the data) ---

        # Metadata augmentation
        if not all(
            k in grid_metadata for k in ["grid_name", "voxel_size", "background"]
        ):
            raise ValueError(
                "grid_metadata must contain 'grid_name', 'voxel_size', and 'background'"
            )

        grid_metadata["wavelet"] = self.wavelet
        grid_metadata["quant_bits"] = self.quant_bits

        meta_bytes = json.dumps(grid_metadata).encode()

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

        # A more fair comparison for uncompressed size
        uncompressed_size = coeff_tensor.nbytes
        ratio = uncompressed_size / compressed_size if compressed_size else float("inf")
        logging.info(
            "Successfully compressed %d frames to %s. Size: %.2f MB",
            n_frames,
            out_path,
            compressed_size / (1024 * 1024),
        )
        logging.info(
            "Uncompressed wavelet coeffs size: %.2f MB. Compression Ratio: %.2f:1",
            uncompressed_size / (1024 * 1024),
            ratio,
        )

    # --- decompress and other methods are unchanged ---
    def decompress(self, in_path: str, out_dir: str):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(in_path, "rb") as fp:
            magic, vmaj, vmin, blk, lvl, rank, n_frames = HEADER_STRUCT.unpack(
                fp.read(HEADER_STRUCT.size)
            )
            if magic != MAGIC:
                raise ValueError("Invalid VDBR file")
            scale, coeff_len, n_blocks, json_bytes = META_STRUCT.unpack(
                fp.read(META_STRUCT.size)
            )
            meta = json.loads(fp.read(json_bytes))
            coords = np.frombuffer(fp.read(n_blocks * 3 * 4), dtype=np.int32).reshape(
                n_blocks, 3
            )
            weights = np.frombuffer(fp.read(rank * 4), dtype=np.float32)
            A = np.frombuffer(fp.read(n_frames * rank * 4), dtype=np.float32).reshape(
                n_frames, rank
            )
            B = np.frombuffer(fp.read(n_blocks * rank * 4), dtype=np.float32).reshape(
                n_blocks, rank
            )
            C = np.frombuffer(fp.read(coeff_len * rank * 4), dtype=np.float32).reshape(
                coeff_len, rank
            )

        # Reconstruct quantized tensor
        logging.info("Reconstructing quantized tensor via CP inverse...")
        q_matrix = cp_decompress(weights, [A, B, C])  # (n_frames, n_blocks, coeff_len)
        q_tensor = (
            np.round(q_matrix).astype(np.int16).reshape(n_frames, n_blocks, coeff_len)
        )
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

        logging.info(
            "Decompression finished; wrote %d frames to %s.", n_frames, out_dir
        )

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
    ap = argparse.ArgumentParser(
        description="OpenVDB sequence compressor (v0.4, with quality controls)"
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (once=INFO, twice=DEBUG)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser(
        "compress", help="(stub) Compress from VDB sequence; not implemented"
    )
    c.add_argument("pattern")
    c.add_argument("out_file")
    c.add_argument("--start", type=int, default=0)
    c.add_argument("--end", type=int, required=True)
    c.add_argument("--block", type=int, default=8)
    c.add_argument("--levels", type=int, default=2)
    c.add_argument("--rank", type=int, default=32)

    c_npy = sub.add_parser(
        "compress-npy", help="Compress from pre-processed .npy leaf blocks"
    )
    c_npy.add_argument(
        "npy_pattern", help="Path pattern for leaf data, e.g., 'frames/f%04d.npy'"
    )
    c_npy.add_argument(
        "origin_pattern",
        help="Path pattern for origin data, e.g., 'frames/f%04d._origins.npy'",
    )
    c_npy.add_argument("out_file", help="Output compressed file path")
    c_npy.add_argument("--start", type=int, default=0, help="Start frame number")
    c_npy.add_argument("--end", type=int, required=True, help="End frame number")

    # --- MODIFIED: Added more control knobs ---
    g_quality = c_npy.add_argument_group("Quality and Compression Controls")
    g_quality.add_argument("--block", type=int, default=8)
    g_quality.add_argument("--levels", type=int, default=2)
    g_quality.add_argument(
        "--rank",
        type=int,
        default=32,
        help="Rank for the CP decomposition. Higher means better quality and larger file size.",
    )
    g_quality.add_argument(
        "--quant-bits",
        type=int,
        default=12,
        help="Number of bits for quantizing wavelet coeffs. (e.g., 8-16). Lower is smaller/lossier.",
    )
    g_quality.add_argument(
        "--keep-coeffs",
        type=float,
        default=100.0,
        help="Percentage of wavelet coefficients to keep (0-100). Lower is smaller/lossier.",
    )
    g_quality.add_argument(
        "--cp-method",
        choices=["two-stage", "direct"],
        default="two-stage",
        help="CP decomposition method. 'direct' is slower but allows rank > n_frames.",
    )

    g_meta = c_npy.add_argument_group("Grid Metadata (Required)")
    g_meta.add_argument(
        "--grid-name",
        type=str,
        required=True,
        help="Name of the original grid (e.g., 'density')",
    )
    g_meta.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        required=True,
        help="Voxel size (e.g., 0.1 0.1 0.1)",
    )
    g_meta.add_argument(
        "--background",
        type=float,
        required=True,
        help="Background value of the grid (e.g., 0.0)",
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

    if args.cmd == "decompress":
        comp = VDBSequenceCompressor()  # Use default params for decompression
        comp.decompress(args.in_file, args.out_dir)
        return

    # --- MODIFIED: Pass all new args to the compressor ---
    comp = VDBSequenceCompressor(
        block=args.block,
        levels=args.levels,
        rank=args.rank,
        quant_bits=args.quant_bits if hasattr(args, "quant_bits") else 12,
        keep_coeffs_percent=args.keep_coeffs if hasattr(args, "keep_coeffs") else 100.0,
        cp_method=args.cp_method if hasattr(args, "cp_method") else "two-stage",
    )

    if args.cmd == "compress":
        paths = [args.pattern % i for i in range(args.start, args.end + 1)]
        try:
            comp.compress(paths, args.out_file)
        except NotImplementedError as e:
            logging.error("%s", e)
    elif args.cmd == "compress-npy":
        npy_paths = [args.npy_pattern % i for i in range(args.start, args.end + 1)]
        origin_paths = [
            args.origin_pattern % i for i in range(args.start, args.end + 1)
        ]
        metadata = {
            "grid_name": args.grid_name,
            "voxel_size": args.voxel_size,
            "background": args.background,
        }
        comp.compress_from_npy(
            npy_paths, origin_paths, args.out_file, metadata, refine_cp=args.refine_cp
        )


if __name__ == "__main__":
    main()
