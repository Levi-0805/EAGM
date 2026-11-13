# Research trajectory datasets

Four Stage7C.2 / Stage8 trajectory tensors exceed GitHub's normal per-file size limit. These files are stored losslessly as ordered 40 MiB parts. Smaller checkpoints are stored at their original run paths.

From the repository root, run `python scripts/restore_research_data.py` to reconstruct the original `.pt` paths. The script checks SHA-256 for every part and reconstructed file, and never overwrites an existing file. Original dataset bytes and experimental protocols are unchanged. `manifest.json` records paths, sizes and hashes.
