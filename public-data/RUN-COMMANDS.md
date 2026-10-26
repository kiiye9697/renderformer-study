# Reproduce the public Cornell-box and metal-teapot experiments

Use Windows, Python 3.11, NVIDIA CUDA-capable GPU, and Blender 4.5.1. The scripts default to `F:/RenderFormer-work` for the isolated environment and model cache; adjust those path constants for another machine. Source data remain in the official repository.

```powershell
git clone https://github.com/microsoft/renderformer.git vendor/renderformer
git -C vendor/renderformer checkout eec6a9b8d592303def45e458dd4ebd09d6572abc
./scripts/setup_environment.ps1 -BasePython python -WorkDir F:/RenderFormer-work
# Set IMAGEIO_FREEIMAGE_LIB to the actual downloaded FreeImage DLL.
./scripts/run_reference_suite.ps1
# Wait for reference rendering to finish before measuring inference.
./scripts/run_inference_suite.ps1
F:/RenderFormer-work/venv/Scripts/python.exe scripts/diagnose_network.py --h5 experiments/cbox/scene.h5 --output experiments/cbox/network-diagnostic.json
```

For the additional low-roughness convergence check, launch Blender with `scripts/blender_reference.py`, the `roughness_015/config.json` and its preserved `mesh/scene.obj`, output stem `roughness_015/reference_8192`, `--spp 8192 --seed 2`. Then run `scripts/check_highlight_convergence.py`. This does not replace the primary 4096-sample reference.

The local full audit `scripts/validate_records.py` checks all eight completed cases and requires the separately retained Sponza experiment folders. Sponza meshes, textures, H5 and Blender files are deliberately not distributed; therefore those two source-scene runs cannot be reproduced from this public repository alone. Their rendered A/B evidence, numeric summaries, camera conditions, and file hashes are provided for inspection.

See [environment and protocol](REPRODUCTION.md), [exact package versions](requirements-lock.txt), [model revision and hash](model-lock.json), [two timing rounds](timing-rounds.json), and [eight-case audit](validation-summary.json). GPU timings contain only two warm forwards per round and are not a stable benchmark.
