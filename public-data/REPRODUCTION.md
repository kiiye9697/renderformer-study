# Local runtime and reproducibility

- Python 3.11 isolated environment: `<work-dir>/venv`.
- PyTorch 2.8.0+cu128, Windows SDPA, fp16 autocast. No FlashAttention or Liger.
- Blender 4.5.1 LTS. Launch with `--factory-startup` to avoid translated node names and user preferences. Reference shaders and cameras use official `bpy_helper==0.0.8`.
- Code and model versions/hash: `model-lock.json`; installed packages: `requirements-lock.txt`.
- `bpy-helper` declares `bpy` as a dependency; `pip check` reports it missing because the reference script runs **inside installed Blender**, not a separate pip bpy runtime. All model/scene-converter imports use the isolated Python environment.
- Original direct `to_blend` import crashed inside Blender when importing the converter's pymeshlab DLLs. The reference adapter imports only official material/camera helpers and the already converted OBJ meshes. Vendor code is unchanged.
- Official converter quantizes diffuse RGB through 8-bit OBJ vertex colors. The adapter matches `floor(255*d)/255` in Cycles. Specular/roughness/emission follow the JSON.
- Blender and H5 triangle vertices are independently compared, with camera/FOV and vertex normals recorded in each `geometry-validation.json`. Float normal differences arise at OBJ import precision.

## Commands

1. Clone official code into `vendor/renderformer` and checkout the commit in `model-lock.json`.
2. Install GPU PyTorch into the isolated environment, then the packages in `requirements-lock.txt` (Blender supplies bpy). Download only `config.json` and `model.safetensors` at the locked Hugging Face revision, and verify SHA256.
3. Run `scripts/run_reference_suite.ps1`. It remeshes the official Utah teapot once to 2048 faces and shares that asset across all five parameter cases. `prepare_teapot.py` saves the complete scene descriptions and predefined image ROIs.
4. Wait for all Cycles work to stop, then run `scripts/run_inference_suite.ps1`. Each case records one cold forward and two warmed forwards with CUDA synchronization, separate from model loading. Texture tensors are cloned because the official pipeline modifies emission channels in place.
5. Run `diagnose_network.py --h5 ... --output ...` separately from timing. Its activation shape/statistics capture is a real forward pass, not an attention or physical irradiance measurement.

## Output conventions

A is Cycles; B is RenderFormer. Linear float EXR is retained. Display images use the same EV=0, channel-wise Reinhard mapping, and IEC sRGB transfer for both images. PSNR and SSIM use these float RGB display values with data_range=1, full image and no crop; linear HDR MAE/RMSE are also recorded. Error maps share limits [0, 0.25]. These are project-defined metrics, not a claim of reproducing the paper's exact evaluation protocol.

The final teapot has diffuse=[0,0,0], white specular=[0.9,0.9,0.9]: an achromatic metal approximation under the supported additive GGX material. Old mixed diffuse/specular reference renders are retained under `previous_dielectric_not_final` and excluded from final results.

Sponza geometry, H5 and blend files remain outside this repository at `<work-dir>`; they are not distributable project source. Only user-authorized rendered images and numeric records may be copied to the public showcase, with explicit provenance and simplification caveats.

