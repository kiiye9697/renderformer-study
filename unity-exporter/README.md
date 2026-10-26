# Restricted Unity static scene exporter

Install this folder as a local UPM package. Unity 2021.3+ Editor only. Select static roots, tag the intended camera MainCamera, then Tools > RenderFormer > Export Selected Static Roots. Choose a parent directory; a fresh timestamped package is created. Each submesh receives one OBJ and an entry in official scene.json.

## Supported subset

- Active enabled MeshRenderer + MeshFilter, readable triangle mesh with vertex normals, one material per submesh.
- Constant opaque URP/Lit material, backface culling, no textures/property blocks; metallic workflow only with metallic=0 or explicit monochromatic specular workflow.
- Square perspective camera without lens shift/physical camera; Unity vertical FOV equals both output axes for square images.
- White emissive meshes. Native Unity light components in selected roots cause rejection. External Unity lights, sky, ambient, probes, baked lighting and postprocessing are not exported; the result is a controlled scene package, not original engine appearance.

World positions are baked, converting Unity `(x,y,z)` to right-handed experiment `(x,z,y)`. Normals use inverse transpose and normalization. Face reversal depends on total parity, including negative object scale. Nonuniform scales are supported; singular transforms are rejected. Multi-submesh meshes are split without remeshing.

Material conversion is explicitly approximate: diffuse = baseColor*(1-specular), roughness = 1-smoothness. URP and reference BRDF implementations are not guaranteed identical. The supported final roughness range is [0.01,1]. No arbitrary NTC shader, texture, transparency, particle, skinned animation, point/directional light or original shading equivalence is claimed. Existing material textures must be deliberately replaced by constants in an independent test copy.

## Important normal limitation

The OBJ preserves exported transformed vertex normals. Official RenderFormer conversion calls smooth_shade or rebuilds face normals and may modify them. Use identical *converted* geometry/normals for both model and Cycles. Do not claim original Unity normal fidelity after that stage. Quantitative comparison must record this boundary.

## Validation

`Tests~/ExporterValidation.cs` is a standalone batch harness: copy to an isolated test project's Assets/Editor, set its output path as needed, invoke `-executeMethod ExporterValidation.Run`. It checks coordinate/winding parity including negative scale, inverse-transpose normal transform, actual OBJ/JSON export and unsupported texture rejection. This is not a full Unity renderer test. Current compile/runtime evidence belongs to runtime/unity-export-validation.log and the independent test result, not to this README.

Original Sponza/NTC project must never be opened or modified for this test. Imported FBX, copied materials, original scene screenshots and other third-party assets remain local and are excluded from the public package.
