"""Prepare held-out material configurations for Bunny and a reduced Sponza."""
import copy
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import trimesh

ROOT = Path(os.environ['RENDERFORMER_WORKSPACE'])
REPO = ROOT / 'repos/renderformer-eec6a9b8d592303def45e458dd4ebd09d6572abc'
OUT = ROOT / 'datasets/training'
OUT.mkdir(exist_ok=True)
asset = ROOT / 'datasets/sponza_source'
asset.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT / 'archives/sponza-gltf.zip') as z:
    z.extractall(asset)
scene = trimesh.load(asset / 'Sponza.gltf', force='scene')
bounds = scene.bounds
center = bounds.mean(axis=0)
scale = 2.0 / np.max(bounds[1] - bounds[0])
objects = {}
total = sum(len(scene.geometry[scene.graph[n][1]].faces) for n in scene.graph.nodes_geometry)
original_faces = 0
reduced_faces = 0
for i, node in enumerate(scene.graph.nodes_geometry):
    transform, key = scene.graph[node]
    mesh = scene.geometry[key].copy()
    mesh.apply_transform(transform)
    original_faces += len(mesh.faces)
    material = getattr(mesh.visual, 'material', None)
    color = np.array(getattr(material, 'baseColorFactor', [180, 180, 180, 255])[:3], dtype=float) / 255
    texture = getattr(material, 'baseColorTexture', None)
    if texture is not None:
        # Explicitly reduced material representation: mean base color, no alpha/normal maps.
        color *= np.asarray(texture.convert('RGB').resize((32, 32))).mean(axis=(0, 1)) / 255
    target = max(16, int(6000 * len(mesh.faces) / total))
    if len(mesh.faces) > target:
        mesh = mesh.simplify_quadric_decimation(face_count=target)
    # glTF is Y up; RenderFormer example scenes are Z up.
    v = (mesh.vertices - center) * scale
    mesh.vertices = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    path = OUT / f'sponza_part_{i:03d}.obj'
    mesh.export(path)
    reduced_faces += len(mesh.faces)
    objects[f'part_{i:03d}'] = dict(mesh_path=path.name,
        transform=dict(translation=[0., 0., 0.], rotation=[0., 0., 0.], scale=[1., 1., 1.], normalize=False),
        material=dict(diffuse=color.tolist(), specular=[0.04]*3, roughness=0.6,
                      emissive=[0., 0., 0.], smooth_shading=True))
bunny = json.loads((REPO / 'examples/cbox-bunny.json').read_text())
for obj in bunny['objects'].values():
    obj['mesh_path'] = os.path.relpath(REPO / 'examples' / obj['mesh_path'], OUT)
light = copy.deepcopy(bunny['objects']['light_0'])
light['transform']['translation'] = [0., 0., 0.25]
light['transform']['scale'] = [4., 4., 4.]
light['material']['emissive'] = [5000., 5000., 5000.]
objects['light_0'] = light
sponza = dict(scene_name='Sponza reduced constant-material study', version='1.0', objects=objects,
              cameras=[dict(position=[0.65, -0.02, -0.08], look_at=[-0.5, 0., -0.05], up=[0., 0., 1.], fov=65.)])
records = []
for family, base, key in [('bunny', bunny, 'bunny'), ('sponza', sponza, None)]:
    for split, roughness in [('train', 0.15), ('train', 0.5), ('train', 0.9), ('validation', 0.3), ('test', 0.7)]:
        cfg = copy.deepcopy(base)
        for name, obj in cfg['objects'].items():
            if (key is not None and name == key) or (key is None and name != 'light_0'):
                obj['material']['roughness'] = roughness
        # Entire material configurations are held out; views are never split randomly.
        camera = cfg['cameras'][0]
        cameras = []
        for offset in [-0.04, 0., 0.04]:
            cam = copy.deepcopy(camera)
            cam['position'][0 if family == 'bunny' else 1] += offset
            cameras.append(cam)
        cfg['cameras'] = cameras
        name = f'{family}_{split}_r{roughness}'
        path = OUT / f'{name}.json'
        path.write_text(json.dumps(cfg, indent=2))
        records.append(dict(name=name, family=family, split=split, roughness=roughness, config=str(path),
                            h5=str(OUT / (name + '.h5')), reference_dir=str(ROOT / 'results/training_reference' / name)))
(OUT / 'manifest.json').write_text(json.dumps(dict(records=records, resolution=128, spp=512,
    sponza_original_faces=original_faces, sponza_reduced_faces=reduced_faces,
    sponza_limitations='Simplified geometry; texture mean colors; no alpha, normal maps or original PBR materials. Both reference and prediction use this representation.'), indent=2))
print('Prepared', len(records), 'configurations; Sponza faces', original_faces, '->', reduced_faces)
