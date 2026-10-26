"""Run inside Blender; uses official material/camera helper and preserved converter OBJ.
Example: blender -b --python scripts/blender_reference.py -- --config ... --mesh ... --output ...
"""
import sys, os, argparse, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append('F:/RenderFormer-work/venv/Lib/site-packages')
sys.path.insert(0, str(ROOT / 'vendor/renderformer/scene_processor'))
import bpy
import numpy as np
from bpy_helper.camera import create_camera, look_at_to_c2w
from bpy_helper.material import create_specular_roughness_material, create_white_emmissive_material

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--mesh', required=True)
p.add_argument('--output', required=True)
p.add_argument('--spp', type=int, default=1024)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--resolution', type=int, default=512)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
os.makedirs(Path(a.output).parent, exist_ok=True)
cfg = json.loads(Path(a.config).read_text())
# Build from preserved split OBJ, never regenerate/remesh a second time.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
triangles, normals = [], []
for name, obj_cfg in cfg['objects'].items():
    bpy.ops.wm.obj_import(filepath=str(Path(a.mesh).parent/'split'/f'{name}.obj'))
    obj = bpy.context.object
    mat = obj_cfg['material']
    if mat['emissive'][0] > 0:
        material = create_white_emmissive_material(strength=mat['emissive'][0],material_name=name)
    else:
        material = create_specular_roughness_material(tuple(mat['diffuse']),tuple(mat['specular']),mat['roughness'],name)
    obj.data.materials.clear(); obj.data.materials.append(material)
    obj.rotation_mode = 'XYZ'; obj.rotation_euler = (0,0,0)
    bpy.context.view_layer.update()
    obj.data.calc_loop_triangles()
    for tri in obj.data.loop_triangles:
        triangles.append([tuple(obj.matrix_world @ obj.data.vertices[i].co) for i in tri.vertices])
        normals.append([tuple(obj.data.corner_normals[i].vector) for i in tri.loops])
np.savez(a.output + '.geometry.npz', triangles=np.asarray(triangles), vn=np.asarray(normals))
# Official converter stores constant diffuse through 8-bit OBJ vertex colors.
# Match this quantization in Cycles instead of the original unquantized JSON.
for name, obj_cfg in cfg['objects'].items():
    mat = obj_cfg['material']
    if not mat['emissive'][0] and mat.get('rand_tri_diffuse_seed') is None:
        value = (np.array(mat['diffuse']) * 255).clip(0,255).astype(int) / 255
        bpy.data.materials[name].node_tree.nodes['Group'].inputs['Diffuse'].default_value = tuple(value)
scene = bpy.context.scene
cam = cfg['cameras'][0]
scene.camera = create_camera(look_at_to_c2w(np.array(cam['position']),np.array(cam['look_at']),np.array(cam['up'])),cam['fov'])
scene.render.engine = 'CYCLES'
scene.render.resolution_x = scene.render.resolution_y = a.resolution
scene.render.resolution_percentage = 100
scene.cycles.samples = a.spp
scene.cycles.device = 'GPU'
scene.render.film_transparent = True
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[1].default_value = 0
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_mode = 'RGBA'
scene.cycles.seed = a.seed
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == 'CUDA'
scene.render.image_settings.color_depth = '32'
scene.render.image_settings.exr_codec = 'ZIP'
scene.render.filepath = str(Path(a.output).with_suffix('.exr'))
start = time.perf_counter()
bpy.ops.render.render(write_still=True)
elapsed = time.perf_counter() - start
bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.output).with_suffix('.blend')))
Path(a.output + '.json').write_text(json.dumps({
    'blender_version': bpy.app.version_string, 'samples': a.spp, 'seed': a.seed,
    'denoising': False, 'adaptive_sampling': False, 'seconds': elapsed,
    'resolution': a.resolution, 'mesh': a.mesh, 'config': a.config,
    'linear_hdr': True, 'reference_implementation': 'official bpy_helper==0.0.8 materials and camera; preserved converter split OBJ import',
    'diffuse_alignment': 'constant colors quantized by floor(255*diffuse)/255 to match official OBJ/H5 converter',
}, indent=2))
