"""Run in Blender to record saved reference cameras for independent H5 comparison."""
from pathlib import Path
import bpy, json, sys
root=Path(__file__).resolve().parents[1]/'experiments'
files=[root/'cbox/reference_4096.blend']+list((root/'teapot').glob('*/reference.blend'))
if '--' in sys.argv:
    files=[Path(x) for x in sys.argv[sys.argv.index('--')+1:]]
for file in files:
    bpy.ops.wm.open_mainfile(filepath=str(file))
    cam=bpy.context.scene.camera
    (file.parent/'blender-camera.json').write_text(json.dumps({
      'c2w':[list(row) for row in cam.matrix_world], 'fov_radians':cam.data.angle,
      'sensor_fit':cam.data.sensor_fit, 'resolution':[bpy.context.scene.render.resolution_x,bpy.context.scene.render.resolution_y]},indent=2))
