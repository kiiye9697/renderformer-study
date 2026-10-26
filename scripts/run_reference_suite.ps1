$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$python='F:/RenderFormer-work/venv/Scripts/python.exe'
$blender='C:/Program Files/Blender Foundation/Blender 4.5/blender.exe'
$env:TEMP='F:/RenderFormer-work'; $env:TMP=$env:TEMP
$vendor="$root/vendor/renderformer"
New-Item -ItemType Directory -Force "$root/experiments/teapot/geometry","$root/runtime" | Out-Null
# The geometry is remeshed once, then shared by all five cases; no per-case remesh.
& $python "$vendor/scene_processor/remesh.py" --input "$vendor/examples/objects/classical/teapot.obj" --target_face_num 2048 --output "$root/experiments/teapot/geometry/teapot_2048.obj"
if($LASTEXITCODE -ne 0) {throw 'Teapot remesh failed'}
& $python "$PSScriptRoot/prepare_teapot.py"
$cases=@(@{Name='cbox';Dir="$root/experiments/cbox";Config="$vendor/examples/cbox.json";Ref='reference_4096'})
foreach($name in @('baseline','roughness_015','roughness_075','wall_000','wall_050')) {
  $cases+=@{Name=$name;Dir="$root/experiments/teapot/$name";Config="$root/experiments/teapot/$name/config.json";Ref='reference'}
}
foreach($case in $cases) {
  & $python "$vendor/scene_processor/convert_scene.py" $case.Config --mesh_path "$($case.Dir)/mesh/scene.obj" --output_h5_path "$($case.Dir)/scene.h5"
  if($LASTEXITCODE -ne 0) {throw "Conversion failed $($case.Name)"}
  & $blender -b --factory-startup --python "$PSScriptRoot/blender_reference.py" -- --config $case.Config --mesh "$($case.Dir)/mesh/scene.obj" --output "$($case.Dir)/$($case.Ref)" --spp 4096 --seed 1
  if(!(Test-Path "$($case.Dir)/$($case.Ref).exr")) {throw "Reference missing $($case.Name)"}
}
# Repeat two representative scenes with fewer samples and a different RNG seed.
foreach($case in @($cases[0],$cases[1])) {
  & $blender -b --factory-startup --python "$PSScriptRoot/blender_reference.py" -- --config $case.Config --mesh "$($case.Dir)/mesh/scene.obj" --output "$($case.Dir)/reference_1024" --spp 1024 --seed 0
}
& $blender -b --factory-startup --python "$PSScriptRoot/check_blender_cameras.py"
