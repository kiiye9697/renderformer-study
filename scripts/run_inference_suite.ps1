$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$python='F:/RenderFormer-work/venv/Scripts/python.exe'
$env:ATTN_IMPL='sdpa'
$env:IMAGEIO_FREEIMAGE_LIB='F:/RenderFormer-work/imageio/freeimage-3.18.0-win64.dll'
$env:HF_HOME='F:/RenderFormer-work/huggingface'
$cases=@(@{Name='cbox';Dir="$root/experiments/cbox";Ref='reference_4096.exr'})
foreach($name in @('baseline','roughness_015','roughness_075','wall_000','wall_050')) {
  $cases+=@{Name=$name;Dir="$root/experiments/teapot/$name";Ref='reference.exr'}
}
foreach($case in $cases) {
  & $python "$PSScriptRoot/infer_measured.py" --h5 "$($case.Dir)/scene.h5" --output "$($case.Dir)/prediction" *> "$root/runtime/infer-$($case.Name).log"
  if($LASTEXITCODE -ne 0) {throw "Inference failed: $($case.Name)"}
  & $python "$PSScriptRoot/evaluate.py" --reference "$($case.Dir)/$($case.Ref)" --prediction "$($case.Dir)/prediction/prediction.exr" --output "$($case.Dir)/results" --title "$($case.Name) | 512 x 512 | fp16 SDPA" *> "$root/runtime/evaluate-$($case.Name).log"
  if($LASTEXITCODE -ne 0) {throw "Evaluation failed: $($case.Name)"}
  Write-Output "Completed $($case.Name)"
}
& $python "$PSScriptRoot/parameter_curves.py"
