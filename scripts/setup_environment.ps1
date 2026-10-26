param([string]$BasePython='python',[string]$WorkDir='F:/RenderFormer-work')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force $WorkDir | Out-Null
$env:TEMP=$WorkDir;$env:TMP=$WorkDir;$env:PIP_CACHE_DIR="$WorkDir/pip-cache"
& $BasePython -m venv "$WorkDir/venv"
$python="$WorkDir/venv/Scripts/python.exe"
& $python -m pip install --upgrade pip
# The lock lists transitive dependencies; --no-deps avoids downloading standalone bpy.
# bpy_helper runs inside the existing Blender installation.
& $python -m pip install --no-deps -r "$root/public-data/requirements-lock.txt" --extra-index-url https://download.pytorch.org/whl/cu128
if($LASTEXITCODE -ne 0){throw 'Environment installation failed'}
$lock=Get-Content "$root/public-data/model-lock.json" -Raw|ConvertFrom-Json
$model="$WorkDir/models/renderformer-v1-base"
New-Item -ItemType Directory -Force $model | Out-Null
foreach($file in @('config.json','model.safetensors')){
  Invoke-WebRequest "https://huggingface.co/$($lock.model_id)/resolve/$($lock.model_revision)/$file" -OutFile "$model/$file"
}
$hash=(Get-FileHash "$model/model.safetensors" -Algorithm SHA256).Hash.ToLowerInvariant()
if($hash -ne $lock.model_sha256){throw 'Model SHA256 verification failed'}
$env:IMAGEIO_USERDIR="$WorkDir/imageio"
& $python -c 'import imageio; imageio.plugins.freeimage.download()'
Write-Output 'Installed pinned environment and verified model. Set IMAGEIO_FREEIMAGE_LIB to the downloaded DLL path.'
