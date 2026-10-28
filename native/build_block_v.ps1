param([string]$VisualStudioRoot)

$ErrorActionPreference = 'Stop'
if (-not $VisualStudioRoot) {
    $h3Vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $VisualStudioRoot = & $h3Vswhere -latest -products '*' -property installationPath
}
$h3Vcvars = Join-Path $VisualStudioRoot 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $h3Vcvars)) { throw 'Visual Studio C++ Build Tools not found' }
$h3Command = '"' + $h3Vcvars + '" >nul && nvcc -shared -O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -allow-unsupported-compiler -Xcompiler=/EHs,/Zc:preprocessor,/wd4819 -gencode arch=compute_80,code=compute_80 -gencode arch=compute_89,code=sm_89 -I native/src -I native/third_party/cutlass/include native/src/block_v_attention.cu -o native/bin/h3_block_v_v3.dll'
Push-Location (Split-Path $PSScriptRoot)
try {
    & cmd /d /s /c $h3Command
    if ($LASTEXITCODE -ne 0) { throw 'block-V CUDA build failed' }
} finally {
    Pop-Location
}
