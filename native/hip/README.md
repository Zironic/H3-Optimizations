# Experimental AMD Sparse Kitchen

This backend adapts Comfy Kitchen's gfx12 INT8 attention for H3's 64Q x 64KV
sparse route. It targets only `gfx1200` and `gfx1201` and has not yet been run
on AMD hardware.

Prebuilt Linux x86-64 and Windows x64 libraries are shipped in `native/hip/bin`
and rebuilt by the `Build experimental AMD Sparse Kitchen` workflow. No
compiler is needed on the test machine. The matching ROCm runtime and a
supported AMD GPU are still required.

Build on a ROCm development host:

```text
cmake -S native/hip -B native/hip/build -G Ninja
cmake --build native/hip/build --config Release
```

The loader searches `native/hip/bin`, `native/hip/lib`, `native/hip/build`, and
`native/hip/build/Release`. A developer can place an explicit library path in
`native/hip/library_path.txt`.

The first use runs per-device full and genuinely sparse numerical checks using
production-shaped strided inputs and NHD output. Automatic H3 sparse selection
accepts the backend only after both checks pass.
