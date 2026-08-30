# Experimental AMD Sparse Kitchen

This backend adapts Comfy Kitchen PR #143's HIP Sol-Attn exact stage and carrier
layouts to H3's existing 64Q x 64KV route. It does not use Sol's router or
approximate tail. The fat binaries target gfx11 (RDNA 3/3.5) and gfx12 (RDNA 4);
the H3 adaptation has not yet been run on AMD hardware.

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
