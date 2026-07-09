# flydsl-gemm
Companion code for [*Porting High-Performance HIP Kernels to FlyDSL*](LINK) blog post.

## Requirements
- `flydsl==0.2.2`

## Running the Code
> Note: the code was hand-tuned for AMD Instinct MI355X GPUs. Due to the specific instructions used, a CDNA4-capable GPU is required to run it.

Once FlyDSL is installed on your system simply run:

```bash
python gemm.py
```

If you see wrong results, try running the same command with the `FLYDSL_RUNTIME_ENABLE_CACHE=0` environment variable.
This will prevent FlyDSL from using previously cache kernels.  
