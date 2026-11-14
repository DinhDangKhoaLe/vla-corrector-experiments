"""Import checks without downloading weights/data or launching a rollout."""
import argparse
import importlib
import importlib.metadata
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin", action="store_true")
    args = parser.parse_args()
    modules = ["torch", "torchvision", "torchcodec", "transformers", "numpy", "scipy", "pyarrow",
               "h5py", "metaworld", "libero.libero.benchmark", "lerobot.policies.pi05.modeling_pi05",
               "laco.controller", "siglip_dynamics.pace"]
    if args.robotwin:
        modules += ["sapien", "mplib", "toppra", "open3d", "curobo",
                    "curobo.curobolib.geom_cu", "pytorch3d", "pytorch3d._C"]
    errors = {}
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"OK {name}")
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    print(f"Python: {sys.version}")
    if "torch" not in errors:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
    for name in ("numpy", "torch", "torchcodec", "sapien", "mplib", "nvidia_curobo", "pytorch3d"):
        try:
            print(name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            pass
    if errors:
        raise SystemExit(json.dumps(errors, indent=2))


if __name__ == "__main__":
    main()
