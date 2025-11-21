#!/usr/bin/env python3
import argparse
import sys
import os
from typing import Optional

# Ensure 'src' root is on sys.path when invoking this file directly
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from integration.hellm import HellmIntegration


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage HE-friendly C++ integration (build and convert).")
    parser.add_argument("--repo_root", type=str, default="/home/ubuntu/Def-acc/third_party/Encryption-friendly_LLM_Architecture")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Configure and build the ciphertext C++ project with CMake.")
    p_build.add_argument("--simulate", action="store_true", help="Skip running commands, validate only.")

    p_patch = sub.add_parser("patch-convert", help="Create a patched convert.py specifying weight_path and save_path.")
    p_patch.add_argument("--weight_path", type=str, required=True, help="Path to model.safetensors")
    p_patch.add_argument("--save_path", type=str, required=True, help="Output container path (e.g., converted_weights_xxx.pth)")
    p_patch.add_argument("--out", type=str, required=False, help="Output script path (default: build/convert_patched.py)")

    p_run = sub.add_parser("run-convert", help="Run a patched convert.py to produce the container.")
    p_run.add_argument("--script", type=str, required=True, help="Path to patched convert script")
    p_run.add_argument("--simulate", action="store_true", help="Skip running script, validate only.")

    args = parser.parse_args()
    integ = HellmIntegration(repo_root=args.repo_root)
    if args.cmd == "build":
        integ.build(simulate=args.simulate)
        print("Build completed (or simulated).")
    elif args.cmd == "patch-convert":
        out = args.out or os.path.join(integ.ciphertext_dir, "build", "convert_patched.py")
        path = integ.patch_convert_script(weight_path=args.weight_path, save_path=args.save_path, out_script_path=out)
        print(f"Patched convert script written to: {path}")
    elif args.cmd == "run-convert":
        integ.run_convert(patched_script_path=args.script, simulate=args.simulate)
        print("convert.py executed (or simulated).")


if __name__ == "__main__":
    main()


