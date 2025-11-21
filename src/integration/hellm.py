from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


def _run(cmd: List[str], cwd: Optional[str] = None, simulate: bool = False) -> int:
    if simulate:
        return 0
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.returncode


@dataclass
class HellmIntegration:
    """
    Thin wrapper around the ciphertext C++ code in the HE-friendly repo.
    Allows building the project and preparing a patched convert.py script.
    """
    repo_root: str = "/home/ubuntu/Def-acc/third_party/Encryption-friendly_LLM_Architecture"

    @property
    def ciphertext_dir(self) -> str:
        return os.path.join(self.repo_root, "ciphertext")

    @property
    def convert_py(self) -> str:
        return os.path.join(self.ciphertext_dir, "convert.py")

    @property
    def build_dir(self) -> str:
        return os.path.join(self.ciphertext_dir, "build")

    def exists(self) -> bool:
        return os.path.isdir(self.ciphertext_dir) and os.path.isfile(self.convert_py)

    def build(self, simulate: bool = False) -> None:
        """
        Configure and build the ciphertext C++ project using CMake.
        """
        if not self.exists():
            raise FileNotFoundError(f"ciphertext directory not found at {self.ciphertext_dir}")
        os.makedirs(self.build_dir, exist_ok=True)
        _run(["cmake", "-S", self.ciphertext_dir, "-B", self.build_dir], simulate=simulate)
        _run(["cmake", "--build", self.build_dir, "-j"], simulate=simulate)

    def patch_convert_script(self, weight_path: str, save_path: str, out_script_path: str) -> str:
        """
        Create a patched copy of convert.py with weight_path and save_path set.
        Returns the path to the patched script.
        """
        if not os.path.isfile(self.convert_py):
            raise FileNotFoundError(f"convert.py not found at {self.convert_py}")
        os.makedirs(os.path.dirname(out_script_path), exist_ok=True)
        with open(self.convert_py, "r", encoding="utf-8") as f:
            src = f.read()
        # Replace the lines that set weight_path and save_path
        src = re.sub(r'weight_path\s*=\s*".*?"', f'weight_path = "{weight_path}"', src)
        src = re.sub(r'save_path\s*=\s*".*?"', f'save_path = "{save_path}"', src)
        with open(out_script_path, "w", encoding="utf-8") as f:
            f.write(src)
        return out_script_path

    def run_convert(self, patched_script_path: str, python_exe: str = "python", simulate: bool = False) -> None:
        """
        Execute the patched convert.py to produce a container for HE C++.
        """
        if not os.path.isfile(patched_script_path):
            raise FileNotFoundError(f"Patched script not found: {patched_script_path}")
        _run([python_exe, patched_script_path], cwd=self.ciphertext_dir, simulate=simulate)


