import os
import tempfile

from integration.hellm import HellmIntegration


def test_hellm_exist_and_patch_convert():
    integ = HellmIntegration()
    assert integ.exists()
    with tempfile.TemporaryDirectory() as tmpd:
        patched = os.path.join(tmpd, "convert_patched.py")
        out = integ.patch_convert_script(
            weight_path="/abs/path/to/model.safetensors",
            save_path="/abs/path/to/converted_weights_mrpc.pth",
            out_script_path=patched,
        )
        assert os.path.isfile(out)
        with open(out, "r", encoding="utf-8") as f:
            src = f.read()
            assert 'weight_path = "/abs/path/to/model.safetensors"' in src
            assert 'save_path = "/abs/path/to/converted_weights_mrpc.pth"' in src


def test_hellm_build_simulate():
    integ = HellmIntegration()
    # Should not raise
    integ.build(simulate=True)


