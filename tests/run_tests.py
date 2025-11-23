#!/usr/bin/env python3
"""
Simple test runner for the HE-friendly pipeline tests.
Run with: python tests/run_tests.py
"""
import os
import sys

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

def run_test_file(test_file):
    """Run a test file and report results."""
    test_path = os.path.join(os.path.dirname(__file__), test_file)
    if not os.path.exists(test_path):
        print(f"⚠️  {test_file} not found")
        return False
    
    try:
        with open(test_path, 'r') as f:
            code = f.read()
        exec(code, {'__name__': '__main__'})
        print(f"✅ {test_file}")
        return True
    except AssertionError as e:
        print(f"❌ {test_file} - AssertionError: {e}")
        return False
    except Exception as e:
        print(f"❌ {test_file} - Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Running HE-friendly pipeline tests...\n")
    
    test_files = [
        "test_wrapper.py",
        "test_wrapper_layernorm.py",
        "test_wrapper_rmsnorm_silu.py",
        "test_attention_approx.py",
        "test_gaussian_kernel.py",
        "test_remez_gelu_default.py",
        "test_chebyshev_silu.py",
    ]
    
    results = []
    for test_file in test_files:
        result = run_test_file(test_file)
        results.append((test_file, result))
    
    print("\n" + "="*50)
    print("Test Summary:")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_file, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_file}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

