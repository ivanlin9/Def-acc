from .wrapper import (
    make_he_ready_model,
    enable_attention_approximation,
    disable_attention_approximation,
    enable_gaussian_kernel_attention,
    set_calibration,
    set_gaussian_alpha,
)
from .calibrate import calibrate_model_ranges

try:
    from .encrypt import (
        encrypt_model,
        EncryptedModel,
        create_he_context,
    )
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    # Define stubs for when TenSEAL is not available
    encrypt_model = None
    EncryptedModel = None
    create_he_context = None

__all__ = [
    "make_he_ready_model",
    "enable_attention_approximation",
    "disable_attention_approximation",
    "enable_gaussian_kernel_attention",
    "set_calibration",
    "set_gaussian_alpha",
    "calibrate_model_ranges",
]

if ENCRYPTION_AVAILABLE:
    __all__.extend([
        "encrypt_model",
        "EncryptedModel",
        "create_he_context",
    ])


