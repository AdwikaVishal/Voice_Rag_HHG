# Minimal stub for the `av` package to satisfy imports in test environments.
# This stub should never be used for real transcoding; it only prevents
# import-time failures in environments where PyAV is not installed.

def open(*args, **kwargs):
    raise RuntimeError("av is not available in this environment (stub)")
