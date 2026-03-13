# app/__init__.py
# SymPy mpmath compatibility fix - MUST run before torch is imported
import sys
try:
    import mpmath
    sys.modules['sympy.mpmath'] = mpmath
except ImportError:
    pass
