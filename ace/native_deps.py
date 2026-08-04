"""
Native extension import-order guard.

On Windows, importing sentence-transformers before pyarrow has been loaded kills
the interpreter with an access violation (exit 0xC0000005). The fault is raised
by pyarrow's own extension module, reached transitively:

    sentence_transformers -> sklearn -> pandas.compat.pyarrow -> pyarrow

Importing pyarrow first is stable, so every module that pulls in
sentence-transformers must call ensure_native_import_order() beforehand. Without
it `import ace.core` crashes outright and no run can start.

Observed with pyarrow 24.0.0 / pandas 3.0.3 / scikit-learn 1.8.0 /
sentence-transformers 5.2.3 / torch 2.10.0 on CPython 3.12 (Windows). Delete this
module once a dependency bump makes the bare import safe again.
"""

_DONE = False


def ensure_native_import_order() -> None:
    """Load pyarrow ahead of sentence-transformers. Safe to call repeatedly."""
    global _DONE
    if _DONE:
        return
    _DONE = True

    try:
        import pyarrow  # noqa: F401
    except Exception:
        # pyarrow absent or already broken: nothing to order, let the real
        # import below surface whatever error it raises.
        pass
