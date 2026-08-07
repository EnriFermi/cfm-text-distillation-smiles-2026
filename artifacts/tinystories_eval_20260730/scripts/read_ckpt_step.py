"""Print the Lightning global_step/epoch of a checkpoint, or -1 on any error.

Reads only archive/data.pkl from the checkpoint zip (tens of KB), never the
tensor payload, so it is cheap on a multi-GB file and safe on a partially
written one.
"""
import sys, zipfile, pickle, io


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass


_cache = {}


def _stub(full):
    if full not in _cache:
        _cache[full] = type(full.replace(".", "_"), (_Stub,), {})
    return _cache[full]


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("collections", "builtins", "copyreg", "__builtin__"):
            try:
                return pickle.Unpickler.find_class(self, module, name)
            except Exception:
                pass
        return _stub(f"{module}.{name}")

    def persistent_load(self, pid):
        return None


def main():
    try:
        path = sys.argv[1]
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("data.pkl"))
            obj = _U(io.BytesIO(z.read(name))).load()
        step = obj.get("global_step")
        epoch = obj.get("epoch")
        print(f"{int(step) if step is not None else -1}\t{epoch}")
    except Exception as exc:
        print(f"-1\t{type(exc).__name__}")


if __name__ == "__main__":
    main()
