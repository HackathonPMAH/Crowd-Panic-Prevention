import pathlib
import py_compile

backend_root = pathlib.Path('backend')
files = list(backend_root.rglob('*.py'))
print('compiling', len(files), 'files')
for p in files:
    py_compile.compile(str(p), doraise=True)
print('compile ok')
