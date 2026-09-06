#!/usr/bin/env bash
set -euo pipefail

# Thin v2 wrapper: repair the embedded Python code generator in the experimental
# v1 script without changing any mathematical candidate definition.
TMP=/tmp/run_apple_cos53_PLUTO_MATH_SWEEP_FIXED_IMPL.sh
python3 - <<'PY'
from pathlib import Path
p=Path('benchmark_support/run_apple_cos53_PLUTO_MATH_SWEEP.sh')
s=p.read_text()
old="""def write_variant(name,kernel,header=None):
    s=base[:start]+kernel+base[end:]
"""
new="""def write_variant(name,kernel,header=None):
    # Kernel builders intentionally use \\n escapes for readability inside the shell
    # heredoc; materialize them as real source newlines before writing C++.
    kernel=kernel.replace('\\\\n','\\n')
    s=base[:start]+kernel+base[end:]
"""
if old not in s:
    raise SystemExit('write_variant patch point not found')
s=s.replace(old,new,1)
Path('/tmp/run_apple_cos53_PLUTO_MATH_SWEEP_FIXED_IMPL.sh').write_text(s)
PY
chmod +x "$TMP"
exec bash "$TMP" "$@"
