#!/usr/bin/env bash
set -euo pipefail
src="benchmark_support/run_apple_cos53_degree2_retuned_dense_1m.sh"
tmp="/tmp/run_apple_cos53_degree2_retuned_dense_1m_fixed.sh"
python3 - "$src" "$tmp" <<'PY'
from pathlib import Path
import sys
src,dst=map(Path,sys.argv[1:3])
s=src.read_text()
s=s.replace("Path(f'/tmp/degree2_retuned_K{{K}}.cpp').write_text(s)",
            "Path(f'/tmp/degree2_retuned_K{K}.cpp').write_text(s)")
if "K{{K}}.cpp" in s:
    raise SystemExit('filename brace fix did not apply')
dst.write_text(s)
PY
bash "$tmp" "$@"
