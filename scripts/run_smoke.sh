#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 OUTPUT_ROOT [additional mvwd arguments...]" >&2
    exit 2
fi

output_root=$1
shift
result_file="$output_root/smoke_probe_last_result.json"
rm -f "$result_file"
set +e
mvwd simulator-smoke --config configs/smoke.yaml --output-root "$output_root" "$@"
cli_status=$?
set -e
python -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
cli_status = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(cli_status or 3)
result = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if cli_status == 0 and result.get("status") == "pass" else 1)
' "$result_file" "$cli_status"
