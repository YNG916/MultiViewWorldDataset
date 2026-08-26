from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".in", ".txt"}
# Build sentinels from fragments so this test file does not itself contain a forbidden tracked path.
FORBIDDEN = (
    "/cvhci/temp/" + "yyang",
    "/home/" + "yyang",
    "OMNIGIBSON_GPU_ID" + "=0",
)



def test_tracked_sources_have_no_development_machine_paths():
    violations = []
    for path in REPOSITORY.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                violations.append(f"{path.relative_to(REPOSITORY)}: {forbidden}")
    assert not violations, violations
