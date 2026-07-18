import argparse
import hashlib
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reference_data" / "caddieset" / "source"
SOURCE_COMMIT = "3c73d9d40580bb8a5a10711ad1fa10735a205ffe"
RAW_BASE_URL = f"https://raw.githubusercontent.com/damilab/CaddieSet/{SOURCE_COMMIT}"
FILES = {
    "data/CaddieSet.csv": {
        "url": f"{RAW_BASE_URL}/data/CaddieSet.csv",
        "sha256": "646459b081863e244d75efe6f09fad4d34750d2fed7fde44db654e7720a9a306",
    },
    "LICENSE": {
        "url": f"{RAW_BASE_URL}/LICENSE",
        "sha256": "63cd54135f363455c5ccc69169fadf26eb609c934d67b26a149dbf4b82d3aee1",
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, output_path, expected_sha256, force=False):
    if output_path.exists() and not force:
        actual_sha256 = file_sha256(output_path)
        if actual_sha256 == expected_sha256:
            return "verified"
        raise ValueError(
            f"기존 파일의 SHA-256이 다릅니다: {output_path}\n"
            "덮어쓰려면 --force를 사용하세요."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            temporary_path.write_bytes(response.read())
        actual_sha256 = file_sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"다운로드 파일의 SHA-256이 다릅니다: {output_path}\n"
                f"expected={expected_sha256}\nactual={actual_sha256}"
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return "downloaded"


def parse_args():
    parser = argparse.ArgumentParser(
        description="고정된 CaddieSet 버전의 CSV와 라이선스를 내려받고 무결성을 검사합니다."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="기존 파일을 다시 내려받습니다.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    for relative_path, source in FILES.items():
        output_path = output_dir / relative_path
        status = download_file(
            source["url"],
            output_path,
            source["sha256"],
            force=args.force,
        )
        print(f"[{status.upper()}] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
