import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.repository_safety import (  # noqa: E402
    DEFAULT_MAX_FILE_BYTES,
    scan_tracked_repository,
)


def main():
    parser = argparse.ArgumentParser(
        description="Git 추적 파일에서 개인정보·원본 미디어·비밀값 유출을 검사합니다."
    )
    parser.add_argument("--repository", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=DEFAULT_MAX_FILE_BYTES / (1024 * 1024),
    )
    args = parser.parse_args()
    if args.max_file_mb <= 0:
        parser.error("--max-file-mb는 0보다 커야 합니다.")

    violations = scan_tracked_repository(
        args.repository,
        max_file_bytes=int(args.max_file_mb * 1024 * 1024),
    )
    if violations:
        print(f"저장소 안전 검사 실패: {len(violations)}건")
        for item in violations:
            location = item["path"]
            if "line" in item:
                location += f":{item['line']}"
            print(f"- {location}: {item['type']}")
        return 1

    print("저장소 안전 검사 통과: 추적 파일에서 차단 대상이 발견되지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
