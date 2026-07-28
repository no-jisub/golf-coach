import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_collection import (  # noqa: E402
    BACKGROUND_TYPES,
    BODY_BUILDS,
    CLUB_TYPES,
    DEFAULT_DATASET_ROOT,
    DISTANCE_BANDS,
    EXPERIENCE_LEVELS,
    HANDEDNESS,
    HEIGHT_BANDS,
    LIGHTING_LEVELS,
    MOBILITY_LEVELS,
    VIEWS,
    create_collection_session,
    deactivate_collection_session,
    new_participant_id,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="웹캠 샘플용 비식별 참가자·촬영 세션을 만들고 활성화합니다."
    )
    parser.add_argument("--participant-id", default=None)
    parser.add_argument("--display-name", default="")
    parser.add_argument("--contact-reference", default="")
    parser.add_argument("--consent-reference", default="")
    parser.add_argument("--consent-confirmed", action="store_true")
    parser.add_argument("--height-band", choices=sorted(HEIGHT_BANDS), default="unspecified")
    parser.add_argument("--body-build", choices=sorted(BODY_BUILDS), default="unspecified")
    parser.add_argument("--mobility", choices=sorted(MOBILITY_LEVELS), default="unspecified")
    parser.add_argument(
        "--experience-level",
        choices=sorted(EXPERIENCE_LEVELS),
        default="unspecified",
    )
    parser.add_argument("--handedness", choices=sorted(HANDEDNESS), default="right")
    parser.add_argument("--view", choices=sorted(VIEWS), default="FACEON")
    parser.add_argument("--club-type", choices=sorted(CLUB_TYPES), default="I7")
    parser.add_argument("--camera-id", default="camera_0")
    parser.add_argument(
        "--distance-band",
        choices=sorted(DISTANCE_BANDS),
        default="unspecified",
    )
    parser.add_argument("--lighting", choices=sorted(LIGHTING_LEVELS), default="unspecified")
    parser.add_argument(
        "--background",
        choices=sorted(BACKGROUND_TYPES),
        default="unspecified",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--notes", default="")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--deactivate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.deactivate:
        changed = deactivate_collection_session(args.dataset_root)
        print("활성 세션을 해제했습니다." if changed else "활성 세션이 없습니다.")
        return 0
    if bool(args.width) != bool(args.height):
        raise SystemExit("--width와 --height는 함께 지정해야 합니다.")

    participant_id = args.participant_id or new_participant_id()
    session = create_collection_session(
        participant_id=participant_id,
        body_profile={
            "height_band": args.height_band,
            "body_build": args.body_build,
            "mobility": args.mobility,
            "experience_level": args.experience_level,
            "handedness": args.handedness,
        },
        capture_conditions={
            "view": args.view,
            "club_type": args.club_type,
            "camera_id": args.camera_id,
            "distance_band": args.distance_band,
            "lighting": args.lighting,
            "background": args.background,
            "resolution": [args.width, args.height] if args.width else None,
            "notes": args.notes,
        },
        consent_confirmed=args.consent_confirmed,
        private_profile={
            "display_name": args.display_name,
            "contact_reference": args.contact_reference,
            "consent_reference": args.consent_reference,
        },
        dataset_root=args.dataset_root,
    )
    print(f"참가자: {session['participant_id']}")
    print(f"세션: {session['session_id']}")
    print(f"비식별 세션: {session['session_path']}")
    print(f"개인정보(로컬 전용): {session['private_path']}")
    print(f"샘플 저장 위치: {session['capture_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
