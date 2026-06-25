import argparse
import json
import math
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = PROJECT_ROOT / "reference_data" / "extracted_landmarks"

CONNECTIONS = [
    (0, 11),
    (0, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

EDITABLE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
STAGES = ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]


class LandmarkEditor:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = json.loads(json_path.read_text(encoding="utf-8"))
        self.image_path = PROJECT_ROOT / self.data["image"]
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {self.image_path}")

        self.height, self.width, _ = self.image.shape
        self.selected_index = None
        self.dragging = False
        self.dirty = False
        self.window_name = "Edit Reference Landmarks"

    def landmark_to_pixel(self, index):
        landmark = self.data["landmarks"][index]
        return int(landmark["x"] * self.width), int(landmark["y"] * self.height)

    def set_landmark_pixel(self, index, x, y):
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        landmark = self.data["landmarks"][index]
        landmark["x"] = x / self.width
        landmark["y"] = y / self.height
        landmark["manually_adjusted"] = True
        self.dirty = True

    def nearest_landmark(self, x, y, max_distance=30):
        nearest_index = None
        nearest_distance = max_distance
        for index in EDITABLE_LANDMARKS:
            px, py = self.landmark_to_pixel(index)
            dist = math.sqrt((px - x) ** 2 + (py - y) ** 2)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_index = index
        return nearest_index

    def draw(self):
        canvas = self.image.copy()

        for start_idx, end_idx in CONNECTIONS:
            cv2.line(
                canvas,
                self.landmark_to_pixel(start_idx),
                self.landmark_to_pixel(end_idx),
                (255, 255, 255),
                3,
            )

        for index in EDITABLE_LANDMARKS:
            point = self.landmark_to_pixel(index)
            selected = index == self.selected_index
            color = (0, 0, 255) if selected else (0, 255, 255)
            radius = 9 if selected else 6
            cv2.circle(canvas, point, radius, color, -1)
            cv2.putText(
                canvas,
                str(index),
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        help_lines = [
            f"{self.data['stage']} | {self.image_path.name}",
            "mouse drag: move joint | arrows: nudge | s: save | q/esc: quit",
            "editable: 0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles",
        ]
        y = 28
        for line in help_lines:
            cv2.rectangle(canvas, (8, y - 22), (max(760, len(line) * 11), y + 8), (0, 0, 0), -1)
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y += 30

        if self.dirty:
            cv2.putText(canvas, "UNSAVED", (16, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        return canvas

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = self.nearest_landmark(x, y)
            if selected is not None:
                self.selected_index = selected
                self.dragging = True
                self.set_landmark_pixel(selected, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.selected_index is not None:
            self.set_landmark_pixel(self.selected_index, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def nudge_selected(self, dx, dy):
        if self.selected_index is None:
            return
        x, y = self.landmark_to_pixel(self.selected_index)
        self.set_landmark_pixel(self.selected_index, x + dx, y + dy)

    def save(self):
        self.json_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.dirty = False
        print(f"[SAVE] {self.json_path.relative_to(PROJECT_ROOT)}")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while True:
            cv2.imshow(self.window_name, self.draw())
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("s"):
                self.save()
            elif key in (81, ord("a")):
                self.nudge_selected(-1, 0)
            elif key in (83, ord("d")):
                self.nudge_selected(1, 0)
            elif key in (82, ord("w")):
                self.nudge_selected(0, -1)
            elif key in (84, ord("x")):
                self.nudge_selected(0, 1)

        cv2.destroyAllWindows()


def find_default_json(stage):
    stage_dir = EXTRACTED_DIR / stage
    json_files = sorted(stage_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"JSON 파일이 없습니다: {stage_dir}")
    return json_files[0]


def main():
    parser = argparse.ArgumentParser(description="추출된 관절 좌표 JSON을 이미지 위에서 수동 보정합니다.")
    parser.add_argument("--stage", choices=STAGES, help="편집할 단계. --json이 없으면 해당 단계의 첫 JSON을 엽니다.")
    parser.add_argument("--json", type=Path, help="직접 편집할 JSON 경로")
    args = parser.parse_args()

    if args.json:
        json_path = args.json
        if not json_path.is_absolute():
            json_path = PROJECT_ROOT / json_path
    elif args.stage:
        json_path = find_default_json(args.stage)
    else:
        raise SystemExit("--stage 또는 --json 중 하나를 지정하세요.")

    editor = LandmarkEditor(json_path)
    editor.run()


if __name__ == "__main__":
    main()
