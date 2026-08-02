"""가이드 병목과 코치 정답 기반 임계값 성능을 분석합니다."""

from collections import Counter, defaultdict

from utils.guide_alignment import STAGE_KEYS


SCORING_ANALYSIS_SCHEMA = "golf-coach-scoring-analysis-v1"
LANDMARK_LABELS = {
    "0": "nose",
    "11": "left_shoulder",
    "12": "right_shoulder",
    "13": "left_elbow",
    "14": "right_elbow",
    "15": "left_wrist",
    "16": "right_wrist",
    "23": "left_hip",
    "24": "right_hip",
    "25": "left_knee",
    "26": "right_knee",
    "27": "left_ankle",
    "28": "right_ankle",
}


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def analyze_regression_bottlenecks(regression_report):
    """단계별 낮은 점수를 관절, 신체 그룹, CaddieSet 항목으로 분해합니다."""
    videos = [
        video
        for video in regression_report.get("videos", {}).values()
        if video.get("status") == "ok"
    ]
    stages = {}
    for stage_key in STAGE_KEYS:
        records = [
            video["stages"][stage_key]
            for video in videos
            if stage_key in video.get("stages", {})
        ]
        group_values = defaultdict(list)
        joint_values = defaultdict(list)
        caddies_warnings = Counter()
        for record in records:
            for group, value in record.get("guide_group_distances", {}).items():
                if value is not None:
                    group_values[group].append(value)
            for index, value in record.get("joint_distances", {}).items():
                if value is not None:
                    joint_values[index].append(value)
            for metric_key, item in record.get("caddieset_items", {}).items():
                if item.get("status") != "pass":
                    caddies_warnings[metric_key] += 1

        group_means = {
            group: _mean(values)
            for group, values in group_values.items()
        }
        joint_means = sorted(
            (
                {
                    "landmark_index": int(index),
                    "label": LANDMARK_LABELS.get(index, index),
                    "mean_distance": _mean(values),
                    "sample_count": len(values),
                }
                for index, values in joint_values.items()
            ),
            key=lambda item: item["mean_distance"],
            reverse=True,
        )
        mean_guide = _mean(record.get("guide_score") for record in records)
        mean_caddieset = _mean(
            record.get("caddieset_score") for record in records
        )
        if mean_guide is None or mean_caddieset is None:
            primary_component = "insufficient"
        elif mean_guide + 5 < mean_caddieset:
            primary_component = "guide"
        elif mean_caddieset + 5 < mean_guide:
            primary_component = "caddieset"
        else:
            primary_component = "balanced"
        stages[stage_key] = {
            "sample_count": len(records),
            "mean_final_score": _mean(
                record.get("final_score") for record in records
            ),
            "mean_guide_score": mean_guide,
            "mean_caddieset_score": mean_caddieset,
            "pass_rate": (
                round(
                    sum(bool(record.get("passed")) for record in records)
                    / len(records),
                    4,
                )
                if records
                else None
            ),
            "primary_component": primary_component,
            "guide_group_means": group_means,
            "top_joint_bottlenecks": joint_means[:5],
            "caddieset_warning_counts": dict(caddies_warnings.most_common()),
        }

    ranked_stages = sorted(
        STAGE_KEYS,
        key=lambda stage_key: (
            stages[stage_key]["mean_final_score"]
            if stages[stage_key]["mean_final_score"] is not None
            else float("inf")
        ),
    )
    return {
        "stage_order_by_bottleneck": ranked_stages,
        "stages": stages,
    }


def confusion_metrics(records, threshold):
    usable = [
        record
        for record in records
        if record.get("label") in {"good", "bad"}
        and record.get("model", {}).get("final_score") is not None
    ]
    tp = fp = tn = fn = 0
    for record in usable:
        predicted_good = float(record["model"]["final_score"]) >= threshold
        actual_good = record["label"] == "good"
        if predicted_good and actual_good:
            tp += 1
        elif predicted_good:
            fp += 1
        elif actual_good:
            fn += 1
        else:
            tn += 1
    total = len(usable)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "threshold": threshold,
        "sample_count": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pass_rate": round((tp + fp) / total, 4) if total else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def analyze_thresholds(ground_truth, thresholds=None):
    thresholds = list(thresholds if thresholds is not None else range(0, 101, 5))
    records = ground_truth.get("records", [])
    overall = [confusion_metrics(records, threshold) for threshold in thresholds]
    scored = [item for item in overall if item["f1"] is not None]
    best = (
        max(scored, key=lambda item: (item["f1"], -abs(item["threshold"] - 70)))
        if scored
        else None
    )
    by_stage = {}
    for stage_key in STAGE_KEYS:
        stage_records = [
            record for record in records if record.get("stage_key") == stage_key
        ]
        by_stage[stage_key] = [
            confusion_metrics(stage_records, threshold)
            for threshold in thresholds
        ]

    subgroup_fields = {
        "height_band": lambda record: record.get("body_profile", {}).get(
            "height_band", "unspecified"
        ),
        "body_build": lambda record: record.get("body_profile", {}).get(
            "body_build", "unspecified"
        ),
    }
    subgroups = {}
    for field, getter in subgroup_fields.items():
        values = sorted({getter(record) for record in records})
        subgroups[field] = {
            value: confusion_metrics(
                [record for record in records if getter(record) == value],
                70,
            )
            for value in values
        }
    return {
        "thresholds": thresholds,
        "overall": overall,
        "best_f1_threshold": best,
        "by_stage": by_stage,
        "subgroups_at_70": subgroups,
    }


def build_scoring_analysis(regression_report, ground_truth=None):
    return {
        "schema": SCORING_ANALYSIS_SCHEMA,
        "scope": {
            "regression_event_scope": regression_report.get("scope", {}).get(
                "event_scope", "unknown"
            ),
            "criterion_tuning_allowed": regression_report.get("scope", {}).get(
                "criterion_tuning_allowed", False
            ),
        },
        "dataset_quality": regression_report.get("dataset_quality"),
        "bottlenecks": analyze_regression_bottlenecks(regression_report),
        "threshold_analysis": (
            analyze_thresholds(ground_truth) if ground_truth is not None else None
        ),
    }


def _format_metric(value):
    return "-" if value is None else f"{value:.3f}"


def render_scoring_analysis_markdown(report):
    reviewed_only = (
        report.get("scope", {}).get("regression_event_scope")
        == "reviewed_only"
    )
    lines = [
        (
            "# Reviewed-only Scoring and Bottleneck Analysis"
            if reviewed_only
            else "# Scoring and Bottleneck Analysis"
        ),
        "",
    ]
    dataset_quality = report.get("dataset_quality")
    if dataset_quality:
        lines.extend(
            [
                "## Dataset Quality Gate",
                "",
                "- 기준 조정 허용: "
                + ("yes" if dataset_quality["criterion_tuning_allowed"] else "no"),
            ]
        )
        lines.extend(
            f"- 경고: {warning}"
            for warning in dataset_quality.get("warnings", [])
        )
        lines.append("")
    lines.extend(
        [
            "## Stage bottlenecks",
            "",
            "| Stage | Samples | Final | Guide | I7 | Pass | Primary | Top joints |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    bottlenecks = report["bottlenecks"]
    for stage_key in STAGE_KEYS:
        stage = bottlenecks["stages"][stage_key]
        joints = ", ".join(
            f"{item['label']}({_format_metric(item['mean_distance'])})"
            for item in stage["top_joint_bottlenecks"][:3]
        )
        pass_rate = (
            f"{stage['pass_rate'] * 100:.1f}%"
            if stage["pass_rate"] is not None
            else "-"
        )
        lines.append(
            f"| {stage_key} | {stage['sample_count']} | "
            f"{_format_metric(stage['mean_final_score'])} | "
            f"{_format_metric(stage['mean_guide_score'])} | "
            f"{_format_metric(stage['mean_caddieset_score'])} | "
            f"{pass_rate} | {stage['primary_component']} | {joints or '-'} |"
        )

    threshold = report.get("threshold_analysis")
    if threshold is not None:
        lines.extend(
            [
                "",
                "## Threshold sweep",
                "",
                "| Threshold | N | TP | FP | TN | FN | Pass | Precision | Recall | F1 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in threshold["overall"]:
            pass_rate = (
                f"{item['pass_rate'] * 100:.1f}%"
                if item["pass_rate"] is not None
                else "-"
            )
            lines.append(
                f"| {item['threshold']} | {item['sample_count']} | {item['tp']} | "
                f"{item['fp']} | {item['tn']} | {item['fn']} | {pass_rate} | "
                f"{_format_metric(item['precision'])} | "
                f"{_format_metric(item['recall'])} | {_format_metric(item['f1'])} |"
            )
    lines.append("")
    return "\n".join(lines)
