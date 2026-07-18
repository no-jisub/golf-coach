from statistics import mean, median

from utils.guide_alignment import STAGE_KEYS


ACCURACY_SCHEMA = "golf-coach-swing-stage-accuracy-v1"
DEFAULT_TOLERANCE_MS = 150.0


def _rounded(value):
    return round(float(value), 3)


def _aggregate_measurements(measurements):
    if not measurements:
        return {
            "evaluated_count": 0,
            "within_tolerance_count": 0,
            "within_tolerance_rate": None,
            "mean_absolute_error_frames": None,
            "median_absolute_error_frames": None,
            "mean_absolute_error_ms": None,
            "median_absolute_error_ms": None,
        }

    absolute_frames = [item["absolute_error_frames"] for item in measurements]
    absolute_ms = [item["absolute_error_ms"] for item in measurements]
    within_count = sum(item["within_tolerance"] for item in measurements)
    return {
        "evaluated_count": len(measurements),
        "within_tolerance_count": within_count,
        "within_tolerance_rate": _rounded(within_count / len(measurements)),
        "mean_absolute_error_frames": _rounded(mean(absolute_frames)),
        "median_absolute_error_frames": _rounded(median(absolute_frames)),
        "mean_absolute_error_ms": _rounded(mean(absolute_ms)),
        "median_absolute_error_ms": _rounded(median(absolute_ms)),
    }


def build_stage_accuracy_report(audit, manifest, *, tolerance_ms=DEFAULT_TOLERANCE_MS):
    """Compare automatic events only with human-reviewed ground-truth events."""
    tolerance_ms = float(tolerance_ms)
    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be zero or greater")

    all_measurements = []
    stage_measurements = {stage_key: [] for stage_key in STAGE_KEYS}
    video_reports = {}

    for video_id, audit_video in audit.get("videos", {}).items():
        ground_truth = manifest["videos"][video_id]
        ground_truth_status = ground_truth["review_status"]
        audit_status = audit_video.get("status")
        video_report = {
            "source": ground_truth["source"],
            "ground_truth_status": ground_truth_status,
        }

        if ground_truth_status == "excluded" or audit_status == "excluded":
            video_report["status"] = "excluded"
        elif audit_status != "ok":
            video_report["status"] = "analysis_failed"
            video_report["error"] = audit_video.get("error", "automatic analysis failed")
        elif ground_truth_status != "reviewed":
            video_report["status"] = "pending_review"
        else:
            fps = float(audit_video.get("video", {}).get("fps") or 0)
            if fps <= 0:
                video_report["status"] = "analysis_failed"
                video_report["error"] = "video fps is missing or invalid"
                video_reports[video_id] = video_report
                continue

            comparisons = {}
            for stage_key in STAGE_KEYS:
                automatic_frame = int(
                    audit_video["stage_detection"]["events"][stage_key]["frame_index"]
                )
                ground_truth_frame = int(ground_truth["events"][stage_key])
                signed_error_frames = automatic_frame - ground_truth_frame
                signed_error_ms = signed_error_frames / fps * 1000.0
                measurement = {
                    "ground_truth_frame": ground_truth_frame,
                    "automatic_frame": automatic_frame,
                    "signed_error_frames": signed_error_frames,
                    "absolute_error_frames": abs(signed_error_frames),
                    "signed_error_ms": _rounded(signed_error_ms),
                    "absolute_error_ms": _rounded(abs(signed_error_ms)),
                    "within_tolerance": abs(signed_error_ms) <= tolerance_ms,
                }
                comparisons[stage_key] = measurement
                all_measurements.append(measurement)
                stage_measurements[stage_key].append(measurement)

            video_report.update(
                {
                    "status": "evaluated",
                    "fps": _rounded(fps),
                    "comparisons": comparisons,
                    "metrics": _aggregate_measurements(list(comparisons.values())),
                }
            )

        video_reports[video_id] = video_report

    status_counts = {
        status: sum(video["status"] == status for video in video_reports.values())
        for status in ("evaluated", "pending_review", "excluded", "analysis_failed")
    }
    summary = {
        "video_count": len(video_reports),
        "reviewed_count": status_counts["evaluated"],
        "pending_review_count": status_counts["pending_review"],
        "excluded_count": status_counts["excluded"],
        "analysis_failed_count": status_counts["analysis_failed"],
        **_aggregate_measurements(all_measurements),
    }
    return {
        "schema": ACCURACY_SCHEMA,
        "stage_order": list(STAGE_KEYS),
        "tolerance_ms": tolerance_ms,
        "summary": summary,
        "stages": {
            stage_key: _aggregate_measurements(stage_measurements[stage_key])
            for stage_key in STAGE_KEYS
        },
        "videos": video_reports,
    }
