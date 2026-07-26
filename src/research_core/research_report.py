"""Combined HTML reports, final validation summaries, and run manifests."""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import BUILD_MARKER, RESEARCH_CORE_VERSION
from .metadata import write_json


MODEL_LABELS = {
    "sgp4": "SGP4 (frozen TLE)",
    "analytical_two_body": "Analytical two-body",
    "numerical_two_body": "Numerical two-body",
    "numerical_j2": "Numerical J2",
    "numerical_j2_drag": "Numerical J2 + simplified drag",
}

MODEL_EXPLANATIONS = {
    "analytical_two_body": (
        "Closed-form Keplerian solution using spherical point-mass Earth gravity only."
    ),
    "numerical_two_body": (
        "Numerical integration of the same spherical point-mass physics used by the analytical model."
    ),
    "numerical_j2": (
        "Numerical integration of point-mass gravity plus the dominant Earth-oblateness term J2."
    ),
    "numerical_j2_drag": (
        "Numerical J2 propagation plus an illustrative exponential atmospheric-drag model."
    ),
    "sgp4": (
        "TLE-specific semi-analytical propagation in TEME using the frozen element set."
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _format_number(value: Any, *, digits: int = 6, unavailable: str = "—") -> str:
    if value is None:
        return unavailable
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _e(value)
    if numeric == 0.0:
        return "0"
    magnitude = abs(numeric)
    if magnitude >= 1.0e5 or magnitude < 1.0e-3:
        return f"{numeric:.{digits}e}"
    return f"{numeric:.{digits}f}"


def _status_class(status: str) -> str:
    normalized = status.lower()
    if normalized == "passed":
        return "status-pass"
    if normalized == "passed_with_warnings":
        return "status-warn"
    return "status-fail"


def _relative_href(report_path: Path, target: Path) -> str:
    return target.relative_to(report_path.parent).as_posix()


def _link(report_path: Path, target: Path, label: str | None = None) -> str:
    href = _relative_href(report_path, target)
    return f'<a href="{_e(href)}">{_e(label or target.name)}</a>'


def _figure(report_path: Path, stem: str, caption: str) -> str:
    png = report_path.parent / "figures" / f"{stem}.png"
    if not png.is_file():
        return ""
    href = _relative_href(report_path, png)
    return (
        '<figure class="figure">'
        f'<a href="{_e(href)}"><img src="{_e(href)}" alt="{_e(caption)}"></a>'
        f'<figcaption>{_e(caption)}</figcaption>'
        "</figure>"
    )


def _validation_counts(validation: dict[str, Any]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "not_evaluated": 0}
    for check in validation.get("checks", []):
        status = str(check.get("status", "not_evaluated"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_final_validation_summary(
    *,
    config: dict[str, Any],
    validation: dict[str, Any],
    warnings: Iterable[str],
    report_filename: str,
) -> dict[str, Any]:
    """Create the compact final validation/claim-level summary."""
    counts = _validation_counts(validation)
    external_enabled = bool(config.get("external_validation", {}).get("enabled", False))
    failed_checks = [
        {
            "validation_id": item.get("validation_id"),
            "name": item.get("name"),
            "measured_value": item.get("measured_value"),
            "criterion": item.get("criterion"),
        }
        for item in validation.get("checks", [])
        if item.get("status") == "failed"
    ]
    return {
        "research_core_version": RESEARCH_CORE_VERSION,
        "build_marker": BUILD_MARKER,
        "created_utc": _utc_now_iso(),
        "experiment_id": config["experiment"]["experiment_id"],
        "case_id": config["experiment"]["case_id"],
        "overall_status": validation.get("overall_status"),
        "validation_check_counts": counts,
        "failed_checks": failed_checks,
        "warnings": list(warnings),
        "external_validation_enabled": external_enabled,
        "external_validation_status": (
            "configured" if external_enabled else "not_performed"
        ),
        "scientific_claim_level": (
            "internally_verified_and_externally_compared"
            if external_enabled and not failed_checks
            else "internally_verified_external_validation_pending"
        ),
        "operational_qualification": "research_only_not_flight_qualified",
        "primary_report": report_filename,
        "interpretation": (
            "Validation checks demonstrate internal consistency and controlled model comparisons. "
            "They do not establish measured orbit truth or operational flight accuracy."
        ),
    }


def write_final_validation_summary(
    output_path: str | Path,
    *,
    config: dict[str, Any],
    validation: dict[str, Any],
    warnings: Iterable[str],
    report_filename: str = "RESEARCH_REPORT.html",
) -> dict[str, Any]:
    summary = build_final_validation_summary(
        config=config,
        validation=validation,
        warnings=warnings,
        report_filename=report_filename,
    )
    write_json(summary, output_path)
    return summary


def _category(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "research_report.html":
        return "primary_report"
    if name in {"run_manifest.json", "final_validation_summary.json", "validation_status.json"}:
        return "validation_and_manifest"
    if "configuration" in name:
        return "configuration"
    if "metadata" in name or "provenance" in name or "diagnostic" in name:
        return "metadata_and_diagnostics"
    if suffix == ".csv":
        return "scientific_data_csv"
    if suffix in {".png", ".pdf"}:
        return "figure"
    if suffix in {".md", ".txt"}:
        return "technical_documentation"
    if suffix == ".json":
        return "scientific_summary_json"
    return "other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_manifest(
    *,
    result_directory: str | Path,
    config: dict[str, Any],
    validation: dict[str, Any],
    warnings: Iterable[str],
    exclude_names: Iterable[str] = ("RUN_MANIFEST.json",),
) -> dict[str, Any]:
    """Build a checksum inventory of one completed result directory."""
    root = Path(result_directory).resolve()
    excluded = set(exclude_names)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        entries.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "category": _category(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    categories: dict[str, int] = {}
    for entry in entries:
        categories[entry["category"]] = categories.get(entry["category"], 0) + 1
    return {
        "research_core_version": RESEARCH_CORE_VERSION,
        "build_marker": BUILD_MARKER,
        "created_utc": _utc_now_iso(),
        "experiment_id": config["experiment"]["experiment_id"],
        "case_id": config["experiment"]["case_id"],
        "result_directory": str(root),
        "validation_status": validation.get("overall_status"),
        "warning_count": len(list(warnings)),
        "file_count_excluding_manifest": len(entries),
        "total_size_bytes_excluding_manifest": sum(item["size_bytes"] for item in entries),
        "category_counts": categories,
        "files": entries,
        "manifest_note": "RUN_MANIFEST.json excludes its own checksum to avoid self-reference.",
    }


def write_run_manifest(
    output_path: str | Path,
    *,
    result_directory: str | Path,
    config: dict[str, Any],
    validation: dict[str, Any],
    warnings: Iterable[str],
) -> dict[str, Any]:
    manifest = build_run_manifest(
        result_directory=result_directory,
        config=config,
        validation=validation,
        warnings=warnings,
    )
    write_json(manifest, output_path)
    return manifest


def _html_shell(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f5f7fa; --panel:#ffffff; --text:#172033; --muted:#596579; --border:#d9e0e8; --accent:#235ea7; --ok:#1f7a45; --warn:#9a6700; --fail:#b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#11151c; --panel:#1a2029; --text:#edf2f7; --muted:#aab4c3; --border:#35404e; --accent:#78aef2; --ok:#66d391; --warn:#f2c14e; --fail:#ff8178; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--text); line-height:1.55; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 6px; font-size:2rem; }}
h2 {{ margin-top:34px; padding-bottom:7px; border-bottom:1px solid var(--border); }}
h3 {{ margin-bottom:8px; }}
a {{ color:var(--accent); }}
.subtitle, .muted {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; }}
.metric {{ font-size:1.45rem; font-weight:650; margin-top:4px; }}
.status {{ display:inline-block; border:1px solid currentColor; border-radius:999px; padding:4px 10px; font-weight:650; }}
.status-pass {{ color:var(--ok); }} .status-warn {{ color:var(--warn); }} .status-fail {{ color:var(--fail); }}
table {{ width:100%; border-collapse:collapse; margin:12px 0; background:var(--panel); }}
th,td {{ border:1px solid var(--border); padding:9px 10px; text-align:left; vertical-align:top; }}
th {{ background:color-mix(in srgb, var(--panel) 80%, var(--accent) 20%); }}
.table-wrap {{ overflow-x:auto; }}
.figure-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }}
.figure {{ margin:0; background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px; }}
.figure img {{ width:100%; height:auto; display:block; }}
.figure figcaption {{ margin-top:8px; color:var(--muted); }}
.callout {{ border-left:5px solid var(--accent); background:var(--panel); padding:14px 16px; margin:16px 0; }}
.warning {{ border-left-color:var(--warn); }}
code {{ background:var(--panel); border:1px solid var(--border); border-radius:5px; padding:1px 5px; }}
ul.compact li {{ margin:4px 0; }}
footer {{ margin-top:40px; color:var(--muted); border-top:1px solid var(--border); padding-top:16px; }}
</style>
</head>
<body><main>{body}</main></body>
</html>
"""


def write_tle_research_report(
    output_path: str | Path,
    *,
    config: dict[str, Any],
    provenance: dict[str, Any],
    model_summary: dict[str, Any],
    ground_summary: dict[str, Any],
    pass_summaries: dict[str, dict[str, Any]],
    passes_by_station: dict[str, dict[str, list[Any]]],
    validation: dict[str, Any],
    warnings: Iterable[str],
    age_report: dict[str, Any],
    frame_roundtrip: dict[str, Any],
    geodetic_roundtrip: dict[str, Any],
    created_files: Iterable[Path],
) -> Path:
    """Write one combined HTML report for the fixed-TLE research pipeline."""
    path = Path(output_path)
    warning_list = list(dict.fromkeys(str(item) for item in warnings))
    counts = _validation_counts(validation)
    station_id = next(iter(pass_summaries))
    pass_summary = pass_summaries[station_id]
    station = pass_summary["station"]

    model_rows: list[str] = []
    runtime_rows: list[str] = []
    for model, values in model_summary["models"].items():
        model_rows.append(
            "<tr>"
            f"<td>{_e(MODEL_LABELS.get(model, model))}</td>"
            f"<td>{_format_number(values['maximum_position_difference_km'], digits=6)} km</td>"
            f"<td>{_format_number(values['final_position_difference_km'], digits=6)} km</td>"
            f"<td>{_format_number(values['rms_position_difference_km'], digits=6)} km</td>"
            "</tr>"
        )
        runtime_rows.append(
            "<tr>"
            f"<td>{_e(MODEL_LABELS.get(model, model))}</td>"
            f"<td>{_format_number(values.get('runtime_seconds'), digits=6)} s</td>"
            f"<td>{_format_number(values.get('function_evaluations'), digits=0)}</td>"
            f"<td>{_e(values.get('solver_status'))}</td>"
            "</tr>"
        )
    runtime_rows.insert(
        0,
        "<tr>"
        f"<td>{_e(MODEL_LABELS['sgp4'])}</td>"
        f"<td>{_format_number(model_summary.get('sgp4_runtime_seconds'), digits=6)} s</td>"
        "<td>—</td><td>completed</td></tr>",
    )

    ground_rows = [
        "<tr>"
        f"<td>{_e(MODEL_LABELS.get(model, model))}</td>"
        f"<td>{_format_number(values['maximum_surface_separation_km'], digits=6)} km</td>"
        f"<td>{_format_number(values['final_surface_separation_km'], digits=6)} km</td>"
        f"<td>{_format_number(values['maximum_absolute_altitude_difference_km'], digits=6)} km</td>"
        "</tr>"
        for model, values in ground_summary["models"].items()
    ]

    pass_model_rows: list[str] = []
    for model, values in pass_summary["models"].items():
        comparison = pass_summary["comparisons_against_sgp4"].get(model)
        pass_model_rows.append(
            "<tr>"
            f"<td>{_e(MODEL_LABELS.get(model, model))}</td>"
            f"<td>{values['pass_count']}</td>"
            f"<td>{_format_number(values['total_visible_time_seconds'], digits=3)} s</td>"
            f"<td>{_format_number(values['maximum_elevation_deg'], digits=3)}°</td>"
            f"<td>{'—' if comparison is None else comparison['matched_pass_count']}</td>"
            f"<td>{'—' if comparison is None else _format_number(comparison['maximum_absolute_aos_difference_seconds'], digits=3) + ' s'}</td>"
            f"<td>{'—' if comparison is None else _format_number(comparison['maximum_absolute_los_difference_seconds'], digits=3) + ' s'}</td>"
            "</tr>"
        )

    sgp4_pass_rows: list[str] = []
    for event in passes_by_station[station_id]["sgp4"]:
        sgp4_pass_rows.append(
            "<tr>"
            f"<td>{_e(event.pass_id)}</td>"
            f"<td>{_e(event.aos_utc)}</td>"
            f"<td>{_e(event.maximum_elevation_utc)}</td>"
            f"<td>{_e(event.los_utc)}</td>"
            f"<td>{_format_number(event.duration_seconds, digits=2)} s</td>"
            f"<td>{_format_number(event.maximum_elevation_deg, digits=3)}°</td>"
            f"<td>{_format_number(event.closest_range_km, digits=3)} km</td>"
            "</tr>"
        )
    if not sgp4_pass_rows:
        sgp4_pass_rows.append('<tr><td colspan="7">No SGP4 pass above the configured mask.</td></tr>')

    validation_rows = [
        "<tr>"
        f"<td>{_e(item.get('validation_id'))}</td>"
        f"<td>{_e(item.get('name'))}</td>"
        f"<td>{_e(item.get('status'))}</td>"
        f"<td>{_e(item.get('criterion'))}</td>"
        "</tr>"
        for item in validation.get("checks", [])
    ]

    caution_items = list(config.get("scientific_cautions", [])) + warning_list
    caution_html = "".join(f"<li>{_e(item)}</li>" for item in dict.fromkeys(caution_items))
    model_explanation_html = "".join(
        f'<div class="card"><h3>{_e(MODEL_LABELS[model])}</h3><p>{_e(MODEL_EXPLANATIONS[model])}</p></div>'
        for model in ("sgp4", "analytical_two_body", "numerical_two_body", "numerical_j2", "numerical_j2_drag")
    )

    file_links = []
    for file_path in sorted(set(Path(item) for item in created_files)):
        if file_path.is_file() and file_path.parent == path.parent and file_path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            file_links.append(f"<li>{_link(path, file_path)}</li>")
    data_links_html = "".join(file_links)

    figures_html = "".join(
        item
        for item in [
            _figure(path, "sgp4_model_position_separation", "Three-dimensional model separation from SGP4"),
            _figure(path, "sgp4_rtn_position_differences", "Radial, along-track and cross-track differences from SGP4"),
            _figure(path, "ground_track_comparison", "Earth-relative ground-track comparison"),
            _figure(path, "ground_track_surface_separation", "Sub-satellite-point surface separation from SGP4"),
            _figure(path, "bremen_elevation_history", "Bremen elevation history and 10-degree mask"),
            _figure(path, "bremen_sgp4_sky_paths", "SGP4 sky paths above Bremen"),
            _figure(path, "bremen_pass_timing_differences", "Pass timing differences relative to SGP4"),
        ]
        if item
    )

    metadata = provenance.get("metadata", {})
    title = f"{config['experiment']['title']} — Research Report"
    body = f"""
<header>
  <h1>{_e(config['experiment']['title'])}</h1>
  <p class="subtitle">Orbital Propagation Research Core {RESEARCH_CORE_VERSION}</p>
  <p><span class="status {_status_class(str(validation.get('overall_status')))}">{_e(validation.get('overall_status'))}</span></p>
</header>

<section class="grid">
  <div class="card"><div class="muted">Experiment</div><div class="metric">{_e(config['experiment']['experiment_id'])}</div><div>{_e(config['experiment']['case_id'])}</div></div>
  <div class="card"><div class="muted">Duration</div><div class="metric">{_format_number(model_summary['duration_hours'], digits=3)} h</div><div>{model_summary['state_count']} output states</div></div>
  <div class="card"><div class="muted">Validation</div><div class="metric">{counts.get('passed', 0)} passed</div><div>{counts.get('failed', 0)} failed · {len(warning_list)} warnings</div></div>
  <div class="card"><div class="muted">Reference</div><div class="metric">Frozen ISS TLE</div><div>NORAD {_e(provenance.get('norad_catalog_number'))}</div></div>
</section>

<div class="callout">
<strong>How to interpret this report:</strong> SGP4, two-body, J2, and J2-plus-drag are different propagation models. Reported differences are model-to-model separations from a shared initial Cartesian state. They are not measured ISS orbit errors.
</div>

<h2>1. Models used</h2>
<div class="grid">{model_explanation_html}</div>

<h2>2. Frozen TLE provenance</h2>
<div class="table-wrap"><table><tbody>
<tr><th>Object</th><td>{_e(provenance.get('object_name', provenance.get('name', 'ISS')))}</td></tr>
<tr><th>NORAD catalog</th><td>{_e(provenance.get('norad_catalog_number'))}</td></tr>
<tr><th>TLE epoch</th><td>{_e(provenance.get('tle_epoch_utc'))}</td></tr>
<tr><th>Run-end TLE age</th><td>{_format_number(age_report.get('end_age_hours'), digits=6)} h</td></tr>
<tr><th>Frozen TLE SHA-256</th><td><code>{_e(provenance.get('tle_file_sha256', provenance.get('file_sha256', '')))}</code></td></tr>
<tr><th>Source</th><td>{_e(metadata.get('source_name', metadata.get('source', 'CelesTrak frozen snapshot')))}</td></tr>
</tbody></table></div>

<h2>3. Frame and coordinate validation</h2>
<section class="grid">
<div class="card"><div class="muted">TEME↔GCRS position round trip</div><div class="metric">{_format_number(frame_roundtrip.get('position_roundtrip_error_m'), digits=6)} m</div></div>
<div class="card"><div class="muted">WGS-84 geodetic reconstruction</div><div class="metric">{_format_number(geodetic_roundtrip.get('maximum_position_residual_m'), digits=6)} m</div></div>
<div class="card"><div class="muted">Comparison frame</div><div class="metric">{_e(model_summary.get('frame'))}</div></div>
<div class="card"><div class="muted">Earth-fixed frame</div><div class="metric">{_e(ground_summary.get('frame'))}</div></div>
</section>

<h2>4. Three-dimensional trajectory comparison</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Maximum separation</th><th>Final separation</th><th>RMS separation</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table></div>

<h2>5. Ground-track comparison</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Maximum surface separation</th><th>Final surface separation</th><th>Maximum altitude difference</th></tr></thead><tbody>{''.join(ground_rows)}</tbody></table></div>

<h2>6. Ground-station passes — {_e(station['name'])}</h2>
<p>Station: {_format_number(station['latitude_deg'], digits=4)}° latitude, {_format_number(station['longitude_deg'], digits=4)}° longitude, {_format_number(station['altitude_m'], digits=1)} m altitude; minimum elevation {_format_number(station['minimum_elevation_deg'], digits=1)}°.</p>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Pass count</th><th>Total visible time</th><th>Highest pass</th><th>Matched vs SGP4</th><th>Max |AOS Δ|</th><th>Max |LOS Δ|</th></tr></thead><tbody>{''.join(pass_model_rows)}</tbody></table></div>
<h3>SGP4 pass schedule</h3>
<div class="table-wrap"><table><thead><tr><th>Pass</th><th>AOS UTC</th><th>Maximum UTC</th><th>LOS UTC</th><th>Duration</th><th>Maximum elevation</th><th>Closest range</th></tr></thead><tbody>{''.join(sgp4_pass_rows)}</tbody></table></div>

<h2>7. Runtime</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Runtime</th><th>Function evaluations</th><th>Solver status</th></tr></thead><tbody>{''.join(runtime_rows)}</tbody></table></div>

<h2>8. Figures</h2>
<div class="figure-grid">{figures_html}</div>

<h2>9. Validation details</h2>
<div class="table-wrap"><table><thead><tr><th>ID</th><th>Check</th><th>Status</th><th>Criterion</th></tr></thead><tbody>{''.join(validation_rows)}</tbody></table></div>

<h2>10. Assumptions, warnings and limitations</h2>
<div class="callout warning"><ul class="compact">{caution_html}</ul></div>

<h2>11. Result files</h2>
<p>The following files contain the exact configuration, state histories, model comparisons, pass tables, diagnostics, and technical summaries used by this report.</p>
<ul class="compact">{data_links_html}</ul>
<p>Integrity inventory: <a href="RUN_MANIFEST.json">RUN_MANIFEST.json</a> · Compact validation result: <a href="FINAL_VALIDATION_SUMMARY.json">FINAL_VALIDATION_SUMMARY.json</a></p>

<footer>Generated {_e(_utc_now_iso())} · Build {_e(BUILD_MARKER)} · Research software only; not flight-qualified.</footer>
"""
    path.write_text(_html_shell(title=title, body=body), encoding="utf-8", newline="\n")
    return path


def write_controlled_research_report(
    output_path: str | Path,
    *,
    config: dict[str, Any],
    orbit_summary: dict[str, Any],
    two_body_summary: dict[str, Any],
    j2_validation: dict[str, Any],
    drag_validation: dict[str, Any],
    validation: dict[str, Any],
    warnings: Iterable[str],
    created_files: Iterable[Path],
) -> Path:
    """Write a compact combined report for the controlled LEO benchmark."""
    path = Path(output_path)
    counts = _validation_counts(validation)
    warning_list = list(dict.fromkeys(str(item) for item in warnings))
    file_links = "".join(
        f"<li>{_link(path, item)}</li>"
        for item in sorted(set(Path(value) for value in created_files))
        if item.is_file() and item.parent == path.parent and item.suffix.lower() in {".csv", ".json", ".md", ".txt"}
    )
    figures = "".join(
        item
        for item in [
            _figure(path, "position_difference_vs_time", "Numerical versus analytical two-body position difference"),
            _figure(path, "raan_evolution", "Numerical and analytical J2 RAAN evolution"),
            _figure(path, "j2_two_body_position_separation", "J2 separation from two-body"),
            _figure(path, "drag_semi_major_axis_difference", "Drag semi-major-axis difference relative to J2"),
            _figure(path, "drag_along_track_separation", "Drag along-track separation relative to J2"),
            _figure(path, "drag_sensitivity_semi_major_axis", "One-at-a-time drag sensitivity"),
        ]
        if item
    )
    cautions = list(config.get("scientific_cautions", [])) + warning_list
    body = f"""
<header><h1>{_e(config['experiment']['title'])}</h1><p class="subtitle">Orbital Propagation Research Core {RESEARCH_CORE_VERSION}</p><p><span class="status {_status_class(str(validation.get('overall_status')))}">{_e(validation.get('overall_status'))}</span></p></header>
<section class="grid">
<div class="card"><div class="muted">Orbital period</div><div class="metric">{_format_number(orbit_summary.get('orbital_period_minutes'), digits=4)} min</div></div>
<div class="card"><div class="muted">Two-body max difference</div><div class="metric">{_format_number(two_body_summary['position_difference_m']['maximum_absolute'], digits=6)} m</div></div>
<div class="card"><div class="muted">J2 RAAN rate</div><div class="metric">{_format_number(j2_validation.get('fitted_raan_rate_deg_day'), digits=6)}°/day</div></div>
<div class="card"><div class="muted">Drag Δa versus J2</div><div class="metric">{_format_number(drag_validation.get('final_semi_major_axis_difference_vs_j2_m'), digits=3)} m</div></div>
</section>
<div class="callout"><strong>Interpretation:</strong> analytical and numerical two-body use identical ideal physics; J2 and drag are added physical-model terms. The comparison verifies implementation and explores model sensitivity, not measured orbit truth.</div>
<h2>Validation</h2><p>{counts.get('passed', 0)} passed · {counts.get('failed', 0)} failed · {len(warning_list)} warnings.</p>
<h2>Figures</h2><div class="figure-grid">{figures}</div>
<h2>Assumptions and limitations</h2><div class="callout warning"><ul class="compact">{''.join(f'<li>{_e(item)}</li>' for item in dict.fromkeys(cautions))}</ul></div>
<h2>Result files</h2><ul class="compact">{file_links}</ul><p><a href="RUN_MANIFEST.json">RUN_MANIFEST.json</a> · <a href="FINAL_VALIDATION_SUMMARY.json">FINAL_VALIDATION_SUMMARY.json</a></p>
<footer>Generated {_e(_utc_now_iso())} · Build {_e(BUILD_MARKER)} · Research software only; not flight-qualified.</footer>
"""
    path.write_text(_html_shell(title=f"{config['experiment']['title']} — Research Report", body=body), encoding="utf-8", newline="\n")
    return path
