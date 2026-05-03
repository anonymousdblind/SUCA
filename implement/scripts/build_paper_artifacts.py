"""Build paper-ready tables and figures from evaluation artifacts.

The script consumes a JSON manifest describing where benchmark summaries,
training diagnostics, timing profiles, and qualitative images live.
It outputs CSV tables, LaTeX row fragments, plots, and a markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple



GENEVAL_SKILLS = ["Object", "Attribute", "Count", "Position", "Verb"]
GENEVAL_MAIN_BUCKETS = {
    "Atom 3--4": [3, 4],
    "Atom 5--6": [5, 6],
    "Atom 7--10": [7, 8, 9, 10],
}
GENEVAL_APPENDIX_BUCKETS = {
    "3--4": [3, 4],
    "5--6": [5, 6],
    "7--8": [7, 8],
    "9--10": [9, 10],
}
SPATIAL_COLUMNS = ["Left/Right", "Above/Below", "Inside/Around", "Front/Behind"]
DPG_COLUMNS = ["Entity", "Attribute", "Relation"]
PLACEHOLDER = "--"


def resolve_path(root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_key(text: str) -> str:
    return text.strip().lower().replace("_", "").replace("-", "").replace("/", "").replace(" ", "")


def lookup_value(mapping: Dict[str, float], target: str, aliases: Optional[Sequence[str]] = None) -> Optional[float]:
    aliases = list(aliases or [])
    aliases.insert(0, target)
    normalized = {normalize_key(key): value for key, value in mapping.items()}
    for alias in aliases:
        key = normalize_key(alias)
        if key in normalized:
            return normalized[key]
    return None


def format_value(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return PLACEHOLDER
    return f"{value:.{digits}f}"


def compute_bucket_counts(metadata_path: Optional[Path]) -> Dict[int, int]:
    if metadata_path is None or not metadata_path.exists():
        return {}

    counts: Dict[int, int] = defaultdict(int)
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            atom_count = sample.get("atom_count")
            if atom_count is not None:
                counts[int(atom_count)] += 1
    return dict(counts)


def aggregate_bucket(atom_scores: Dict[str, float], atoms: Iterable[int], counts: Dict[int, int]) -> Optional[float]:
    values: List[Tuple[float, int]] = []
    for atom in atoms:
        value = lookup_value(atom_scores, str(atom))
        if value is None:
            continue
        weight = counts.get(atom, 1)
        values.append((value, weight))

    if not values:
        return None

    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def normalize_benchmark_summary(payload: Dict[str, object], fmt: str, model_key: Optional[str]) -> Dict[str, object]:
    if fmt == "normalized_benchmark":
        return payload

    if fmt == "formal_comparison":
        if model_key is None:
            raise ValueError("formal_comparison sources require model_key")
        model_payload = payload[model_key]
        return {
            "overall": {"soft_tifa_gm": model_payload.get("soft_tifa_gm")},
            "skills": model_payload.get("per_skill", {}),
            "atom_count": model_payload.get("per_atom", {}),
        }

    if fmt == "comparison_report":
        if model_key is None:
            raise ValueError("comparison_report sources require model_key")
        model_payload = payload["results"][model_key]
        skills = {skill: model_payload[skill] for skill in GENEVAL_SKILLS if skill in model_payload}
        return {
            "overall": {"soft_tifa_gm": model_payload.get("soft_tifa_gm"), "Overall": model_payload.get("Overall")},
            "skills": skills,
            "dimensions": payload.get("dimensions", {}),
            "categories": payload.get("categories", {}),
        }

    raise ValueError(f"Unsupported summary format: {fmt}")


def load_run_summary(root: Path, spec: Dict[str, object]) -> Dict[str, object]:
    path = resolve_path(root, str(spec["path"]))
    payload = load_json(path)
    fmt = str(spec.get("format", "normalized_benchmark"))
    model_key = spec.get("model_key")
    return normalize_benchmark_summary(payload, fmt, str(model_key) if model_key is not None else None)


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_tex_rows(path: Path, rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" & ".join(row) + r" \\" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_geneval_tables(root: Path, manifest: Dict[str, object], summaries: Dict[str, Dict[str, object]], output_dir: Path) -> List[str]:
    benchmark = manifest.get("benchmarks", {}).get("geneval2")
    if not benchmark:
        return []

    metadata_path = None
    if benchmark.get("metadata_path"):
        metadata_path = resolve_path(root, str(benchmark["metadata_path"]))
    counts = compute_bucket_counts(metadata_path)

    methods_order = manifest.get("methods_order", list(benchmark["runs"].keys()))
    headers = ["Method", *GENEVAL_SKILLS, *GENEVAL_MAIN_BUCKETS.keys()]
    rows: List[List[str]] = []
    appendix_rows: List[List[str]] = []

    for method in methods_order:
        summary = summaries.get(f"geneval2::{method}")
        if summary is None and method not in benchmark["runs"]:
            continue

        skill_map = dict(summary.get("skills", {})) if summary else {}
        atom_scores = dict(summary.get("atom_count", {})) if summary else {}

        row = [method]
        for skill in GENEVAL_SKILLS:
            row.append(format_value(lookup_value(skill_map, skill), digits=2))
        for label, atoms in GENEVAL_MAIN_BUCKETS.items():
            row.append(format_value(aggregate_bucket(atom_scores, atoms, counts), digits=2))
        rows.append(row)

    for label, atoms in GENEVAL_APPENDIX_BUCKETS.items():
        base = summaries.get("geneval2::SD3.5-Medium")
        flow = summaries.get("geneval2::Flow-GRPO w/o \\method")
        suca = summaries.get("geneval2::\\method")
        base_value = aggregate_bucket(dict(base.get("atom_count", {})) if base else {}, atoms, counts)
        flow_value = aggregate_bucket(dict(flow.get("atom_count", {})) if flow else {}, atoms, counts)
        suca_value = aggregate_bucket(dict(suca.get("atom_count", {})) if suca else {}, atoms, counts)
        delta = None if suca_value is None or flow_value is None else suca_value - flow_value
        appendix_rows.append([
            label,
            format_value(base_value),
            format_value(flow_value),
            format_value(suca_value),
            format_value(delta),
        ])

    csv_path = output_dir / "tables" / "geneval_main.csv"
    tex_path = output_dir / "tables" / "geneval_main_rows.tex"
    write_csv(csv_path, headers, rows)
    write_tex_rows(tex_path, rows)

    appendix_csv = output_dir / "tables" / "geneval_atom_v3.csv"
    appendix_tex = output_dir / "tables" / "geneval_atom_v3_rows.tex"
    write_csv(appendix_csv, ["Atom count", "Base", "Flow-GRPO", "\\method", "Delta"], appendix_rows)
    write_tex_rows(appendix_tex, appendix_rows)

    return [str(csv_path), str(tex_path), str(appendix_csv), str(appendix_tex)]


def build_spatial_tables(manifest: Dict[str, object], summaries: Dict[str, Dict[str, object]], output_dir: Path) -> List[str]:
    benchmark = manifest.get("benchmarks", {}).get("spatialgeneval")
    if not benchmark:
        return []

    methods_order = manifest.get("methods_order", list(benchmark["runs"].keys()))
    rows: List[List[str]] = []
    appendix_rows: List[List[str]] = []

    for method in methods_order:
        summary = summaries.get(f"spatialgeneval::{method}")
        if summary is None and method not in benchmark["runs"]:
            continue
        dimensions = dict(summary.get("dimensions", {})) if summary else {}
        overall = dict(summary.get("overall", {})) if summary else {}
        row = [method]
        for label in SPATIAL_COLUMNS:
            row.append(format_value(lookup_value(dimensions, label), digits=2))
        row.append(format_value(lookup_value(overall, "Overall", aliases=["avg_acc"]), digits=2))
        rows.append(row)

    for label in SPATIAL_COLUMNS:
        base = summaries.get("spatialgeneval::SD3.5-Medium")
        flow = summaries.get("spatialgeneval::Flow-GRPO w/o \\method")
        suca = summaries.get("spatialgeneval::\\method")
        base_value = lookup_value(dict(base.get("dimensions", {})) if base else {}, label)
        flow_value = lookup_value(dict(flow.get("dimensions", {})) if flow else {}, label)
        suca_value = lookup_value(dict(suca.get("dimensions", {})) if suca else {}, label)
        delta = None if suca_value is None or flow_value is None else suca_value - flow_value
        appendix_rows.append([
            label,
            format_value(base_value),
            format_value(flow_value),
            format_value(suca_value),
            format_value(delta),
        ])

    csv_path = output_dir / "tables" / "spatial_main.csv"
    tex_path = output_dir / "tables" / "spatial_main_rows.tex"
    write_csv(csv_path, ["Method", *SPATIAL_COLUMNS, "Overall"], rows)
    write_tex_rows(tex_path, rows)

    appendix_csv = output_dir / "tables" / "spatial_dim_v3.csv"
    appendix_tex = output_dir / "tables" / "spatial_dim_v3_rows.tex"
    write_csv(appendix_csv, ["Dimension", "Base", "Flow-GRPO", "\\method", "Delta"], appendix_rows)
    write_tex_rows(appendix_tex, appendix_rows)
    return [str(csv_path), str(tex_path), str(appendix_csv), str(appendix_tex)]


def build_dpg_tables(manifest: Dict[str, object], summaries: Dict[str, Dict[str, object]], output_dir: Path) -> List[str]:
    benchmark = manifest.get("benchmarks", {}).get("dpgbench")
    if not benchmark:
        return []

    methods_order = manifest.get("methods_order", list(benchmark["runs"].keys()))
    rows: List[List[str]] = []
    appendix_rows: List[List[str]] = []

    for method in methods_order:
        summary = summaries.get(f"dpgbench::{method}")
        if summary is None and method not in benchmark["runs"]:
            continue
        categories = dict(summary.get("categories", {})) if summary else {}
        overall = dict(summary.get("overall", {})) if summary else {}
        row = [method]
        for label in DPG_COLUMNS:
            row.append(format_value(lookup_value(categories, label), digits=2))
        row.append(format_value(lookup_value(overall, "Overall"), digits=2))
        rows.append(row)

    for label in DPG_COLUMNS:
        base = summaries.get("dpgbench::SD3.5-Medium")
        flow = summaries.get("dpgbench::Flow-GRPO w/o \\method")
        suca = summaries.get("dpgbench::\\method")
        base_value = lookup_value(dict(base.get("categories", {})) if base else {}, label)
        flow_value = lookup_value(dict(flow.get("categories", {})) if flow else {}, label)
        suca_value = lookup_value(dict(suca.get("categories", {})) if suca else {}, label)
        delta = None if suca_value is None or flow_value is None else suca_value - flow_value
        appendix_rows.append([
            label,
            format_value(base_value),
            format_value(flow_value),
            format_value(suca_value),
            format_value(delta),
        ])

    csv_path = output_dir / "tables" / "dpg_main.csv"
    tex_path = output_dir / "tables" / "dpg_main_rows.tex"
    write_csv(csv_path, ["Method", *DPG_COLUMNS, "Overall"], rows)
    write_tex_rows(tex_path, rows)

    appendix_csv = output_dir / "tables" / "dpg_cat_v3.csv"
    appendix_tex = output_dir / "tables" / "dpg_cat_v3_rows.tex"
    write_csv(appendix_csv, ["Category", "Base", "Flow-GRPO", "\\method", "Delta"], appendix_rows)
    write_tex_rows(appendix_tex, appendix_rows)
    return [str(csv_path), str(tex_path), str(appendix_csv), str(appendix_tex)]


def build_ablation_table(manifest: Dict[str, object], summaries: Dict[str, Dict[str, object]], output_dir: Path) -> List[str]:
    ablations = manifest.get("ablations")
    if not ablations:
        return []

    rows: List[List[str]] = []
    for entry in ablations:
        name = entry["name"]
        geneval_ref = entry.get("geneval_ref") or entry.get("ref")
        spatial_ref = entry.get("spatial_ref") or entry.get("ref")
        dpg_ref = entry.get("dpg_ref") or entry.get("ref")

        geneval = summaries.get(f"geneval2::{geneval_ref}") if geneval_ref else None
        spatial = summaries.get(f"spatialgeneval::{spatial_ref}") if spatial_ref else None
        dpg = summaries.get(f"dpgbench::{dpg_ref}") if dpg_ref else None

        rows.append([
            name,
            format_value(lookup_value(dict(geneval.get("overall", {})) if geneval else {}, "soft_tifa_gm")),
            format_value(lookup_value(dict(spatial.get("overall", {})) if spatial else {}, "avg_acc", aliases=["Overall"])),
            format_value(lookup_value(dict(dpg.get("overall", {})) if dpg else {}, "Overall")),
            str(entry.get("relative_cost", PLACEHOLDER)),
        ])

    csv_path = output_dir / "tables" / "ablations.csv"
    tex_path = output_dir / "tables" / "ablations_rows.tex"
    write_csv(csv_path, ["Configuration", "GenEval2", "SpatialGenEval", "DPG-Bench", "Relative Cost"], rows)
    write_tex_rows(tex_path, rows)
    return [str(csv_path), str(tex_path)]


def build_config_table(manifest: Dict[str, object], output_dir: Path) -> List[str]:
    config_rows = manifest.get("training_config")
    if not config_rows:
        return []

    rows = [[str(cell) for cell in row] for row in config_rows]
    csv_path = output_dir / "tables" / "config_v3.csv"
    tex_path = output_dir / "tables" / "config_v3_rows.tex"
    write_csv(csv_path, ["Component", "Value", "Notes"], rows)
    write_tex_rows(tex_path, rows)
    return [str(csv_path), str(tex_path)]


def build_time_breakdown(root: Path, manifest: Dict[str, object], output_dir: Path) -> List[str]:
    spec = manifest.get("time_breakdown")
    if not spec:
        return []
    payload = load_json(resolve_path(root, str(spec["path"])))
    rows = []
    for stage in ["Image sampling", "Semantic parsing", "VLM verification", "Routing aggregation", "Backward step"]:
        value = payload.get(stage)
        rows.append([stage, format_value(value), spec.get("interpretation", {}).get(stage, PLACEHOLDER)])
    csv_path = output_dir / "tables" / "time_v3.csv"
    tex_path = output_dir / "tables" / "time_v3_rows.tex"
    write_csv(csv_path, ["Stage", "Time share", "Interpretation"], rows)
    write_tex_rows(tex_path, rows)
    return [str(csv_path), str(tex_path)]


def parse_reward_curve_csv(path: Path, label: str) -> Dict[str, List[Tuple[float, float]]]:
    curves: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        lower_fields = {name.lower(): name for name in fields}
        long_format = all(key in lower_fields for key in ["step", "unit_type", "value"])

        for row in reader:
            if long_format:
                unit_type = row[lower_fields["unit_type"]]
                step = float(row[lower_fields["step"]])
                value = float(row[lower_fields["value"]])
                method_name = row.get(lower_fields.get("method", ""), label)
                curves[f"{method_name}::{unit_type}"] .append((step, value))
                continue

            step_field = lower_fields.get("step")
            if step_field is None:
                raise ValueError(f"Reward curve CSV requires a step column: {path}")
            step = float(row[step_field])
            for unit_type in ["Attribute", "Count", "Entity", "Relation"]:
                if unit_type in row and row[unit_type]:
                    curves[f"{label}::{unit_type}"].append((step, float(row[unit_type])))
    return curves


def build_reward_curves(root: Path, manifest: Dict[str, object], output_dir: Path) -> List[str]:
    reward_curves = manifest.get("reward_curves")
    if not reward_curves:
        return []

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to build reward curve figures") from exc

    all_curves: Dict[str, List[Tuple[float, float]]] = {}
    for entry in reward_curves:
        label = entry["label"]
        path = resolve_path(root, str(entry["path"]))
        all_curves.update(parse_reward_curve_csv(path, label))

    figure_path = output_dir / "figures" / "reward_curves_v3.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    unit_types = ["Attribute", "Count", "Entity", "Relation"]
    color_map = {
        "Flow-GRPO w/o \\method": "#666666",
        "\\method": "#1f77b4",
    }
    for axis, unit_type in zip(axes.flatten(), unit_types):
        for key, points in sorted(all_curves.items()):
            method_name, curve_unit = key.split("::", 1)
            if curve_unit != unit_type:
                continue
            points = sorted(points)
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            axis.plot(xs, ys, label=method_name, linewidth=2.0, color=color_map.get(method_name))
        axis.set_title(unit_type)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Step")
        axis.set_ylabel("Reward")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return [str(figure_path)]


def build_qualitative_grid(root: Path, manifest: Dict[str, object], output_dir: Path) -> List[str]:
    qualitative = manifest.get("qualitative")
    if not qualitative:
        return []

    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to build qualitative figure grids") from exc

    rows = qualitative.get("rows", [])
    columns = qualitative.get("columns", [])
    if not rows or not columns:
        return []

    cell_width = int(qualitative.get("cell_width", 320))
    cell_height = int(qualitative.get("cell_height", 320))
    label_width = int(qualitative.get("label_width", 240))
    header_height = 60
    canvas = Image.new(
        "RGB",
        (label_width + cell_width * len(columns), header_height + cell_height * len(rows)),
        color="white",
    )
    drawer = ImageDraw.Draw(canvas)

    for column_index, column_name in enumerate(columns):
        x = label_width + column_index * cell_width + 10
        drawer.text((x, 18), column_name, fill="black")

    for row_index, row_spec in enumerate(rows):
        y = header_height + row_index * cell_height + 12
        drawer.text((12, y), row_spec["label"], fill="black")
        for column_index, column_name in enumerate(columns):
            image_path = resolve_path(root, row_spec["images"][column_name])
            image = Image.open(image_path).convert("RGB")
            image = image.resize((cell_width, cell_height))
            x = label_width + column_index * cell_width
            canvas.paste(image, (x, header_height + row_index * cell_height))

    output_path = output_dir / "figures" / str(qualitative.get("output_name", "qualitative_grid.png"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return [str(output_path)]


def build_partial_results(root: Path, manifest: Dict[str, object], summaries: Dict[str, Dict[str, object]], output_dir: Path) -> List[str]:
    partial = manifest.get("partial_results")
    if not partial:
        return []

    diagnostics = load_json(resolve_path(root, str(partial["training_diagnostics"]["path"])))
    geneval_base = summaries.get("geneval2::SD3.5-Medium")
    geneval_sft = summaries.get("geneval2::SFT warmup")
    spatial_base = summaries.get("spatialgeneval::SD3.5-Medium")
    spatial_sft = summaries.get("spatialgeneval::SFT warmup")
    best_rl = diagnostics.get("best_rl_run", {})

    rows = [
        ["GenEval2 soft_tifa_gm", format_value(lookup_value(dict(geneval_base.get("overall", {})) if geneval_base else {}, "soft_tifa_gm")), format_value(lookup_value(dict(geneval_sft.get("overall", {})) if geneval_sft else {}, "soft_tifa_gm")), PLACEHOLDER],
        ["GenEval2 Attribute", format_value(lookup_value(dict(geneval_base.get("skills", {})) if geneval_base else {}, "Attribute")), format_value(lookup_value(dict(geneval_sft.get("skills", {})) if geneval_sft else {}, "Attribute")), PLACEHOLDER],
        ["GenEval2 Count", format_value(lookup_value(dict(geneval_base.get("skills", {})) if geneval_base else {}, "Count")), format_value(lookup_value(dict(geneval_sft.get("skills", {})) if geneval_sft else {}, "Count")), PLACEHOLDER],
        ["SpatialGenEval avg_acc", format_value(lookup_value(dict(spatial_base.get("overall", {})) if spatial_base else {}, "avg_acc", aliases=["Overall"])), format_value(lookup_value(dict(spatial_sft.get("overall", {})) if spatial_sft else {}, "avg_acc", aliases=["Overall"])), PLACEHOLDER],
        ["ImageReward", PLACEHOLDER, PLACEHOLDER, format_value(best_rl.get("imagereward"))],
        ["Attribute reward change", PLACEHOLDER, PLACEHOLDER, format_value(best_rl.get("attribute_reward_change_pct"), digits=0) + "%" if best_rl.get("attribute_reward_change_pct") is not None else PLACEHOLDER],
        ["Advantage separability", PLACEHOLDER, PLACEHOLDER, format_value(best_rl.get("advantage_separability_x"), digits=0) + "x" if best_rl.get("advantage_separability_x") is not None else PLACEHOLDER],
    ]

    csv_path = output_dir / "tables" / "partial_results.csv"
    tex_path = output_dir / "tables" / "partial_results_rows.tex"
    write_csv(csv_path, ["Metric", "Base SD3.5", "SFT warmup", "Best RL run"], rows)
    write_tex_rows(tex_path, rows)
    return [str(csv_path), str(tex_path)]


def build_report(output_dir: Path, generated_files: List[str], warnings: List[str]) -> Path:
    report_path = output_dir / "reports" / "paper_artifact_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Paper Artifact Build Report", "", "## Generated Files", ""]
    for file_name in generated_files:
        lines.append(f"- {file_name}")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper tables and figures from evaluation artifacts")
    parser.add_argument("--manifest", required=True, help="JSON manifest describing artifact sources")
    parser.add_argument("--output-dir", default="paper_artifacts", help="Directory to store generated tables and figures")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    root = manifest_path.parent.parent if manifest_path.parent.name == "docs" else manifest_path.parent
    manifest = load_json(manifest_path)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict[str, object]] = {}
    warnings: List[str] = []
    for benchmark_name, benchmark_spec in manifest.get("benchmarks", {}).items():
        for method, source_spec in benchmark_spec.get("runs", {}).items():
            try:
                summaries[f"{benchmark_name}::{method}"] = load_run_summary(root, source_spec)
            except FileNotFoundError:
                warnings.append(f"Missing source for {benchmark_name}::{method}: {source_spec['path']}")
            except Exception as exc:
                warnings.append(f"Failed to load {benchmark_name}::{method}: {exc}")

    generated_files: List[str] = []
    generated_files.extend(build_geneval_tables(root, manifest, summaries, output_dir))
    generated_files.extend(build_spatial_tables(manifest, summaries, output_dir))
    generated_files.extend(build_dpg_tables(manifest, summaries, output_dir))
    generated_files.extend(build_partial_results(root, manifest, summaries, output_dir))
    generated_files.extend(build_ablation_table(manifest, summaries, output_dir))
    generated_files.extend(build_config_table(manifest, output_dir))

    try:
        generated_files.extend(build_time_breakdown(root, manifest, output_dir))
    except FileNotFoundError as exc:
        warnings.append(f"Time breakdown file missing: {exc}")

    try:
        generated_files.extend(build_reward_curves(root, manifest, output_dir))
    except FileNotFoundError as exc:
        warnings.append(f"Reward curve file missing: {exc}")
    except ValueError as exc:
        warnings.append(f"Reward curve build failed: {exc}")
    except RuntimeError as exc:
        warnings.append(f"Reward curve build skipped: {exc}")

    try:
        generated_files.extend(build_qualitative_grid(root, manifest, output_dir))
    except FileNotFoundError as exc:
        warnings.append(f"Qualitative image missing: {exc}")
    except RuntimeError as exc:
        warnings.append(f"Qualitative figure build skipped: {exc}")

    report_path = build_report(output_dir, generated_files, warnings)
    print(f"Generated {len(generated_files)} artifact files")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()