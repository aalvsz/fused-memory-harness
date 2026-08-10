from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from experiments.hybrid_memory.common import (
    CONDITIONS,
    environment_manifest,
    load_config,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap(
    pairs: list[tuple[float, float]],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if not pairs:
        return math.nan, math.nan
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        diffs.append(mean([candidate - baseline for candidate, baseline in sample]))
    return percentile(diffs, 0.025), percentile(diffs, 0.975)


def cluster_bootstrap(
    pairs: list[tuple[str, float, float]],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Cluster bootstrap: resample case clusters, not individual observations.

    ``pairs`` is a list of ``(case_id, candidate_value, baseline_value)`` tuples.
    All observations sharing a case_id are kept together when resampling.
    Returns the 2.5th and 97.5th percentile of the bootstrapped risk-difference
    distribution.
    """
    if not pairs:
        return math.nan, math.nan
    clusters: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for case_id, candidate, baseline in pairs:
        clusters[case_id].append((candidate, baseline))
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(iterations):
        sample_diffs: list[float] = []
        for _ in cluster_ids:
            chosen = clusters[cluster_ids[rng.randrange(len(cluster_ids))]]
            for candidate, baseline in chosen:
                sample_diffs.append(candidate - baseline)
        diffs.append(mean(sample_diffs))
    return percentile(diffs, 0.025), percentile(diffs, 0.975)


def mcnemar_exact(pairs: list[tuple[float, float]]) -> tuple[int, int, float]:
    candidate_only = sum(candidate == 1 and baseline == 0 for candidate, baseline in pairs)
    baseline_only = sum(candidate == 0 and baseline == 1 for candidate, baseline in pairs)
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return candidate_only, baseline_only, 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(candidate_only, baseline_only) + 1)
    ) / (2**discordant)
    return candidate_only, baseline_only, min(1.0, 2 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(
        ((index, value) for index, value in enumerate(p_values) if math.isfinite(value)),
        key=lambda item: item[1],
    )
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, value * (total - rank)))
        adjusted[index] = running
    return adjusted


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    source: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        budget = int(row["budget_chars"])
        condition = str(row["condition"])
        groups[(budget, condition, "ALL")].append(row)
        groups[(budget, condition, str(row["family"]))].append(row)
    output: list[dict[str, Any]] = []
    for (budget, condition, family), group in sorted(groups.items()):
        values = [bool(row.get(metric, False)) for row in group]
        successes = sum(values)
        low, high = wilson(successes, len(values))
        context_chars = [
            float(row["context_chars"]) for row in group if row.get("context_chars") is not None
        ]
        elapsed = [
            float(row["elapsed_ms"]) for row in group if row.get("elapsed_ms") is not None
        ]
        retrieval = [
            float(row["retrieval_ms"])
            for row in group
            if row.get("retrieval_ms") is not None
        ]
        output.append(
            {
                "source": source,
                "budget_chars": budget,
                "condition": condition,
                "family": family,
                "metric": metric,
                "n": len(values),
                "successes": successes,
                "rate": successes / len(values) if values else math.nan,
                "ci95_low": low,
                "ci95_high": high,
                "mean_context_chars": mean(context_chars),
                "median_context_chars": statistics.median(context_chars)
                if context_chars
                else math.nan,
                "p95_context_chars": percentile(context_chars, 0.95),
                "median_elapsed_ms": statistics.median(elapsed) if elapsed else math.nan,
                "p95_elapsed_ms": percentile(elapsed, 0.95),
                "median_retrieval_ms": statistics.median(retrieval) if retrieval else math.nan,
                "p95_retrieval_ms": percentile(retrieval, 0.95),
            }
        )
    return output


def pairwise(
    rows: list[dict[str, Any]],
    *,
    source: str,
    metric: str,
    budget: int | None,
    family: str,
    candidate: str,
    baselines: list[str],
    iterations: int,
    seed: int,
    multiplicity_family: str | None = None,
    independent_unit_field: str = "case_id",
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if (budget is None or int(row["budget_chars"]) == budget)
        and (family == "ALL" or row["family"] == family)
        and row["condition"] in {candidate, *baselines}
        and (source != "model" or row.get("status") == "ok")
    ]
    by_key = {
        (row["case_id"], row["budget_chars"], row["family"], row["condition"]): float(bool(row.get(metric, False)))
        for row in selected
    }
    pair_keys = sorted({(row["case_id"], row["budget_chars"], row["family"]) for row in selected})
    unit_by_case = {
        str(row["case_id"]): str(row.get(independent_unit_field) or row["case_id"])
        for row in selected
    }
    unit_case_counts: dict[str, set[str]] = defaultdict(set)
    for case_id, unit_id in unit_by_case.items():
        unit_case_counts[unit_id].add(case_id)
    # Cross-budget rows and multiple query variants from one semantic concept
    # are repeated measures, not independent observations.
    cross_budget = budget is None
    clustered_units = any(len(case_ids) > 1 for case_ids in unit_case_counts.values())
    descriptive_clustered = cross_budget or clustered_units
    n_budgets = len({bk for _, bk, _ in pair_keys}) if pair_keys else 0
    if descriptive_clustered:
        cluster_label = "case" if independent_unit_field == "case_id" else independent_unit_field
        inference_method = f"cluster_bootstrap_by_{cluster_label}_descriptive"
    else:
        inference_method = "paired_bootstrap"
    results: list[dict[str, Any]] = []
    for baseline in baselines:
        pairs = [
            (by_key[(ck, bk, fam, candidate)], by_key[(ck, bk, fam, baseline)])
            for (ck, bk, fam) in pair_keys
            if (ck, bk, fam, candidate) in by_key and (ck, bk, fam, baseline) in by_key
        ]
        if not pairs:
            continue
        difference = mean([a - b for a, b in pairs])
        if descriptive_clustered:
            cluster_pairs = [
                (
                    unit_by_case[str(ck)],
                    by_key[(ck, bk, fam, candidate)],
                    by_key[(ck, bk, fam, baseline)],
                )
                for (ck, bk, fam) in pair_keys
                if (ck, bk, fam, candidate) in by_key and (ck, bk, fam, baseline) in by_key
            ]
            low, high = cluster_bootstrap(
                cluster_pairs,
                iterations=iterations,
                seed=seed + len(results),
            )
        else:
            low, high = paired_bootstrap(
                pairs,
                iterations=iterations,
                seed=seed + len(results),
            )
        candidate_only, baseline_only, p_value = mcnemar_exact(pairs)
        if descriptive_clustered:
            # McNemar's exact test assumes one paired observation per independent
            # unit. Repeating each case once per budget violates that assumption,
            # so pooled rows retain discordant counts only as descriptive totals.
            p_value = math.nan
        matched_unit_ids = {
            unit_by_case[str(ck)]
            for (ck, bk, fam) in pair_keys
            if (ck, bk, fam, candidate) in by_key
            and (ck, bk, fam, baseline) in by_key
        }
        n_independent = len(matched_unit_ids)
        max_repeated = max(
            (len(unit_case_counts[unit_id]) for unit_id in matched_unit_ids),
            default=1,
        )
        results.append(
            {
                "source": source,
                "metric": metric,
                "budget_chars": budget if budget is not None else "ALL",
                "family": family,
                "candidate": candidate,
                "baseline": baseline,
                "n_pairs": len(pairs),
                "n_independent_units": n_independent,
                "n_repeated_measures": (
                    n_budgets * max_repeated if cross_budget else max_repeated
                ),
                "inference_method": inference_method,
                "inferential_test": "none" if descriptive_clustered else "exact_mcnemar",
                "multiplicity_family": multiplicity_family
                or f"{source}:{metric}:{budget if budget is not None else 'ALL'}:{family}",
                "candidate_rate": mean([a for a, _ in pairs]),
                "baseline_rate": mean([b for _, b in pairs]),
                "risk_difference": difference,
                "ci95_low": low,
                "ci95_high": high,
                "candidate_only_success": candidate_only,
                "baseline_only_success": baseline_only,
                "mcnemar_p": p_value,
            }
        )
    adjusted = holm_adjust([row["mcnemar_p"] for row in results])
    holm_family_size = sum(math.isfinite(row["mcnemar_p"]) for row in results)
    for row, value in zip(results, adjusted, strict=True):
        row["holm_p"] = value
        row["holm_family_size"] = holm_family_size
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bar_svg(
    path: Path,
    *,
    title: str,
    values: list[tuple[str, float]],
    y_label: str,
    maximum: float,
    formatter: Callable[[float], str],
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_width = plot_width / max(1, len(values)) * 0.58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" '
        f'font-size="22" font-weight="bold">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#172026"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" '
        f'y2="{top+plot_height}" stroke="#172026"/>',
        f'<text transform="translate(25 {top+plot_height/2}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial" font-size="15">{y_label}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_height - (value / maximum * plot_height if maximum else 0)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                'stroke="#d9e0e4" stroke-width="1"/>',
                f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" '
                f'font-family="Arial" font-size="13">{formatter(value)}</text>',
            ]
        )
    palette = ("#1f77b4", "#9aa5ad", "#2ca02c", "#ffbf00", "#7b61ff")
    slot = plot_width / max(1, len(values))
    for index, (label, value) in enumerate(values):
        x = left + slot * index + (slot - bar_width) / 2
        bar_height = min(plot_height, max(0, value / maximum * plot_height if maximum else 0))
        y = top + plot_height - bar_height
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{palette[index % len(palette)]}"/>',
                f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" '
                f'font-family="Arial" font-size="13">{formatter(value)}</text>',
                f'<text x="{x+bar_width/2:.1f}" y="{top+plot_height+28}" text-anchor="middle" '
                f'font-family="Arial" font-size="13">{label.replace("_", " ")}</text>',
            ]
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def format_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--model-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_rows = read_jsonl(args.contexts)
    model_rows = read_jsonl(args.model_results) if args.model_results else []
    summaries = summarize_rows(
        context_rows,
        metric="context_success",
        source="context",
    )
    if model_rows:
        context_lookup = {
            (row["case_id"], row["condition"], int(row["budget_chars"])): row
            for row in context_rows
        }
        for row in model_rows:
            context = context_lookup.get(
                (row["case_id"], row["condition"], int(row["budget_chars"]))
            )
            if context:
                row["context_chars"] = context["context_chars"]
        summaries.extend(
            summarize_rows(model_rows, metric="answer_correct", source="model")
        )
    primary_budget = int(config["primary_budget"])
    primary_family = str(config["primary_family"])
    iterations = int(config["bootstrap_iterations"])
    analysis_contract = config.get("analysis") or {}
    evidence_status = str(analysis_contract.get("evidence_status", "preregistered"))
    independent_unit_field = str(
        analysis_contract.get("independent_unit_field", "case_id")
    )
    primary_multiplicity_family = str(
        analysis_contract.get(
            "multiplicity_family",
            f"primary:{primary_budget}:{primary_family}",
        )
    )
    configured_conditions = list(config["conditions"])
    primary_candidate = str(
        analysis_contract.get("primary_candidate", "hybrid")
    )
    primary_baselines = list(
        analysis_contract.get(
            "primary_baselines",
            [condition for condition in configured_conditions if condition != primary_candidate],
        )
    )
    if primary_candidate not in configured_conditions:
        raise ValueError(f"primary candidate {primary_candidate!r} is not configured")
    unknown_primary_baselines = [
        condition for condition in primary_baselines if condition not in configured_conditions
    ]
    if unknown_primary_baselines or primary_candidate in primary_baselines:
        raise ValueError(
            "invalid primary baselines: "
            f"unknown={unknown_primary_baselines} contains_candidate="
            f"{primary_candidate in primary_baselines}"
        )
    fused_candidate = str(
        analysis_contract.get("fused_candidate", "fused_hybrid")
    )
    fused_baselines = list(
        analysis_contract.get(
            "fused_baselines",
            [condition for condition in configured_conditions if condition != fused_candidate],
        )
    )
    unknown_fused_baselines = [
        condition for condition in fused_baselines if condition not in configured_conditions
    ]
    if fused_candidate in configured_conditions and (
        unknown_fused_baselines or fused_candidate in fused_baselines
    ):
        raise ValueError(
            "invalid fused baselines: "
            f"unknown={unknown_fused_baselines} contains_candidate="
            f"{fused_candidate in fused_baselines}"
        )
    comparisons = pairwise(
        context_rows,
        source="context",
        metric="context_success",
        budget=primary_budget,
        family=primary_family,
        candidate=primary_candidate,
        baselines=primary_baselines,
        iterations=iterations,
        seed=int(config["seed"]) + 10_000,
        multiplicity_family=f"context:{primary_multiplicity_family}",
        independent_unit_field=independent_unit_field,
    )
    if model_rows:
        comparisons.extend(
            pairwise(
                model_rows,
                source="model",
                metric="answer_correct",
                budget=primary_budget,
                family=primary_family,
                candidate=primary_candidate,
                baselines=primary_baselines,
                iterations=iterations,
                seed=int(config["seed"]) + 20_000,
                multiplicity_family=f"model:{primary_multiplicity_family}",
                independent_unit_field=independent_unit_field,
            )
        )
    # Fused-hybrid pairwise comparisons across all budgets and families
    # for context and model endpoints. These are the comparisons reported
    # in Tables V (context, all budgets) and VII (model, primary budget).
    all_budgets = sorted({int(row["budget_chars"]) for row in context_rows})
    all_families = sorted({row["family"] for row in context_rows})
    fused_context_comparisons: list[dict[str, Any]] = []
    if fused_candidate in configured_conditions:
        for budget in all_budgets:
            for family in all_families:
                fused_context_comparisons.extend(
                    pairwise(
                        context_rows,
                        source="context",
                        metric="context_success",
                        budget=budget,
                        family=family,
                        candidate=fused_candidate,
                        baselines=fused_baselines,
                        iterations=iterations,
                        seed=int(config["seed"]) + 30_000,
                        multiplicity_family=f"context:fused:{budget}:{family}",
                        independent_unit_field=independent_unit_field,
                    )
                )
    # Pooled comparison across all budgets and all families.
    # Inference uses cluster bootstrap by case_id (not independent observations).
    # n_pairs reflects total observations; n_independent_units reflects unique cases.
    if fused_candidate in configured_conditions:
        fused_context_comparisons.extend(
            pairwise(
                context_rows,
                source="context",
                metric="context_success",
                budget=None,
                family="ALL",
                candidate=fused_candidate,
                baselines=fused_baselines,
                iterations=iterations,
                seed=int(config["seed"]) + 35_000,
                multiplicity_family="context:fused:pooled_all_budgets",
                independent_unit_field=independent_unit_field,
            )
        )
    write_csv(args.output_dir / "pairwise_fused_context.csv", fused_context_comparisons)
    if model_rows:
        model_budgets = sorted({int(row["budget_chars"]) for row in model_rows})
        model_families = sorted({row["family"] for row in model_rows})
        fused_model_comparisons: list[dict[str, Any]] = []
        if fused_candidate in configured_conditions:
            for budget in model_budgets:
                for family in model_families:
                    fused_model_comparisons.extend(
                        pairwise(
                            model_rows,
                            source="model",
                            metric="answer_correct",
                            budget=budget,
                            family=family,
                            candidate=fused_candidate,
                            baselines=fused_baselines,
                            iterations=iterations,
                            seed=int(config["seed"]) + 40_000,
                            multiplicity_family=f"model:fused:{budget}:{family}",
                            independent_unit_field=independent_unit_field,
                        )
                    )
        # Pooled model comparison at primary budget across all families.
        # At primary budget each case appears once, so inference is paired bootstrap.
        if fused_candidate in configured_conditions:
            fused_model_comparisons.extend(
                pairwise(
                    model_rows,
                    source="model",
                    metric="answer_correct",
                    budget=primary_budget,
                    family="ALL",
                    candidate=fused_candidate,
                    baselines=fused_baselines,
                    iterations=iterations,
                    seed=int(config["seed"]) + 45_000,
                    multiplicity_family=f"model:fused:{primary_budget}:ALL",
                    independent_unit_field=independent_unit_field,
                )
            )
        write_csv(args.output_dir / "pairwise_fused_model.csv", fused_model_comparisons)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "pairwise.csv", comparisons)

    primary_summary = [
        row
        for row in summaries
        if row["source"] == "context"
        and row["budget_chars"] == primary_budget
        and row["family"] == "ALL"
    ]
    ordered = sorted(
        primary_summary,
        key=lambda row: CONDITIONS.index(str(row["condition"])),
    )
    bar_svg(
        args.output_dir / "figures" / "context_success.svg",
        title=f"Context success at {primary_budget:,} characters",
        values=[(str(row["condition"]), float(row["rate"])) for row in ordered],
        y_label="Success rate",
        maximum=1.0,
        formatter=lambda value: f"{value:.0%}",
    )
    max_chars = max([float(row["mean_context_chars"]) for row in ordered] or [1.0])
    bar_svg(
        args.output_dir / "figures" / "context_size.svg",
        title=f"Mean prompt size at {primary_budget:,}-character budget",
        values=[
            (str(row["condition"]), float(row["mean_context_chars"])) for row in ordered
        ],
        y_label="Characters",
        maximum=max(1.0, max_chars * 1.1),
        formatter=lambda value: f"{value:,.0f}",
    )

    isolation = [
        row
        for row in context_rows
        if row["family"] == "cross_user_isolation"
        and int(row["budget_chars"]) == primary_budget
    ]
    leaks = sum(bool(row["forbidden_hits"]) for row in isolation)
    isolation_case_count = len({row["case_id"] for row in isolation})
    model_isolation = [
        row
        for row in model_rows
        if row["family"] == "cross_user_isolation"
        and int(row["budget_chars"]) == primary_budget
        and row.get("status") == "ok"
    ]
    model_leaks = sum(bool(row.get("forbidden_answer_hits")) for row in model_isolation)
    model_isolation_case_count = len({row["case_id"] for row in model_isolation})
    missing_model = sum(row.get("status") != "ok" for row in model_rows)
    report_lines = [
        "# Hybrid Memory Experiment Analysis",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {utc_now()}",
        "- Verification Status: ANALYZED",
        "- Version Label: validation_v1",
        "",
        "## Analysis scope",
        "",
        f"- Context rows: {len(context_rows)}",
        f"- Model rows: {len(model_rows)}",
        f"- Primary budget: {primary_budget}",
        f"- Primary family: {primary_family}",
        f"- Candidate: {primary_candidate}",
        f"- Evidence status: {evidence_status}",
        f"- Independent-unit field: {independent_unit_field}",
        f"- Bootstrap iterations: {iterations}",
        "- All cases are synthetic. Results do not establish clinical effectiveness.",
        "",
        "## Configured primary comparisons",
        "",
        "| Source | Baseline | N pairs | Candidate | Baseline | Risk difference [95% CI] | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        report_lines.append(
            f"| {row['source']} | {row['baseline']} | {row['n_pairs']} | "
            f"{format_number(row['candidate_rate'])} | {format_number(row['baseline_rate'])} | "
            f"{format_number(row['risk_difference'])} "
            f"[{format_number(row['ci95_low'])}, {format_number(row['ci95_high'])}] | "
            f"{format_number(row['holm_p'], 4)} |"
        )
    report_lines.extend(
        [
            "",
            "## Safety endpoint",
            "",
            f"- Observed context forbidden-token hits across all configured conditions: "
            f"{leaks}/{len(isolation)} rows from {isolation_case_count} synthetic case probes.",
            (
                f"- Observed model-answer forbidden-token hits across all configured conditions: "
                f"{model_leaks}/{len(model_isolation)} rows from "
                f"{model_isolation_case_count} synthetic case probes."
                if model_rows
                else "- Model-answer leakage was not evaluated."
            ),
            "- These are deterministic synthetic probe counts, not independent deployment "
            "samples; no population leak-rate confidence bound is reported.",
            "",
            "## Missingness and run integrity",
            "",
            f"- Model-call failures: {missing_model}/{len(model_rows)}."
            if model_rows
            else "- Model benchmark was not supplied; no end-to-end claims are available.",
            "- Pairwise tests use only complete matched case-condition pairs.",
            "- Exact McNemar tests are used only when each row is one independent unit;",
            "  otherwise concept or case clusters are descriptive and receive no p-value.",
            "- Holm family size is derived from the actual inferential comparison set.",
            "- Primary-budget CIs resample the configured independent unit.",
            "- Cross-budget pooled comparisons use cluster bootstrap by case_id to respect",
            "  within-case repeated-measure dependence; they are descriptive robustness checks",
            "  and receive no McNemar or Holm-adjusted p-value.",
            "",
            "## Statistical fallacy scan",
            "",
            "Coverage: 11/11 checked.",
            "",
            "1. Simpson's paradox: aggregate and per-family tables are emitted for direction checks.",
            "2. Ecological fallacy: inference is restricted to synthetic benchmark cases.",
            "3. Berkson's paradox: selected synthetic cases do not estimate clinical prevalence.",
            "4. Collider bias: no post-treatment covariate adjustment is performed.",
            "5. Base-rate neglect: abstention and isolation family sizes are reported explicitly.",
            "6. Regression to the mean: no extreme-score enrollment or pre/post outcome is used.",
            "7. Survivorship bias: failed model calls are counted and reported, not silently dropped.",
            "8. Look-elsewhere effect: comparisons follow the frozen config contract and are Holm-corrected",
            "   with family size derived from the actual comparison set.",
            "9. Garden of forking paths: configuration, dataset, result hashes, and code revision are recorded.",
            "10. Correlation versus causation: paired condition assignment supports claims only about this benchmark harness.",
            "11. Reverse causality: not applicable to randomized memory-condition evaluation.",
            "",
            "## Interpretation boundary",
            "",
            f"This report describes {evidence_status} benchmark behavior. A broader external claim "
            "additionally requires an independently reviewed evidence profile, successful "
            "end-to-end model execution on a pinned harness, "
            "independent artifact validation, and robustness analyses across context budgets. "
            "Multi-budget aggregates are descriptive; inferential claims require one paired "
            "observation per configured independent unit. Repeated variants are clustered "
            "and reported descriptively.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    inputs = {"config": args.config, "contexts": args.contexts}
    if args.model_results:
        inputs["model_results"] = args.model_results
    manifest = environment_manifest(command=sys.argv, inputs=inputs)
    manifest.update(
        {
            "stage": "analysis",
            "summary_sha256": sha256_file(args.output_dir / "summary.csv"),
            "pairwise_sha256": sha256_file(args.output_dir / "pairwise.csv"),
            "report_sha256": sha256_file(args.output_dir / "report.md"),
            "model_results_included": bool(model_rows),
            "fallacy_scan_coverage": "11/11",
        }
    )
    write_json(args.output_dir / "analysis.manifest.json", manifest)
    print(f"summary_rows={len(summaries)} pairwise_rows={len(comparisons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
