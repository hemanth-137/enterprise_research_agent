
import json
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "evals_open_ragbench_with_queries.json"
OUTPUT_FILE = "rag_eval_report.md"

BENCHMARK_SIZE = 3000


# ============================================================
# LOAD DATA
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Extract evaluator scores
df["correctness"] = df["evals"].apply(
    lambda x: x.get("judge_correctness")
)

df["faithfulness"] = df["evals"].apply(
    lambda x: x.get("faithfulness")
)

df["context_sufficiency"] = df["evals"].apply(
    lambda x: x.get("context_sufficiency")
)

n = len(df)


# ============================================================
# HELPERS
# ============================================================

def pct(count, total):
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def result(count, total):
    """
    Example:
    **78.0% (78/100)**
    """
    return f"**{pct(count, total)} ({count}/{total})**"


def metric_row(metric, value):
    return f"| {metric:<35} | {value:>15} |"


# ============================================================
# OVERALL METRICS
# ============================================================

fully_correct = (df["correctness"] == 1.0).sum()

correct_or_partial = (
    df["correctness"].isin([0.5, 1.0])
).sum()

fully_faithful = (
    df["faithfulness"] == 1.0
).sum()

sufficient_context = (
    df["context_sufficiency"] == 1.0
).sum()

avg_correctness = df["correctness"].mean()
avg_faithfulness = df["faithfulness"].mean()
avg_context = df["context_sufficiency"].mean()


# ============================================================
# FAILURE ANALYSIS
# ============================================================

correct_faithful_sufficient = (
    (df["correctness"] == 1.0)
    & (df["faithfulness"] == 1.0)
    & (df["context_sufficiency"] == 1.0)
).sum()

correct_but_insufficient = (
    (df["correctness"] == 1.0)
    & (df["context_sufficiency"] == 0.0)
).sum()

incorrect_despite_sufficient = (
    (df["correctness"] < 1.0)
    & (df["context_sufficiency"] == 1.0)
).sum()

insufficient_context = (
    df["context_sufficiency"] == 0.0
).sum()

unfaithful = (
    df["faithfulness"] < 1.0
).sum()


# ============================================================
# BUILD MARKDOWN
# ============================================================

report = []

report.append("# RAG Evaluation Report")
report.append("")

report.append(
    f"**Benchmark:** Open-RAGBench  \n"
    f"**Evaluation sample:** {n}/{BENCHMARK_SIZE} queries "
    f"({n / BENCHMARK_SIZE * 100:.2f}% of benchmark)"
)

report.append("")
report.append(
    "> Results are based on the evaluated sample and should not "
    "be interpreted as full-benchmark performance."
)


# ============================================================
# KEY RESULTS
# ============================================================

report.append("")
report.append("### RAG Quality Results")
report.append("")

report.append("| Metric | Result |")
report.append("| -------- | -----: |")

report.append(
    f"| Fully Correct | "
    f"{result(fully_correct, n)} |"
)

report.append(
    f"| Correct or Partial | "
    f"{result(correct_or_partial, n)} |"
)

report.append(
    f"| Fully Faithful | "
    f"{result(fully_faithful, n)} |"
)

report.append(
    f"| Sufficient Context | "
    f"{result(sufficient_context, n)} |"
)

report.append(
    f"| Average Correctness | "
    f"**{avg_correctness:.3f}** |"
)

report.append(
    f"| Average Faithfulness | "
    f"**{avg_faithfulness:.3f}** |"
)

report.append(
    f"| Average Context Sufficiency | "
    f"**{avg_context:.3f}** |"
)


# ============================================================
# FAILURE ANALYSIS
# ============================================================

report.append("")
report.append("### Failure Analysis")
report.append("")

report.append("| Outcome | Queries | Rate |")
report.append("| -------- | -----: | ----: |")

report.append(
    f"| Correct + Faithful + Sufficient Context | "
    f"{correct_faithful_sufficient} | "
    f"**{pct(correct_faithful_sufficient, n)}** |"
)

report.append(
    f"| Correct but Insufficient Context | "
    f"{correct_but_insufficient} | "
    f"**{pct(correct_but_insufficient, n)}** |"
)

report.append(
    f"| Incorrect despite Sufficient Context | "
    f"{incorrect_despite_sufficient} | "
    f"**{pct(incorrect_despite_sufficient, n)}** |"
)

report.append(
    f"| Insufficient Context | "
    f"{insufficient_context} | "
    f"**{pct(insufficient_context, n)}** |"
)

report.append(
    f"| Unfaithful | "
    f"{unfaithful} | "
    f"**{pct(unfaithful, n)}** |"
)


# ============================================================
# BY QUERY TYPE
# ============================================================

if "type" in df.columns:

    report.append("")
    report.append("### Performance by Query Type")
    report.append("")

    report.append(
        "| Metric | " +
        " | ".join(
            str(x) for x in df["type"].dropna().unique()
        ) +
        " |"
    )

    types = list(df["type"].dropna().unique())

    report.append(
        "| -------- | " +
        " | ".join("---:" for _ in types) +
        " |"
    )

    # Queries
    row = ["Queries"]
    for query_type in types:
        count = (df["type"] == query_type).sum()
        row.append(str(count))

    report.append("| " + " | ".join(row) + " |")

    # Fully correct
    row = ["Fully Correct"]
    for query_type in types:
        group = df[df["type"] == query_type]
        count = (group["correctness"] == 1.0).sum()
        row.append(pct(count, len(group)))

    report.append(
        "| " + " | ".join(row) + " |"
    )

    # Faithful
    row = ["Fully Faithful"]
    for query_type in types:
        group = df[df["type"] == query_type]
        count = (group["faithfulness"] == 1.0).sum()
        row.append(pct(count, len(group)))

    report.append(
        "| " + " | ".join(row) + " |"
    )

    # Context
    row = ["Sufficient Context"]
    for query_type in types:
        group = df[df["type"] == query_type]
        count = (group["context_sufficiency"] == 1.0).sum()
        row.append(pct(count, len(group)))

    report.append(
        "| " + " | ".join(row) + " |"
    )


# ============================================================
# BY SOURCE
# ============================================================

if "source" in df.columns:

    report.append("")
    report.append("### Performance by Source")
    report.append("")

    report.append(
        "| Source | Queries | Fully Correct | Faithful | "
        "Sufficient Context |"
    )

    report.append(
        "| -------- | -----: | ------------: | -------: | "
        "-----------------: |"
    )

    # Sort sources by correctness, worst first
    source_stats = []

    for source, group in df.groupby("source", dropna=False):

        correct = (
            group["correctness"] == 1.0
        ).sum()

        faithful = (
            group["faithfulness"] == 1.0
        ).sum()

        context = (
            group["context_sufficiency"] == 1.0
        ).sum()

        source_stats.append({
            "source": str(source),
            "n": len(group),
            "correct": correct,
            "faithful": faithful,
            "context": context,
        })

    source_stats.sort(
        key=lambda x: x["correct"] / x["n"]
        if x["n"] else 0
    )

    for s in source_stats:

        report.append(
            f"| {s['source']} | "
            f"{s['n']} | "
            f"{pct(s['correct'], s['n'])} | "
            f"{pct(s['faithful'], s['n'])} | "
            f"{pct(s['context'], s['n'])} |"
        )


# ============================================================
# WRITE REPORT
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"Report generated: {OUTPUT_FILE}")