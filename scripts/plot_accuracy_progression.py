from pathlib import Path


# Same scoring standard for every point:
# binary claim-level accuracy with POSSIBLE_SUPPORT counted as supported.
# The x-axis intentionally hides run names per request; labels below are kept
# only for the legend/metadata and future auditability.
RUNS = [
    ("Sean rules", 81.77),
    ("Sean rules canonicalized", 83.52),
    ("Metadata preserved", 61.76),
    ("Ollama markdown stripped", 75.51),
    ("Ollama case-targeted GPU", 78.00),
    ("Live API normal-vs-edge audit", 61.90),
    ("Llama 3.2 false-negative analysis", 79.30),
    ("Live API mostly-normal 16 analytics", 78.10),
    ("Live API mostly-normal 16 rerun", 63.20),
    ("Live API mostly-normal 16 rerun analysis", 68.80),
    ("50-query max5", 89.80),
]

Y_MIN = 60
Y_MAX = 100


def main() -> None:
    output_path = Path("artifacts/test_reports/accuracy_progression_possible_supported.svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"Iteration {i}" for i in range(1, len(RUNS) + 1)]
    accuracies = [accuracy for _, accuracy in RUNS]

    width = 1200
    height = 620
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 135
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_pos(index: int) -> float:
        if len(accuracies) == 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * index / (len(accuracies) - 1))

    def y_pos(value: float) -> float:
        clamped = min(max(value, Y_MIN), Y_MAX)
        return margin_top + plot_height - (plot_height * (clamped - Y_MIN) / (Y_MAX - Y_MIN))

    points = [(x_pos(i), y_pos(value)) for i, value in enumerate(accuracies)]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">Accuracy Progression Across Iterations</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">Binary claim-level accuracy; POSSIBLE_SUPPORT counted as supported</text>',
    ]

    for tick in range(Y_MIN, Y_MAX + 1, 10):
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{tick}%</text>')

    parts.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333"/>',
            f'<text x="22" y="{margin_top + plot_height / 2}" transform="rotate(-90 22,{margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="14">Accuracy (%)</text>',
            f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="3"/>',
        ]
    )

    for index, ((x, y), label, accuracy) in enumerate(zip(points, labels, accuracies)):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#111">{accuracy:.1f}%</text>')
        parts.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" transform="rotate(45 {x:.1f},{height - margin_bottom + 24})" text-anchor="start" font-family="Arial" font-size="12" fill="#333">{label}</text>')

    parts.append(f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">Final point: 50-query max5, POSSIBLE_SUPPORT counted as supported</text>')
    parts.append("</svg>")

    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
