from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "ui_redesign"
SIZE = (1280, 720)
INK = "#111827"
MUTED = "#687386"
LINE = "#dfe6f1"
PANEL = "#ffffff"
BG = "#f5f7fb"
PURPLE = "#6d4aff"
BLUE = "#377dff"
GREEN = "#12b76a"
ORANGE = "#f59e0b"
RED = "#e5484d"
CYAN = "#0ea5e9"
COLORS = [PURPLE, BLUE, GREEN, ORANGE, RED, CYAN]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "Malgun Gothic Bold" if bold else "Malgun Gothic",
        "Segoe UI Bold" if bold else "Segoe UI",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_10 = font(10)
FONT_11 = font(11)
FONT_12 = font(12)
FONT_13 = font(13)
FONT_14 = font(14)
FONT_16 = font(16, bold=True)
FONT_18 = font(18, bold=True)
FONT_22 = font(22, bold=True)
FONT_28 = font(28, bold=True)


def rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str | None = None,
    radius: int = 16,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    fill: str = INK,
    font_obj: ImageFont.ImageFont = FONT_13,
) -> None:
    draw.text(xy, value, fill=fill, font=font_obj)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str) -> None:
    text(draw, xy, value.upper(), fill=MUTED, font_obj=FONT_10)


def line_label(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, caption: str = "") -> None:
    draw.rounded_rectangle((x, y + 2, x + 4, y + 38), radius=3, fill=PURPLE)
    text(draw, (x + 12, y), title, font_obj=FONT_16)
    if caption:
        text(draw, (x + 12, y + 23), caption, fill=MUTED, font_obj=FONT_11)


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    value: str,
    hint: str,
    color: str,
) -> None:
    rect(draw, box, fill=PANEL, outline=LINE, radius=15)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1, y1, x2, y1 + 5), radius=15, fill=color)
    label(draw, (x1 + 15, y1 + 17), title)
    text(draw, (x1 + 15, y1 + 43), value, font_obj=FONT_22)
    text(draw, (x1 + 15, y1 + 72), hint, fill=MUTED, font_obj=FONT_11)


def small_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    rect(draw, box, fill=PANEL, outline=LINE, radius=18)
    text(draw, (box[0] + 22, box[1] + 20), title, font_obj=FONT_18)


def base(active: str, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)

    rect(draw, (28, 24, 258, 696), fill="#faf9ff", outline="#ded8ff", radius=22)
    rect(draw, (46, 42, 240, 92), fill="#ffffff", outline="#ebe7ff", radius=15)
    rect(draw, (62, 56, 90, 82), fill=PURPLE, radius=8)
    text(draw, (102, 51), "MARS Console", font_obj=FONT_16)
    text(draw, (102, 72), "Multimodal AI", fill=MUTED, font_obj=FONT_10)

    groups = [
        (
            "Dashboard",
            [
                ("control-room", "Control Room"),
                ("search", "Search"),
                ("recommendation", "Recommendation"),
            ],
        ),
        (
            "Operations",
            [
                ("experiments", "Experiments"),
                ("model-ops", "Model Ops"),
                ("live-logs", "Live Logs"),
            ],
        ),
        ("Submit", [("qa-gate", "QA Gate"), ("guide", "Guide")]),
    ]
    y = 118
    for group, items in groups:
        text(draw, (54, y), group.upper(), fill="#9188a8", font_obj=FONT_10)
        y += 17
        for slug, name in items:
            active_item = slug == active
            fill = PURPLE if active_item else "#ffffff"
            outline = PURPLE if active_item else "#ebe7ff"
            rect(draw, (46, y, 240, y + 38), fill=fill, outline=outline, radius=11)
            dot = "#ffffff" if active_item else COLORS[len(name) % len(COLORS)]
            draw.ellipse((62, y + 13, 74, y + 25), fill=dot)
            text(
                draw,
                (88, y + 10),
                name,
                fill="#ffffff" if active_item else "#303246",
                font_obj=FONT_12,
            )
            y += 44
        y += 9

    rect(draw, (46, 590, 240, 644), fill="#ffffff", outline="#ded8ff", radius=15)
    label(draw, (62, 604), "API Connection")
    text(draw, (62, 624), "Live API connected", fill="#087443", font_obj=FONT_12)

    rect(draw, (285, 24, 1228, 92), fill="#ffffff", outline=LINE, radius=18)
    label(draw, (315, 43), subtitle)
    text(draw, (315, 61), title, font_obj=FONT_22)
    for index, badge in enumerate(["Live API", "full", "v0019"]):
        x = 1035 + index * 62
        rect(draw, (x, 46, x + 54, 68), fill="#f8fafc", outline=LINE, radius=11)
        text(draw, (x + 10, 51), badge, fill=GREEN if index == 0 else MUTED, font_obj=FONT_10)
    return image, draw


def bar_chart(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], values: list[int], labels: list[str]
) -> None:
    x1, y1, x2, y2 = box
    draw.line((x1, y2, x2, y2), fill="#d9e2ef", width=2)
    max_value = max(values) or 1
    step = (x2 - x1) / max(len(values), 1)
    for index, value in enumerate(values):
        height = int((y2 - y1 - 18) * value / max_value)
        bx = int(x1 + index * step + step * 0.25)
        bw = max(22, int(step * 0.48))
        rect(draw, (bx, y2 - height, bx + bw, y2), fill=COLORS[index % len(COLORS)], radius=9)
        text(draw, (bx - 2, y2 + 8), labels[index], fill=MUTED, font_obj=FONT_10)


def donut(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], values: list[int]) -> None:
    total = sum(values) or 1
    start = -90
    for index, value in enumerate(values):
        end = start + (value / total) * 360
        draw.pieslice(box, start, end, fill=COLORS[index % len(COLORS)])
        start = end
    x1, y1, x2, y2 = box
    inset = int((x2 - x1) * 0.30)
    draw.ellipse((x1 + inset, y1 + inset, x2 - inset, y2 - inset), fill=PANEL)


def area_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    draw.line((x1, y2, x2, y2), fill="#d9e2ef", width=2)
    height = max(y2 - y1, 1)
    widths = [0.0, 0.18, 0.43, 0.67, 0.88, 1.0]
    levels = [0.30, 0.86, 0.83, 0.94, 0.68, 0.98]
    points = [
        (int(x1 + (x2 - x1) * width), int(y2 - height * level))
        for width, level in zip(widths, levels, strict=True)
    ]
    polygon = points + [(x2, y2), (x1, y2)]
    draw.polygon(polygon, fill="#98bbfb")
    draw.line(points, fill=BLUE, width=5)
    purchase = [(x, max(y1, y - int(height * 0.10))) for x, y in points]
    draw.line(purchase, fill=GREEN, width=4)


def product_grid(draw: ImageDraw.ImageDraw, x: int, y: int, count: int = 3) -> None:
    for index in range(count):
        left = x + index * 185
        rect(draw, (left, y, left + 165, y + 178), fill=PANEL, outline=LINE, radius=16)
        rect(
            draw,
            (left + 12, y + 12, left + 153, y + 96),
            fill="#eef3f8",
            outline="#e5edf7",
            radius=13,
        )
        draw.ellipse((left + 58, y + 28, left + 108, y + 78), fill=COLORS[index % len(COLORS)])
        rect(draw, (left + 18, y + 108, left + 88, y + 121), fill="#edf1f7", radius=6)
        text(draw, (left + 18, y + 130), f"P000{index + 1}", fill=MUTED, font_obj=FONT_11)
        text(draw, (left + 18, y + 150), f"Score 0.{87 - index * 6}", font_obj=FONT_12)


def timeline(draw: ImageDraw.ImageDraw, x: int, y: int, rows: list[tuple[str, str, str]]) -> None:
    for index, (event, title, meta) in enumerate(rows):
        top = y + index * 66
        color = [BLUE, CYAN, ORANGE, GREEN][index % 4]
        if index < len(rows) - 1:
            draw.line((x + 15, top + 33, x + 15, top + 72), fill="#dbe4f0", width=2)
        draw.ellipse((x, top, x + 30, top + 30), fill=color)
        rect(draw, (x + 42, top - 4, x + 365, top + 52), fill=PANEL, outline="#e5ebf4", radius=13)
        draw.rounded_rectangle((x + 42, top - 4, x + 47, top + 52), radius=4, fill=color)
        text(draw, (x + 60, top + 3), event, fill=color, font_obj=FONT_12)
        text(draw, (x + 60, top + 20), title, font_obj=FONT_13)
        text(draw, (x + 60, top + 38), meta, fill=MUTED, font_obj=FONT_10)


def control_room() -> Image.Image:
    image, draw = base("control-room", "Control Room", "operations snapshot")
    line_label(draw, 300, 118, "Operations Snapshot", "Quality, lift, and training state")
    metrics = [
        ("Events", "1,000,000", "behavior logs", PURPLE),
        ("Search NDCG", "0.676", "target pass", BLUE),
        ("Reco HitRate", "0.463", "target pass", GREEN),
        ("CVR Lift", "-0.02%", "watch", RED),
    ]
    for index, item in enumerate(metrics):
        card(draw, (300 + index * 230, 172, 510 + index * 230, 272), *item)
    line_label(draw, 300, 306, "Live Status & Readiness", "Event flow and service readiness")
    small_panel(draw, (300, 360, 855, 650), "Live Event Flow")
    area_chart(draw, (330, 430, 810, 600))
    small_panel(draw, (880, 360, 1220, 650), "Service Readiness")
    for index, name in enumerate(
        ["Processed Data", "Search Index", "Recommendation Model", "Redis Feature Store", "Reports"]
    ):
        y = 410 + index * 43
        text(draw, (910, y), name, fill=MUTED, font_obj=FONT_12)
        rect(draw, (1130, y - 5, 1190, y + 20), fill="#ecfdf3", outline="#b7ebcd", radius=12)
        text(draw, (1146, y), "Ready", fill="#087443", font_obj=FONT_10)
    return image


def search_quality() -> Image.Image:
    image, draw = base("search", "Search Quality", "retrieval quality and feedback")
    line_label(draw, 300, 118, "Offline Quality Summary", "MRR/NDCG/Recall and latency targets")
    metrics = [
        ("MRR@10", "0.615", "target >= 0.55", PURPLE),
        ("NDCG@10", "0.676", "target >= 0.50", GREEN),
        ("Recall@10", "0.866", "held-out qrels", BLUE),
        ("p95", "76 ms", "target <= 200 ms", ORANGE),
    ]
    for index, item in enumerate(metrics):
        card(draw, (300 + index * 230, 172, 510 + index * 230, 272), *item)
    line_label(draw, 300, 306, "Search Request", "text / image / hybrid input in one area")
    rect(draw, (300, 356, 1220, 420), fill=PANEL, outline=LINE, radius=18)
    text(draw, (330, 378), "Search type: text", fill=MUTED, font_obj=FONT_12)
    text(draw, (540, 378), "Query: black socks", font_obj=FONT_16)
    text(draw, (820, 378), "Top K: 10", fill=MUTED, font_obj=FONT_12)
    line_label(draw, 300, 448, "Search Results", "ranked products with image, score, and feedback")
    product_grid(draw, 300, 500, 4)
    rect(draw, (1045, 512, 1210, 620), fill="#f8fafc", outline=LINE, radius=15)
    text(draw, (1064, 532), "Feedback", font_obj=FONT_16)
    for idx, event in enumerate(["View", "Cart", "Purchase"]):
        rect(draw, (1064, 562 + idx * 25, 1165, 582 + idx * 25), fill=COLORS[idx + 1], radius=10)
        text(draw, (1082, 565 + idx * 25), event, fill="#ffffff", font_obj=FONT_10)
    return image


def recommendation() -> Image.Image:
    image, draw = base("recommendation", "Recommendation", "personalized ranking")
    line_label(draw, 300, 118, "Offline Quality Summary", "ranking quality and serving latency")
    metrics = [
        ("Recall@300", "0.635", "target pass", PURPLE),
        ("AUC", "0.859", "target pass", BLUE),
        ("HitRate@50", "0.463", "target pass", GREEN),
        ("p95 Total", "35 ms", "live budget", ORANGE),
    ]
    for index, item in enumerate(metrics):
        card(draw, (300 + index * 230, 172, 510 + index * 230, 272), *item)
    line_label(
        draw, 300, 306, "Recommendation Evidence", "recent behavior timeline plus session context"
    )
    small_panel(draw, (300, 360, 720, 650), "Recent User Behavior")
    timeline(
        draw,
        330,
        410,
        [
            ("View", "P0012 / Slim denim", "blue / jeans / 00:43"),
            ("Cart", "P0148 / Black socks", "basic / socks / 00:42"),
            ("Purchase", "P0215 / Red trousers", "red / trousers / 00:41"),
        ],
    )
    small_panel(draw, (750, 360, 1220, 650), "Session Context")
    for idx, (name, value) in enumerate(
        [
            ("Known user", "99 history events"),
            ("Recent response", "view 4 / cart 1"),
            ("Serving strategy", "A/B basic ranking"),
        ]
    ):
        rect(
            draw,
            (785 + idx * 135, 410, 900 + idx * 135, 492),
            fill="#f8fafc",
            outline="#e7ebf2",
            radius=14,
        )
        label(draw, (798 + idx * 135, 426), name)
        text(draw, (798 + idx * 135, 452), value, font_obj=FONT_12)
    product_grid(draw, 785, 520, 2)
    return image


def experiments() -> Image.Image:
    image, draw = base("experiments", "Experiment Center", "A/B analytics")
    line_label(draw, 300, 118, "Experiment Selection", "default A/B key and bucket assignment")
    rect(draw, (300, 168, 1220, 230), fill=PANEL, outline=LINE, radius=18)
    text(draw, (330, 188), "Experiment key: mars_default", font_obj=FONT_16)
    text(draw, (700, 188), "Control vs Treatment", fill=MUTED, font_obj=FONT_12)
    line_label(draw, 300, 258, "Experiment Effect Summary", "CTR/CVR lift and statistical test")
    metrics = [
        ("CTR Lift", "-0.11%", "click rate", BLUE),
        ("CVR Lift", "-0.02%", "conversion rate", RED),
        ("p-value", "0.4032", "statistical test", PURPLE),
    ]
    for index, item in enumerate(metrics):
        card(draw, (300 + index * 250, 312, 530 + index * 250, 412), *item)
    small_panel(draw, (300, 455, 760, 650), "Traffic Distribution")
    donut(draw, (445, 500, 605, 660), [50, 50])
    small_panel(draw, (790, 455, 1220, 650), "Funnel Snapshot")
    bar_chart(draw, (830, 520, 1175, 615), [120, 13, 1], ["imp", "view", "buy"])
    return image


def model_ops() -> Image.Image:
    image, draw = base("model-ops", "Model Ops", "pipeline and distributions")
    line_label(draw, 300, 118, "Serving Pipeline", "data to search, ranking, reranking, serving")
    stages = ["Data", "Search", "Candidate", "Ranking", "Rerank", "Serving"]
    for index, stage in enumerate(stages):
        x = 300 + index * 150
        rect(draw, (x, 172, x + 130, 260), fill=PANEL, outline=LINE, radius=15)
        draw.ellipse((x + 15, 190, x + 43, 218), fill=COLORS[index % len(COLORS)])
        text(draw, (x + 15, 228), stage, font_obj=FONT_12)
    line_label(
        draw, 300, 294, "Operations Status", "runtime data, retraining readiness, latency budget"
    )
    status_rows = [
        ("Runtime Data", [("Products", "50k"), ("Users", "10k"), ("Events", "1.0M")]),
        ("Retrain Readiness", [("New logs", "2,481"), ("Threshold", "10k"), ("Action", "Monitor")]),
        (
            "Latency Budget",
            [("Candidate p95", "12 ms"), ("Ranking p95", "18 ms"), ("Total p95", "35 ms")],
        ),
    ]
    for index, (title, rows) in enumerate(status_rows):
        small_panel(draw, (300 + index * 310, 345, 585 + index * 310, 495), title)
        for row_index, (name, value) in enumerate(rows):
            y = 390 + row_index * 30
            text(draw, (325 + index * 310, y), name, fill=MUTED, font_obj=FONT_11)
            text(draw, (500 + index * 310, y), value, font_obj=FONT_12)
    small_panel(draw, (300, 525, 735, 690), "Persona Distribution")
    donut(draw, (470, 558, 594, 682), [23, 19, 17, 16, 14, 11])
    small_panel(draw, (765, 525, 1220, 690), "Live Behavior Mix")
    donut(draw, (940, 558, 1064, 682), [65, 32, 2, 1])
    return image


def live_logs() -> Image.Image:
    image, draw = base("live-logs", "Live Logs", "behavior stream and continuous training")
    line_label(
        draw, 300, 118, "Continuous Training Summary", "log growth, CTR/CVR, hitrate, retrain state"
    )
    metrics = [
        ("Status", "Watching", "monitoring", BLUE),
        ("New logs", "2,481", "threshold 10k", GREEN),
        ("CTR", "10.9%", "live rate", PURPLE),
        ("Retrain", "Monitor", "condition unmet", ORANGE),
    ]
    for index, item in enumerate(metrics):
        card(draw, (300 + index * 230, 172, 510 + index * 230, 272), *item)
    line_label(
        draw, 300, 306, "Live Behavior Logs", "surface cards, event flow, and recent event timeline"
    )
    for index, title in enumerate(["Search", "Recommendation"]):
        rect(
            draw,
            (300 + index * 230, 360, 510 + index * 230, 467),
            fill=PANEL,
            outline=LINE,
            radius=16,
        )
        text(draw, (320 + index * 230, 382), title, fill=COLORS[index], font_obj=FONT_16)
        text(draw, (320 + index * 230, 412), ["139,644", "116,227"][index], font_obj=FONT_22)
        text(draw, (320 + index * 230, 442), "view / cart / purchase", fill=MUTED, font_obj=FONT_10)
    small_panel(draw, (300, 500, 740, 685), "Minute Event Flow")
    area_chart(draw, (330, 555, 708, 628))
    for index, (name, color) in enumerate(
        [("Search", BLUE), ("View", CYAN), ("Cart", ORANGE), ("Purchase", GREEN)]
    ):
        x = 335 + index * 92
        draw.line((x, 654, x + 32, 654), fill=color, width=4)
        text(draw, (x + 38, 647), name, fill=MUTED, font_obj=FONT_10)
    small_panel(draw, (765, 360, 1220, 685), "Recent Live Events")
    timeline(
        draw,
        790,
        420,
        [
            ("Search", "black socks", "query event"),
            ("View", "P0148", "search result"),
            ("Cart", "P0148", "user action"),
        ],
    )
    return image


def qa_gate() -> Image.Image:
    image, draw = base("qa-gate", "Requirement Check", "submission verification")
    line_label(draw, 300, 118, "Submission Gate", "required runtime, quality, and data checks")
    for index, (label_name, value, color) in enumerate(
        [("PASS", "31", GREEN), ("WARN", "2", ORANGE), ("FAIL", "0", RED)]
    ):
        rect(
            draw,
            (300 + index * 300, 172, 560 + index * 300, 280),
            fill=PANEL,
            outline=LINE,
            radius=18,
        )
        draw.rounded_rectangle(
            (300 + index * 300, 172, 560 + index * 300, 178), radius=18, fill=color
        )
        label(draw, (325 + index * 300, 198), label_name)
        text(draw, (325 + index * 300, 225), value, font_obj=FONT_28)
    rect(draw, (300, 330, 1220, 650), fill=PANEL, outline=LINE, radius=20)
    headers = ["Runtime", "Search Quality", "Recommendation", "A/B Testing", "Continuous Training"]
    for index, row in enumerate(headers):
        y = 372 + index * 52
        text(draw, (335, y), row, font_obj=FONT_16)
        text(draw, (560, y + 2), "current value meets target", fill=MUTED, font_obj=FONT_12)
        rect(draw, (1060, y - 4, 1140, y + 24), fill="#ecfdf3", outline="#b7ebcd", radius=14)
        text(draw, (1083, y + 1), "PASS", fill="#087443", font_obj=FONT_11)
        draw.line((325, y + 38, 1185, y + 38), fill="#edf0f5", width=1)
    return image


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = {
        "control-room.png": control_room,
        "search.png": search_quality,
        "recommendation.png": recommendation,
        "experiments.png": experiments,
        "model-ops.png": model_ops,
        "live-logs.png": live_logs,
        "qa-gate.png": qa_gate,
    }
    for name, factory in pages.items():
        path = OUT_DIR / name
        factory().save(path, optimize=True)
        print(path)


if __name__ == "__main__":
    main()
