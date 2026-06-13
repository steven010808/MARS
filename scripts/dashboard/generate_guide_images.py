from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "ui_redesign"
SIZE = (1280, 720)
PALETTE = ["#6d4aff", "#377dff", "#12b76a", "#f59e0b", "#e5484d", "#0ea5e9"]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_12 = font(12)
FONT_14 = font(14)
FONT_16 = font(16)
FONT_18 = font(18, bold=True)
FONT_24 = font(24, bold=True)
FONT_34 = font(34, bold=True)


def rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str | None = None,
    radius: int = 18,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def base(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SIZE, "#f7f8fc")
    draw = ImageDraw.Draw(image)
    rect(draw, (30, 30, 245, 690), fill="#faf9ff", outline="#e2ddff", radius=24)
    draw.text((58, 58), "MARS", fill="#171229", font=FONT_24)
    draw.text((58, 92), "dashboard", fill="#7a728f", font=FONT_14)
    items = [
        ("Control", "#6d4aff"),
        ("Search", "#377dff"),
        ("Recommendation", "#12b76a"),
        ("Experiments", "#f59e0b"),
        ("Model Ops", "#111827"),
        ("Live Logs", "#0ea5e9"),
        ("QA Gate", "#e5484d"),
    ]
    y = 142
    for label, color in items:
        fill = "#ffffff"
        if label.lower().replace(" ", "-") in title.lower().replace(" ", "-"):
            fill = color
        rect(draw, (54, y, 220, y + 44), fill=fill, outline="#ebe7ff", radius=12)
        draw.ellipse((68, y + 14, 82, y + 28), fill=color if fill == "#ffffff" else "#ffffff")
        text_color = "#ffffff" if fill != "#ffffff" else "#303246"
        draw.text((94, y + 13), label, fill=text_color, font=FONT_14)
        y += 54
    draw.text((285, 46), subtitle.upper(), fill="#687386", font=FONT_12)
    draw.text((285, 68), title, fill="#111827", font=FONT_34)
    rect(draw, (1048, 52, 1218, 88), fill="#ffffff", outline="#e3e7ef", radius=18)
    draw.text((1070, 62), "Live API ready", fill="#087443", font=FONT_14)
    return image, draw


def card(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, value: str, color: str
) -> None:
    rect(draw, box, fill="#ffffff", outline="#e3e7ef", radius=16)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1, y1, x2, y1 + 6), radius=16, fill=color)
    draw.text((x1 + 18, y1 + 20), title.upper(), fill="#687386", font=FONT_12)
    draw.text((x1 + 18, y1 + 48), value, fill="#111827", font=FONT_24)


def donut(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], values: list[int]) -> None:
    total = sum(values)
    start = -90
    for index, value in enumerate(values):
        end = start + (value / total) * 360
        draw.pieslice(box, start, end, fill=PALETTE[index % len(PALETTE)])
        start = end
    inset = 52
    x1, y1, x2, y2 = box
    draw.ellipse((x1 + inset, y1 + inset, x2 - inset, y2 - inset), fill="#ffffff")


def bars(
    draw: ImageDraw.ImageDraw, origin: tuple[int, int], values: list[int], labels: list[str]
) -> None:
    x, y = origin
    max_value = max(values)
    for index, value in enumerate(values):
        height = int(210 * value / max_value)
        x1 = x + index * 78
        rect(
            draw,
            (x1, y + 230 - height, x1 + 44, y + 230),
            fill=PALETTE[index % len(PALETTE)],
            radius=10,
        )
        draw.text((x1 - 6, y + 242), labels[index], fill="#687386", font=FONT_12)


def product_cards(draw: ImageDraw.ImageDraw, y: int, *, color: str) -> None:
    for index in range(3):
        x = 285 + index * 300
        rect(draw, (x, y, x + 260, y + 210), fill="#ffffff", outline="#e3e7ef", radius=18)
        rect(draw, (x + 14, y + 14, x + 246, y + 118), fill=PALETTE[index], radius=14)
        draw.text((x + 18, y + 134), f"P0000{index + 1}", fill="#687386", font=FONT_12)
        draw.text((x + 18, y + 156), "Fashion item", fill="#111827", font=FONT_18)
        rect(draw, (x + 176, y + 174, x + 238, y + 196), fill=color, radius=12)


def control_room() -> Image.Image:
    image, draw = base("Control Room", "runtime overview")
    labels = ["Events", "Search", "Reco", "CVR"]
    values = ["1.0M", "0.676", "0.347", "+8.4%"]
    for idx, (label, value) in enumerate(zip(labels, values, strict=True)):
        card(draw, (285 + idx * 225, 130, 490 + idx * 225, 245), label, value, PALETTE[idx])
    rect(draw, (285, 290, 890, 610), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((315, 318), "Live Event Flow", fill="#111827", font=FONT_24)
    for idx, color in enumerate(PALETTE[:4]):
        points = [(330 + step * 70, 540 - ((step * 19 + idx * 31) % 170)) for step in range(8)]
        draw.line(points, fill=color, width=6, joint="curve")
    rect(draw, (920, 290, 1218, 610), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((950, 318), "Readiness", fill="#111827", font=FONT_24)
    for idx, label in enumerate(["data", "search", "recsys", "redis"]):
        draw.text((955, 372 + idx * 48), label, fill="#687386", font=FONT_16)
        rect(draw, (1110, 366 + idx * 48, 1180, 392 + idx * 48), fill="#ecfdf3", radius=13)
    return image


def search_quality() -> Image.Image:
    image, draw = base("Search Quality", "faiss retrieval")
    for idx, (label, value) in enumerate(
        [("MRR@10", "0.615"), ("NDCG@10", "0.676"), ("p95", "92 ms")]
    ):
        card(draw, (285 + idx * 230, 130, 500 + idx * 230, 245), label, value, PALETTE[idx])
    rect(draw, (285, 285, 1218, 350), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((318, 306), "black minimal jacket", fill="#111827", font=FONT_24)
    product_cards(draw, 390, color="#6d4aff")
    return image


def recommendation() -> Image.Image:
    image, draw = base("Recommendation", "personalized ranking")
    for idx, (label, value) in enumerate(
        [("Recall@300", "0.412"), ("AUC", "0.781"), ("Coverage", "0.294")]
    ):
        card(draw, (285 + idx * 230, 130, 500 + idx * 230, 245), label, value, PALETTE[idx + 1])
    rect(draw, (285, 290, 740, 610), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((315, 318), "Stage Latency", fill="#111827", font=FONT_24)
    bars(draw, (330, 350), [78, 42, 21], ["cand", "rank", "rerank"])
    product_cards(draw, 360, color="#12b76a")
    return image


def experiments() -> Image.Image:
    image, draw = base("Experiments", "a/b analytics")
    for idx, (label, value) in enumerate(
        [("CTR Lift", "+6.2%"), ("CVR Lift", "+8.4%"), ("p-value", "0.031")]
    ):
        card(draw, (285 + idx * 230, 130, 500 + idx * 230, 245), label, value, PALETTE[idx + 2])
    rect(draw, (285, 290, 760, 610), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((315, 318), "Impression Share", fill="#111827", font=FONT_24)
    donut(draw, (420, 370, 610, 560), [52, 48])
    rect(draw, (800, 290, 1218, 610), fill="#ffffff", outline="#e3e7ef", radius=20)
    bars(draw, (850, 350), [100, 63, 18], ["imp", "click", "buy"])
    return image


def model_ops() -> Image.Image:
    image, draw = base("Model Ops", "pipeline and personas")
    stages = ["Data", "Search", "Candidate", "Ranking", "Serving"]
    for idx, stage in enumerate(stages):
        rect(
            draw,
            (285 + idx * 180, 130, 440 + idx * 180, 230),
            fill="#ffffff",
            outline="#e3e7ef",
            radius=16,
        )
        draw.ellipse((305 + idx * 180, 154, 333 + idx * 180, 182), fill=PALETTE[idx])
        draw.text((305 + idx * 180, 192), stage, fill="#111827", font=FONT_16)
    rect(draw, (285, 280, 760, 620), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((315, 310), "Persona Distribution", fill="#111827", font=FONT_24)
    donut(draw, (430, 370, 630, 570), [22, 18, 17, 16, 14, 13])
    rect(draw, (800, 280, 1218, 620), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((830, 310), "Live Behavior Mix", fill="#111827", font=FONT_24)
    bars(draw, (850, 350), [84, 71, 39, 18], ["view", "cart", "buy", "search"])
    return image


def live_logs() -> Image.Image:
    image, draw = base("Live Logs", "behavior stream")
    labels = ["Search", "Reco", "Dashboard"]
    for idx, label in enumerate(labels):
        card(
            draw,
            (285 + idx * 300, 130, 560 + idx * 300, 250),
            label,
            f"{(idx + 2) * 438}",
            PALETTE[idx],
        )
    rect(draw, (285, 292, 820, 620), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((315, 320), "Surface Response Volume", fill="#111827", font=FONT_24)
    bars(draw, (340, 365), [100, 68, 42, 84, 52, 25], ["s-i", "s-c", "s-v", "r-i", "r-c", "r-v"])
    rect(draw, (860, 292, 1218, 620), fill="#ffffff", outline="#e3e7ef", radius=20)
    draw.text((890, 320), "Event Type Mix", fill="#111827", font=FONT_24)
    donut(draw, (940, 385, 1130, 575), [38, 24, 18, 12, 8])
    return image


def qa_gate() -> Image.Image:
    image, draw = base("QA Gate", "submission check")
    rect(draw, (285, 130, 1218, 620), fill="#ffffff", outline="#e3e7ef", radius=20)
    headers = ["Runtime", "Search", "Recommendation", "Training", "A/B"]
    for idx, header in enumerate(headers):
        y = 180 + idx * 78
        draw.text((330, y), header, fill="#111827", font=FONT_24)
        rect(draw, (650, y - 4, 760, y + 30), fill="#ecfdf3", radius=16)
        draw.text((682, y + 3), "PASS", fill="#087443", font=FONT_14)
        draw.line((315, y + 52, 1180, y + 52), fill="#edf0f5", width=2)
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
