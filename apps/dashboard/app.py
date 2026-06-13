from __future__ import annotations

import base64
import os
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.api_client import DashboardResponse, MarsApiClient

ROOT = Path(__file__).resolve().parents[2]
STYLE_PATH = Path(__file__).with_name("styles.css")
GUIDE_IMAGE_DIR = ROOT / "artifacts" / "ui_redesign"
SAMPLE_IMAGE_URL = "data/external/hm/raw/images/052/0528568002.jpg"
DEFAULT_EXPERIMENT_KEY = "mars_default"
PRODUCT_TONES = [
    ("#111827", "#9ca3af"),
    ("#0f766e", "#99f6e4"),
    ("#2563eb", "#bfdbfe"),
    ("#7c3aed", "#ddd6fe"),
    ("#b45309", "#fed7aa"),
    ("#15803d", "#bbf7d0"),
]
MARS_CHART_COLORS = [
    "#6d4aff",
    "#377dff",
    "#12b76a",
    "#f59e0b",
    "#e5484d",
    "#0ea5e9",
    "#111827",
    "#f97316",
]
EVENT_COLOR_MAP = {
    "exposure": "#6d4aff",
    "search": "#377dff",
    "view": "#0ea5e9",
    "cart": "#f59e0b",
    "purchase": "#12b76a",
}
EVENT_FLOW_TYPES = ("search", "view", "cart", "purchase")
EVENT_TYPE_ALIASES = {
    "click": "view",
    "product_view": "view",
    "view_item": "view",
    "add_to_cart": "cart",
    "cart_add": "cart",
    "conversion": "purchase",
    "order": "purchase",
}
EVENT_LABELS_KO = {
    "exposure": "노출",
    "search": "검색",
    "view": "조회",
    "cart": "장바구니",
    "purchase": "구매",
    "event": "이벤트",
}
SURFACE_LABELS_KO = {
    "search": "검색",
    "recommendation": "추천",
    "simulator": "시뮬레이터",
    "dashboard": "대시보드",
}
LIVE_METRIC_LABELS_KO = {
    "impressions": "노출",
    "clicks": "조회",
    "carts": "장바구니",
    "conversions": "구매",
}
NAV_ITEMS = [
    (
        "control-room",
        "Control Room",
        "nav_control_title",
        "nav_control_caption",
        "bi-house-door-fill",
    ),
    ("search", "Search", "nav_search_title", "nav_search_caption", "bi-search"),
    (
        "recommendation",
        "Recommendation",
        "nav_recommendation_title",
        "nav_recommendation_caption",
        "bi-stars",
    ),
    (
        "experiments",
        "Experiments",
        "nav_experiments_title",
        "nav_experiments_caption",
        "bi-graph-up-arrow",
    ),
    ("model-ops", "Model Ops", "nav_model_ops_title", "nav_model_ops_caption", "bi-cpu"),
    (
        "training",
        "Continuous Training",
        "nav_training_title",
        "nav_training_caption",
        "bi-broadcast-pin",
    ),
    ("qa-gate", "QA Gate", "nav_qa_gate_title", "nav_qa_gate_caption", "bi-check2-square"),
    ("guide", "Guide", "nav_guide_title", "nav_guide_caption", "bi-journal-text"),
]
NAV_BY_SLUG = {slug: section for slug, section, _, _, _ in NAV_ITEMS}
NAV_ITEM_BY_SLUG = {slug: item for slug, *item in NAV_ITEMS}
NAV_GROUPS = [
    ("nav_group_core", "Core", ("control-room", "search", "recommendation")),
    ("nav_group_ops", "Operations", ("experiments", "model-ops", "training")),
    ("nav_group_submit", "Submit", ("qa-gate", "guide")),
]

TRANSLATIONS = {
    "ko": {
        "nav_control_title": "Control Room",
        "nav_control_caption": "홈",
        "nav_experiments_title": "Experiments",
        "nav_experiments_caption": "실험 분석",
        "nav_model_ops_title": "Model Ops",
        "nav_model_ops_caption": "모델 운영",
        "nav_qa_gate_title": "QA Gate",
        "nav_qa_gate_caption": "제출 검증",
        "nav_guide_title": "Guide",
        "nav_guide_caption": "대시보드 안내",
        "sidebar_language": "언어",
        "sidebar_api": "API 연결",
        "nav_group_core": "대시보드",
        "nav_group_ops": "운영",
        "nav_group_submit": "제출 & 가이드",
        "api_base_url": "API base URL",
        "api_status_live": "Live API 연결됨",
        "api_status_demo": "Demo fallback 사용 중",
        "badge_live_api": "Live API",
        "badge_demo_data": "Demo data",
        "badge_api_metrics": "API metrics",
        "badge_fallback_metrics": "Fallback metrics",
        "control_subtitle": "검색, 추천, 실험, 모델 운영 상태를 한 화면에서 확인합니다.",
        "experiments_title": "Experiment Center",
        "experiments_subtitle": "Control과 treatment의 CTR, CVR, 유의성을 비교합니다.",
        "model_ops_subtitle": "데이터, 인덱스, 추천 파이프라인, 재학습 상태를 점검합니다.",
        "qa_title": "Requirement Check",
        "qa_caption": "요구사항별 기능과 지표가 현재 런타임에서 노출되는지 확인합니다. PASS는 충족, WARN은 데모/품질 확인 필요, FAIL은 현재 연결 상태에서 충족되지 않음을 의미합니다.",
        "qa_fallback_warning": "현재 QA Gate는 실제 FastAPI 런타임이 아니라 대시보드 fallback 데모 데이터로 검사 중입니다. API/Redis/전체 규모 데이터처럼 서버 연결이 필요한 항목은 FAIL 또는 WARN으로 보일 수 있습니다.",
        "qa_footer": "현재 dev-mode의 빨간색/노란색 행은 다음 점검을 위한 안내입니다. 먼저 표가 이해 가능하고 완성된 뒤 최적화를 진행하면 됩니다.",
        "guide_title": "Dashboard Guide",
        "guide_subtitle": "각 페이지에서 무엇을 보고 어떤 순서로 확인하면 되는지 빠르게 훑어봅니다.",
    },
    "en": {
        "nav_control_title": "Control Room",
        "nav_control_caption": "Home",
        "nav_experiments_title": "Experiments",
        "nav_experiments_caption": "A/B analysis",
        "nav_model_ops_title": "Model Ops",
        "nav_model_ops_caption": "Operations",
        "nav_qa_gate_title": "QA Gate",
        "nav_qa_gate_caption": "Submission check",
        "nav_guide_title": "Guide",
        "nav_guide_caption": "Dashboard help",
        "sidebar_language": "Language",
        "sidebar_api": "API Connection",
        "nav_group_core": "Dashboard",
        "nav_group_ops": "Operations",
        "nav_group_submit": "Submit & Guide",
        "api_base_url": "API base URL",
        "api_status_live": "Live API connected",
        "api_status_demo": "Using demo fallback",
        "badge_live_api": "Live API",
        "badge_demo_data": "Demo data",
        "badge_api_metrics": "API metrics",
        "badge_fallback_metrics": "Fallback metrics",
        "control_subtitle": "Monitor search, recommendation, experiments, and model operations in one place.",
        "experiments_title": "Experiment Center",
        "experiments_subtitle": "Compare CTR, CVR, and statistical significance for control and treatment.",
        "model_ops_subtitle": "Check data, index, recommendation pipeline, and retraining status.",
        "qa_title": "Requirement Check",
        "qa_caption": "Check whether each required feature and metric is visible in the current runtime. PASS means satisfied, WARN needs demo or quality review, and FAIL means the current connection does not satisfy it.",
        "qa_fallback_warning": "QA Gate is currently checking dashboard fallback demo data, not the live FastAPI runtime. Items that need API, Redis, or full-scale data can appear as FAIL or WARN.",
        "qa_footer": "Current dev-mode red/yellow rows are expected to guide the next pass. Optimization comes after this table is complete and understandable.",
        "guide_title": "Dashboard Guide",
        "guide_subtitle": "A quick walkthrough of what to check on each dashboard page and in which order.",
    },
}

TRANSLATIONS["ko"].update(
    {
        "nav_control_title": "통합 현황",
        "nav_control_caption": "홈",
        "nav_experiments_title": "실험 분석",
        "nav_experiments_caption": "A/B 테스트",
        "nav_model_ops_title": "모델 운영",
        "nav_model_ops_caption": "운영 상태",
        "nav_qa_gate_title": "제출 검증",
        "nav_qa_gate_caption": "요구사항 점검",
        "nav_guide_title": "가이드",
        "nav_guide_caption": "대시보드 안내",
        "topbar_kicker": "MARS 추천 콘솔",
        "sidebar_language": "언어",
        "sidebar_api": "API 연결",
        "api_status_live": "Live API 연결됨",
        "api_status_demo": "데모 대체 데이터 사용 중",
        "badge_live_api": "Live API",
        "badge_demo_data": "데모 데이터",
        "badge_api_metrics": "API 지표",
        "badge_fallback_metrics": "대체 지표",
        "control_subtitle": "검색, 추천, 실험, 모델 운영 상태를 한 화면에서 확인합니다.",
        "experiments_title": "실험 센터",
        "experiments_subtitle": "대조군과 실험군의 CTR, CVR, 유의성을 비교합니다.",
        "model_ops_subtitle": "데이터, 인덱스, 추천 파이프라인, 재학습 상태를 점검합니다.",
        "qa_title": "요구사항 점검",
        "qa_caption": "요구사항별 기능과 지표가 현재 런타임에서 노출되는지 확인합니다. PASS는 충족, WARN은 데모/품질 확인 필요, FAIL은 현재 연결 상태에서 충족되지 않음을 의미합니다.",
        "qa_fallback_warning": "현재 QA Gate는 실제 FastAPI 런타임이 아니라 대시보드 대체 데모 데이터로 검사 중입니다. API/Redis/전체 규모 데이터처럼 서버 연결이 필요한 항목은 FAIL 또는 WARN으로 보일 수 있습니다.",
        "qa_footer": "현재 dev-mode의 빨간색/노란색 행은 다음 점검을 위한 안내입니다. 먼저 표가 이해 가능하고 완성된 뒤 최적화를 진행하면 됩니다.",
        "guide_title": "대시보드 가이드",
        "guide_subtitle": "각 페이지에서 무엇을 보고 어떤 순서로 확인하면 되는지 빠르게 훑어봅니다.",
    }
)
TRANSLATIONS["en"].update({"topbar_kicker": "MARS Recommender Console"})
TRANSLATIONS["ko"].update(
    {
        "nav_search_title": "검색",
        "nav_search_caption": "검색 품질",
        "nav_recommendation_title": "추천",
        "nav_recommendation_caption": "추천 품질",
        "nav_training_title": "라이브 로그",
        "nav_training_caption": "행동 로그/학습",
    }
)
TRANSLATIONS["en"].update(
    {
        "nav_search_title": "Search",
        "nav_search_caption": "Retrieval quality",
        "nav_recommendation_title": "Recommendation",
        "nav_recommendation_caption": "Ranking pipeline",
        "nav_training_title": "Live Logs",
        "nav_training_caption": "Behavior stream",
    }
)

STATUS_LABELS_KO = {
    "ready": "준비됨",
    "missing": "누락",
    "demo": "데모",
    "watching": "관찰 중",
    "retrain_required": "재학습 필요",
    "unavailable": "사용 불가",
    "ok": "정상",
    "active": "활성",
    "archived": "보관됨",
    "true": "예",
    "false": "아니오",
}

ARTIFACT_LABELS_KO = {
    "data": "데이터",
    "search_index": "검색 인덱스",
    "recommender": "추천기",
    "reports": "리포트",
    "processed_dir": "처리 데이터",
    "processed data": "처리 데이터",
    "manifest": "산출물 명세",
    "recsys models": "추천 모델",
    "registry": "모델 등록부",
    "events": "이벤트",
    "products": "상품",
    "users": "사용자",
    "search_predictions": "검색 예측",
    "recommendation_predictions": "추천 예측",
}

REQ_GROUPS_KO = {
    "Runtime": "런타임",
    "Simulator": "시뮬레이터",
    "Search API": "검색 API",
    "Search Quality": "검색 품질",
    "Recommendation API": "추천 API",
    "Recommendation Quality": "추천 품질",
    "Feature Store": "피처 스토어",
    "A/B Testing": "A/B 테스트",
    "Continuous Training": "지속 학습",
}

REQ_ITEMS_KO = {
    "API health": "API 상태",
    "Redis feature store": "Redis 피처 스토어",
    "API and dashboard containers": "API/대시보드 컨테이너",
    "Artifact readiness": "산출물 준비 상태",
    "Products": "상품 수",
    "Users": "사용자 수",
    "Events": "이벤트 수",
    "6 personas": "6개 페르소나",
    "Event types": "이벤트 유형",
    "Train/valid/test split": "학습/검증/테스트 분할",
    "Required response fields": "필수 응답 필드",
    "Text search": "텍스트 검색",
    "Image search": "이미지 검색",
    "Hybrid search": "하이브리드 검색",
    "Search quality status": "검색 품질 상태",
    "Candidate latency": "후보 생성 지연",
    "Total latency": "전체 지연",
    "MAB exploration slot": "MAB 탐색 슬롯",
    "Redis lookup latency": "Redis 조회 지연",
    "CTR by bucket": "버킷별 CTR",
    "CVR by bucket": "버킷별 CVR",
    "p-value and 95% CI": "p-value 및 95% CI",
    "Live CTR/CVR": "실시간 CTR/CVR",
    "Retrain decision": "재학습 판단",
    "Model registry": "모델 레지스트리",
}


def compact_html(markup: str) -> str:
    return "".join(line.strip() for line in markup.splitlines())


def active_language() -> str:
    value = st.query_params.get("lang") or st.session_state.get("mars_lang", "en")
    if isinstance(value, list):
        value = value[0] if value else "en"
    lang = "ko" if str(value).lower().startswith("ko") else "en"
    st.session_state["mars_lang"] = lang
    return lang


def active_nav_slug() -> str:
    value = st.query_params.get("section", "control-room")
    if isinstance(value, list):
        value = value[0] if value else "control-room"
    slug = str(value or "control-room")
    return slug if slug in NAV_BY_SLUG else "control-room"


def tr(key: str, lang: str) -> str:
    table = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return table.get(key, TRANSLATIONS["en"].get(key, key))


def ui_text(lang: str, ko_text: str, en_text: str) -> str:
    return ko_text if lang == "ko" else en_text


def status_label(value: Any, lang: str) -> str:
    text = str(value if value is not None else "unknown")
    if lang != "ko":
        return text
    key = text.strip().lower().replace(" ", "_")
    return STATUS_LABELS_KO.get(key, text)


def retrain_trigger_label(training: dict[str, Any], lang: str) -> str:
    active = bool(training.get("retrain_trigger_active", training.get("should_retrain", False)))
    return ui_text(lang, "필요", "Needed") if active else ui_text(lang, "대기 중", "Monitoring")


def retrain_trigger_hint(training: dict[str, Any], lang: str) -> str:
    active = bool(training.get("retrain_trigger_active", training.get("should_retrain", False)))
    if active:
        return ui_text(lang, "조건 충족", "conditions met")
    return ui_text(lang, "조건 미충족", "conditions not met")


def training_action_label(value: Any, lang: str) -> str:
    text = str(value or "").strip()
    if lang != "ko":
        return text or "Waiting for monitoring signal."
    lowered = text.lower()
    if not text or "waiting for monitoring signal" in lowered:
        return "모니터링 신호 대기 중"
    if "no retrain trigger condition is active" in lowered:
        return "재학습 조건 미충족: 모니터링 유지"
    return text


def artifact_label(value: Any, lang: str) -> str:
    text = str(value)
    key = text.strip().replace("_", " ").lower()
    if lang != "ko":
        return text.replace("_", " ").title()
    return ARTIFACT_LABELS_KO.get(key, ARTIFACT_LABELS_KO.get(text, text.replace("_", " ")))


def nav_href(slug: str, lang: str) -> str:
    return f"?section={slug}&lang={lang}"


def render_sidebar_nav(active_slug: str, lang: str) -> None:
    nav_groups = []
    for group_key, fallback_label, slugs in NAV_GROUPS:
        links = []
        for slug in slugs:
            _, title_key, caption_key, icon = NAV_ITEM_BY_SLUG[slug]
            active = " active" if slug == active_slug else ""
            label = tr(title_key, lang)
            caption = tr(caption_key, lang)
            links.append(
                compact_html(
                    f"""
                    <a class="mars-nav-item{active}" href="{nav_href(slug, lang)}" target="_self">
                      <i class="bi {icon}"></i>
                      <span>{escape(label)}</span>
                      <small>{escape(caption)}</small>
                    </a>
                    """
                )
            )
        nav_groups.append(
            compact_html(
                f"""
                <section class="mars-nav-group" aria-label="{escape(tr(group_key, lang) or fallback_label)}">
                  <div class="mars-nav-group-label">{escape(tr(group_key, lang) or fallback_label)}</div>
                  {"".join(links)}
                </section>
                """
            )
        )
    st.sidebar.markdown(
        compact_html(
            f"""
            <div class="mars-sidebar-brand">
              <div class="mars-brand-mark" aria-hidden="true"><i class="bi bi-grid-1x2-fill"></i></div>
              <div>
                <span>Multimodal AI</span>
                <strong>MARS Console</strong>
              </div>
            </div>
            <nav class="mars-nav">
              {"".join(nav_groups)}
            </nav>
            <div class="mars-sidebar-spacer"></div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_sidebar_controls(active_slug: str, lang: str, default_api_url: str) -> str:
    ko_active = " active" if lang == "ko" else ""
    en_active = " active" if lang == "en" else ""
    st.sidebar.markdown(
        compact_html(
            f"""
            <div class="mars-sidebar-tools">
              <div class="mars-sidebar-label"><i class="bi bi-translate"></i>{escape(tr("sidebar_language", lang))}</div>
              <div class="mars-lang-switch" aria-label="{escape(tr("sidebar_language", lang))}">
                <a class="mars-lang-option{ko_active}" href="{nav_href(active_slug, "ko")}" target="_self">
                  <span>한국어</span>
                  <small>KO</small>
                </a>
                <a class="mars-lang-option{en_active}" href="{nav_href(active_slug, "en")}" target="_self">
                  <span>English</span>
                  <small>EN</small>
                </a>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        compact_html(
            f"""
            <div class="mars-sidebar-api-label">
              <i class="bi bi-plug"></i>{escape(tr("sidebar_api", lang))}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    return st.sidebar.text_input(
        tr("api_base_url", lang),
        value=default_api_url,
        key="api_base_url",
        label_visibility="collapsed",
    )


def render_sidebar_connection_status(
    health: DashboardResponse,
    metrics_response: DashboardResponse,
    lang: str,
) -> None:
    live = not health.is_demo and health.data.get("status") == "ok"
    source_live = not metrics_response.is_demo
    label = tr("api_status_live", lang) if live and source_live else tr("api_status_demo", lang)
    tone = "ok" if live and source_live else "warn"
    icon = "bi-check-circle" if live and source_live else "bi-broadcast-pin"
    st.sidebar.markdown(
        compact_html(
            f"""
            <div class="mars-connection-status {tone}">
              <i class="bi {icon}"></i>
              <span>{escape(label)}</span>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def experiment_key_picker(
    lang: str,
    *,
    key: str,
    allow_disabled: bool = False,
    label: str | None = None,
) -> str | None:
    default_label = ui_text(lang, "기본 A/B", "Default A/B")
    custom_label = ui_text(lang, "직접 입력", "Custom key")
    disabled_label = ui_text(lang, "A/B 끄기", "A/B off")
    options = [default_label]
    if allow_disabled:
        options.append(disabled_label)
    options.append(custom_label)

    selected = st.segmented_control(
        label or ui_text(lang, "A/B 실험", "A/B experiment"),
        options,
        default=default_label,
        key=f"{key}_mode",
    )
    if selected == custom_label:
        value = st.text_input(
            ui_text(lang, "직접 실험 키", "Custom experiment key"),
            value=DEFAULT_EXPERIMENT_KEY,
            key=f"{key}_custom",
        )
        return value.strip() or DEFAULT_EXPERIMENT_KEY
    if selected == disabled_label:
        st.caption(
            ui_text(
                lang,
                "A/B 없이 기본 추천 전략으로 요청합니다.",
                "Requests the default recommendation strategy without A/B bucketing.",
            )
        )
        return None

    st.caption(
        ui_text(
            lang,
            f"{DEFAULT_EXPERIMENT_KEY}. A/B 버킷은 사용자 ID로 자동 배정됩니다.",
            f"{DEFAULT_EXPERIMENT_KEY}. Buckets are assigned by user ID.",
        )
    )
    return DEFAULT_EXPERIMENT_KEY


def inject_css() -> None:
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def system_event_counts(system: dict[str, Any]) -> tuple[int, int, int]:
    base_events = int(system.get("base_events", system.get("events", 0)) or 0)
    live_events = int(system.get("live_events", system.get("logged_events", 0)) or 0)
    total_events = int(system.get("total_events", base_events + live_events) or 0)
    return base_events, live_events, total_events


def system_event_caption(system: dict[str, Any], lang: str) -> str:
    base_events, live_events, _ = system_event_counts(system)
    return ui_text(
        lang,
        f"기준 {format_int(base_events)} + 라이브 {format_int(live_events)}",
        f"base {format_int(base_events)} + live {format_int(live_events)}",
    )


def format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def lift_tone(value: Any) -> str:
    lift = safe_float(value)
    if lift > 0:
        return "green"
    if lift < 0:
        return "red"
    return "orange"


def lift_icon(value: Any) -> str:
    lift = safe_float(value)
    if lift > 0:
        return "bi-arrow-up-right-circle"
    if lift < 0:
        return "bi-arrow-down-right-circle"
    return "bi-dash-circle"


def lift_hint(value: Any, lang: str, ko_metric: str, en_metric: str) -> str:
    lift = safe_float(value)
    if lift > 0:
        return ui_text(lang, f"{ko_metric} 개선", f"{en_metric} improved")
    if lift < 0:
        return ui_text(lang, f"{ko_metric} 하락", f"{en_metric} lower")
    return ui_text(lang, "차이 없음", "no lift")


def treatment_lift_hint(value: Any, lang: str) -> str:
    lift = safe_float(value)
    if lift > 0:
        return ui_text(lang, "실험군이 대조군 대비 우세", "treatment above control")
    if lift < 0:
        return ui_text(lang, "실험군이 대조군 대비 낮음", "treatment below control")
    return ui_text(lang, "대조군과 동일", "same as control")


def format_ms(value: Any) -> str:
    try:
        return f"{float(value):.1f} ms"
    except (TypeError, ValueError):
        return "0.0 ms"


def format_price(value: Any) -> str:
    try:
        return f"W {int(float(value)):,}"
    except (TypeError, ValueError):
        return "W 0"


def compact_runtime_label(value: Any, *, max_chars: int = 18) -> str:
    text = str(value or "unknown")
    if text == "unknown":
        return text
    if ":" in text:
        text = text.split(":", 1)[0]
    if "/" in text and len(text) > max_chars:
        text = text.rsplit("/", 1)[-1]
    if len(text) > max_chars:
        return f"{text[: max_chars - 3]}..."
    return text


def compact_source_label(value: Any, *, lang: str) -> str:
    text = str(value or "unknown")
    parts = [part for part in text.split(":") if part]
    if len(parts) >= 2:
        prefix = parts[0]
        surface = parts[-1]
        if lang == "ko":
            surface_label = SURFACE_LABELS_KO.get(surface, surface)
            if prefix == "live":
                return f"실시간 {surface_label}"
            if prefix == "demo":
                return f"데모 {surface_label}"
        if prefix in {"live", "demo"}:
            return f"{prefix}:{surface}"
    return compact_runtime_label(text, max_chars=20)


def display_log_source_label(value: Any, *, lang: str) -> str:
    text = str(value or "unknown").strip()
    if text == "api_events_jsonl":
        return ui_text(lang, "실시간 행동 로그", "live behavior stream")
    if text in {"demo", "fallback"}:
        return ui_text(lang, "데모 로그", "demo stream")
    return compact_runtime_label(text, max_chars=22)


def display_strategy_label(value: Any, *, lang: str) -> str:
    text = str(value or "unknown").strip()
    normalized = text.replace("_", "").replace("-", "").lower()
    if normalized == "rankonlycontrol":
        return ui_text(lang, "기본 랭킹", "Rank only")
    if normalized == "control":
        return ui_text(lang, "대조군", "Control")
    if normalized == "treatment":
        return ui_text(lang, "실험군", "Treatment")
    return compact_runtime_label(text, max_chars=22)


def display_reason_label(value: Any, *, lang: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part for part in text.split(":") if part]
    if not parts:
        return text
    head = display_strategy_label(parts[0], lang=lang)
    tail_map = {
        "long_term_preference": ui_text(lang, "장기 선호", "long-term preference"),
        "session_interest": ui_text(lang, "세션 관심", "session interest"),
        "popular_fallback": ui_text(lang, "인기 기반", "popular fallback"),
        "exploration": ui_text(lang, "탐색 슬롯", "exploration slot"),
    }
    tail = [tail_map.get(part, part.replace("_", " ")) for part in parts[1:]]
    return " · ".join([head, *tail])


def recommendation_table_frame(rows: list[dict[str, Any]], *, lang: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if not frame.empty and "reason" in frame.columns:
        frame = frame.copy()
        frame["reason"] = frame["reason"].map(lambda value: display_reason_label(value, lang=lang))
    return frame


def figure_has_pie_trace(fig: go.Figure) -> bool:
    return any(str(getattr(trace, "type", "")).lower() == "pie" for trace in fig.data)


def pie_text_positions(values: Any, *, outside_threshold: float = 0.04) -> str | list[str]:
    raw_values = [] if values is None else list(values)
    numeric_values = []
    for value in raw_values:
        numeric_values.append(0.0 if pd.isna(value) else float(value))
    total = sum(value for value in numeric_values if value > 0)
    if total <= 0:
        return "inside"
    positions = [
        "outside" if value > 0 and value / total < outside_threshold else "inside"
        for value in numeric_values
    ]
    return positions if "outside" in positions else "inside"


def plotly_clean(fig: go.Figure, *, height: int = 330) -> go.Figure:
    title_text = fig.layout.title.text if fig.layout.title else None
    if str(title_text).strip().lower() == "undefined":
        title_text = None
    showlegend = bool(fig.layout.showlegend) if fig.layout.showlegend is not None else True
    is_pie = figure_has_pie_trace(fig)
    if is_pie:
        chart_height = max(height, 370 if showlegend else height)
        margin = dict(l=42, r=42, t=28, b=96 if showlegend else 34)
        legend_y = -0.09
        legend_item_width = 94
        legend_gap = 8
    else:
        chart_height = max(height, 380 if showlegend else height)
        margin = dict(l=64, r=34, t=34, b=122 if showlegend else 66)
        legend_y = -0.19
        legend_item_width = 78
        legend_gap = 10
    fig.update_layout(
        height=chart_height,
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827", family="Malgun Gothic, Segoe UI, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#334155"),
            itemwidth=legend_item_width,
            itemsizing="constant",
            tracegroupgap=legend_gap,
        ),
        legend_title_text="",
        colorway=MARS_CHART_COLORS,
        uniformtext_minsize=10,
        uniformtext_mode="show" if is_pie else "hide",
    )
    if title_text:
        fig.update_layout(title=dict(text=title_text, font=dict(size=16), x=0.02, xanchor="left"))
    else:
        fig.update_layout(title_text="")
    if is_pie:
        fig.update_traces(
            automargin=True,
            domain=dict(x=[0.06, 0.94], y=[0.16, 0.94]),
            selector=dict(type="pie"),
        )
    fig.update_xaxes(automargin=True, tickfont=dict(size=11), title_standoff=14)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11), title_standoff=16)
    return fig


def pie_readable(fig: go.Figure, *, textinfo: str = "label+percent") -> go.Figure:
    fig.update_layout(showlegend=True)
    for trace in fig.data:
        if str(getattr(trace, "type", "")).lower() != "pie":
            continue
        trace.update(
            textinfo=textinfo,
            textposition=pie_text_positions(trace.values),
            insidetextorientation="horizontal",
            textfont=dict(size=12, color="#ffffff"),
            outsidetextfont=dict(size=11, color="#334155"),
            marker=dict(line=dict(color="#ffffff", width=2)),
            hovertemplate="%{label}<br>%{value}<extra></extra>",
            automargin=True,
        )
    return fig


def badge_html(label: str, *, tone: str = "neutral", icon: str = "bi-check-circle") -> str:
    return f'<span class="mars-badge {tone}"><i class="bi {icon}"></i>{escape(label)}</span>'


def topbar(
    *,
    section: str,
    health: DashboardResponse,
    metrics_response: DashboardResponse,
    metrics: dict[str, Any],
    lang: str,
) -> None:
    system = metrics.get("system", {})
    api_ok = not health.is_demo and health.data.get("status") == "ok"
    api_badge = badge_html(
        tr("badge_live_api", lang) if api_ok else tr("badge_demo_data", lang),
        tone="ok" if api_ok else "warn",
        icon="bi-broadcast",
    )
    mode = str(metrics.get("mode") or system.get("mode") or "demo")
    source = (
        tr("badge_api_metrics", lang)
        if not metrics_response.is_demo
        else tr("badge_fallback_metrics", lang)
    )
    st.markdown(
        compact_html(
            f"""
            <div class="mars-topbar">
              <div class="mars-brand">
                <div class="mars-logo"><i class="bi bi-grid-1x2-fill"></i></div>
                <div>
                  <div class="mars-brand-kicker">{escape(tr("topbar_kicker", lang))}</div>
                  <h1>{escape(section)}</h1>
                </div>
              </div>
              <div class="mars-top-actions">
                {api_badge}
                {badge_html(mode, tone="neutral", icon="bi-box")}
                {badge_html(source, tone="neutral", icon="bi-activity")}
                {badge_html(str(system.get("active_model_version", "unknown")), tone="neutral", icon="bi-cpu")}
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        compact_html(
            f"""
            <div class="mars-page-head">
              <h2>{escape(title)}</h2>
              <p>{escape(subtitle)}</p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def metric_card_html(
    label: str,
    value: str,
    hint: str,
    *,
    tone: str = "purple",
    icon: str = "bi-activity",
) -> str:
    return compact_html(
        f"""
        <div class="mars-metric-card {tone}">
          <div class="metric-top">
            <span>{escape(label)}</span>
            <i class="bi {icon}"></i>
          </div>
          <strong>{escape(value)}</strong>
          <small>{escape(hint)}</small>
        </div>
        """
    )


def metric_grid_html(cards: str, *, columns: int = 5, class_name: str = "") -> str:
    return compact_html(
        f'<div class="mars-metric-grid {escape(class_name)}" style="--metric-columns:{columns};">'
        f"{cards}</div>"
    )


def panel_html(title: str, body: str, *, class_name: str = "") -> None:
    st.markdown(
        compact_html(f'<div class="mars-panel {class_name}"><h3>{escape(title)}</h3>{body}</div>'),
        unsafe_allow_html=True,
    )


def chart_heading_html(title: str, caption: str = "") -> str:
    caption_html = f"<p>{escape(caption)}</p>" if caption else ""
    return compact_html(
        f"""
        <div class="mars-chart-heading">
          <strong>{escape(title)}</strong>
          {caption_html}
        </div>
        """
    )


@st.cache_data(show_spinner=False)
def image_data_uri(path_text: str, mtime: float) -> str:
    path = Path(path_text)
    if not path.exists():
        return ""
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def guide_card_html(
    *,
    title: str,
    eyebrow: str,
    body: str,
    bullets: list[str],
    image_name: str,
) -> str:
    image_path = GUIDE_IMAGE_DIR / image_name
    image_mtime = image_path.stat().st_mtime if image_path.exists() else 0.0
    image_uri = image_data_uri(str(image_path), image_mtime)
    bullet_html = "".join(f"<li>{escape(item)}</li>" for item in bullets)
    image_html = (
        f'<img src="{image_uri}" alt="{escape(title)} dashboard screenshot" />'
        if image_uri
        else '<div class="mars-guide-missing">Screenshot pending</div>'
    )
    return compact_html(
        f"""
        <article class="mars-guide-card">
          <div class="mars-guide-copy">
            <span>{escape(eyebrow)}</span>
            <h3>{escape(title)}</h3>
            <p>{escape(body)}</p>
            <ul>{bullet_html}</ul>
          </div>
          <div class="mars-guide-media">{image_html}</div>
        </article>
        """
    )


def product_card_html(
    row: dict[str, Any],
    index: int,
    *,
    image_lookup: dict[str, str],
    lang: str = "en",
) -> str:
    tone_a, tone_b = PRODUCT_TONES[index % len(PRODUCT_TONES)]
    product_id = escape(str(row.get("product_id", "")))
    name = escape(str(row.get("name", product_id or "Unknown product")))
    category = escape(str(row.get("category", "fashion")))
    price = format_price(row.get("price"))
    score = float(row.get("score", 0) or 0)
    exploration = bool(row.get("is_exploration", False))
    rank = index + 1
    tag = (
        ui_text(lang, f"탐색 {rank}", f"Explore {rank}")
        if exploration
        else ui_text(lang, f"순위 {rank}", f"Rank {rank}")
    )
    image_uri = image_display_uri(product_image_ref(row, image_lookup))
    visual = (
        f'<img class="product-thumb" src="{image_uri}" alt="{name}" loading="lazy" />'
        if image_uri
        else '<div class="product-thumb missing"><i class="bi bi-image"></i></div>'
    )
    return compact_html(
        f"""
        <div class="mars-product-card" style="--tone-a:{tone_a};--tone-b:{tone_b};">
          <div class="product-visual">{visual}<span class="product-rank">{escape(tag)}</span></div>
          <div class="product-body">
            <div class="product-title-row">
              <div class="product-name">{name}</div>
              <b class="product-score">{score:.3f}</b>
            </div>
            <div class="product-meta">
              <span>{product_id}</span>
              <span>{category}</span>
            </div>
            <div class="product-foot">
              <span>{escape(ui_text(lang, "가격", "Price"))}</span>
              <strong>{escape(price)}</strong>
            </div>
          </div>
        </div>
        """
    )


def product_grid(
    rows: list[dict[str, Any]],
    *,
    empty: str = "No products returned.",
    lang: str = "en",
) -> None:
    if not rows:
        st.info(empty)
        return
    image_lookup = load_product_images()
    cards = "".join(
        product_card_html(row, index, image_lookup=image_lookup, lang=lang)
        for index, row in enumerate(rows[:8])
    )
    st.markdown(f'<div class="mars-product-grid">{cards}</div>', unsafe_allow_html=True)


def chip_list_html(values: Any, *, empty: str, limit: int = 4) -> str:
    if not isinstance(values, (list, tuple, set)) or not values:
        return f'<span class="mars-chip muted">{escape(empty)}</span>'
    chips = []
    for value in list(values)[:limit]:
        text = str(value or "").strip()
        if text:
            chips.append(f'<span class="mars-chip">{escape(text)}</span>')
    if len(values) > limit:
        chips.append(f'<span class="mars-chip muted">+{len(values) - limit}</span>')
    return "".join(chips) if chips else f'<span class="mars-chip muted">{escape(empty)}</span>'


def session_context_html(context: dict[str, Any], *, lang: str = "en") -> str:
    if not isinstance(context, dict):
        context = {}
    event_counts = (
        context.get("event_counts") if isinstance(context.get("event_counts"), dict) else {}
    )
    strategy = display_strategy_label(
        context.get("recommendation_strategy_label")
        or context.get("recommendation_strategy")
        or "unknown",
        lang=lang,
    )
    history_count = int(context.get("history_count", 0) or 0)
    recent_events = int(context.get("num_recent_events", 0) or 0)
    recent_clicks = (
        context.get("recent_clicks") if isinstance(context.get("recent_clicks"), list) else []
    )
    recent_products = (
        context.get("recent_products") if isinstance(context.get("recent_products"), list) else []
    )
    recent_categories = (
        context.get("recent_categories")
        if isinstance(context.get("recent_categories"), list)
        else []
    )
    cold_start = bool(context.get("cold_start", False))
    session_interest = str(context.get("session_interest") or "").strip()
    status = (
        ui_text(lang, "콜드스타트", "Cold start")
        if cold_start
        else ui_text(lang, "기존 사용자", "Known user")
    )
    total_events = (
        sum(int(value or 0) for value in event_counts.values()) if event_counts else recent_events
    )
    clicked = len(recent_clicks)
    viewed = int(event_counts.get("view", 0) or 0)
    purchased = int(event_counts.get("purchase", 0) or 0)
    category_chips = chip_list_html(
        recent_categories,
        empty=ui_text(lang, "최근 카테고리 없음", "No recent categories"),
        limit=4,
    )
    product_chips = chip_list_html(
        recent_products or recent_clicks,
        empty=ui_text(lang, "최근 상품 없음", "No recent products"),
        limit=3,
    )
    interest = session_interest or ui_text(lang, "명시 관심 없음", "No explicit interest")
    return compact_html(
        f"""
        <div class="mars-context-grid">
          <div class="mars-context-card primary">
            <span>{escape(ui_text(lang, "세션 상태", "Session state"))}</span>
            <strong>{escape(status)}</strong>
            <small>{escape(ui_text(lang, f"누적 이력 {format_int(history_count)}건", f"{format_int(history_count)} historical events"))}</small>
          </div>
          <div class="mars-context-card">
            <span>{escape(ui_text(lang, "최근 반응", "Recent response"))}</span>
            <strong>{format_int(total_events)}</strong>
            <small>{escape(ui_text(lang, f"조회 {format_int(viewed)} · 클릭 {format_int(clicked)} · 구매 {format_int(purchased)}", f"views {format_int(viewed)} · clicks {format_int(clicked)} · purchases {format_int(purchased)}"))}</small>
          </div>
          <div class="mars-context-card">
            <span>{escape(ui_text(lang, "서빙 전략", "Serving strategy"))}</span>
            <strong>{escape(strategy)}</strong>
            <small>{escape(interest)}</small>
          </div>
        </div>
        <div class="mars-chip-row">
          <b>{escape(ui_text(lang, "관심 카테고리", "Categories"))}</b>
          {category_chips}
        </div>
        <div class="mars-chip-row">
          <b>{escape(ui_text(lang, "최근 상품", "Recent products"))}</b>
          {product_chips}
        </div>
        """
    )


def product_summary_html(
    row: dict[str, Any],
    *,
    image_available: bool,
    image_ref: str | None,
    lang: str = "en",
) -> str:
    product_id = str(row.get("product_id", "")).strip()
    name = str(row.get("name", product_id or "unknown")).strip()
    category = str(row.get("category", "unknown")).strip()
    reason = display_reason_label(row.get("reason") or row.get("source"), lang=lang)
    reason_html = (
        f"<div><span>{escape(ui_text(lang, '근거', 'Reason'))}</span><b>{escape(reason)}</b></div>"
        if reason
        else ""
    )
    image_state = (
        ui_text(lang, "이미지 표시 가능", "image available")
        if image_available
        else ui_text(lang, "이미지 경로 없음", "image unavailable")
    )
    image_hint = image_ref if image_ref and not image_available else image_state
    return compact_html(
        f"""
        <div class="mars-info-list">
          <div><span>{escape(ui_text(lang, "상품 ID", "Product ID"))}</span><b>{escape(product_id)}</b></div>
          <div><span>{escape(ui_text(lang, "상품명", "Name"))}</span><b>{escape(name)}</b></div>
          <div><span>{escape(ui_text(lang, "카테고리", "Category"))}</span><b>{escape(category)}</b></div>
          <div><span>{escape(ui_text(lang, "가격", "Price"))}</span><b>{escape(format_price(row.get("price")))}</b></div>
          <div><span>{escape(ui_text(lang, "점수", "Score"))}</span><b>{float(row.get("score", 0) or 0):.3f}</b></div>
          <div><span>{escape(ui_text(lang, "이미지", "Image"))}</span><b>{escape(image_hint)}</b></div>
          {reason_html}
        </div>
        """
    )


def source_badge(response: DashboardResponse, lang: str = "en") -> None:
    if response.is_demo:
        st.markdown(
            '<span class="mars-status warn"><i class="bi bi-broadcast-pin"></i>'
            f" {escape(ui_text(lang, '데모 대체 데이터', 'Demo fallback data'))}</span>",
            unsafe_allow_html=True,
        )
        if response.error:
            st.caption(
                f"{ui_text(lang, 'API 대체 데이터 사용 사유', 'API fallback reason')}: {response.error}"
            )
    else:
        st.markdown(
            '<span class="mars-status ok"><i class="bi bi-check-circle"></i>'
            f" {escape(ui_text(lang, 'Live API 연결됨', 'Live API connected'))}</span>",
            unsafe_allow_html=True,
        )


def status_text(ok: bool, *, warn: bool = False) -> str:
    if ok:
        return "PASS"
    return "WARN" if warn else "FAIL"


def target_status(value: Any, target: float, *, higher_is_better: bool = True) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "FAIL"
    passed = numeric >= target if higher_is_better else numeric <= target
    return status_text(passed)


def status_dataframe(rows: list[dict[str, Any]], lang: str = "en") -> None:
    if not rows:
        st.info(ui_text(lang, "아직 검증 행이 없습니다.", "No verification rows available yet."))
        return

    counts = {
        status: sum(1 for row in rows if str(row.get("status", "")).upper() == status)
        for status in ("PASS", "WARN", "FAIL")
    }
    total = max(len(rows), 1)
    summary = compact_html(
        f"""
        <div class="mars-qa-summary">
          <div class="pass"><span>PASS</span><strong>{counts["PASS"]}</strong><small>{counts["PASS"] / total:.0%}</small></div>
          <div class="warn"><span>WARN</span><strong>{counts["WARN"]}</strong><small>{counts["WARN"] / total:.0%}</small></div>
          <div class="fail"><span>FAIL</span><strong>{counts["FAIL"]}</strong><small>{counts["FAIL"] / total:.0%}</small></div>
        </div>
        """
    )
    header = (
        ("그룹", "항목", "현재", "목표", "상태", "근거")
        if lang == "ko"
        else ("Group", "Item", "Current", "Target", "Status", "Where")
    )
    body_rows = []
    for row in rows:
        status = str(row.get("status", "WARN")).upper()
        tone = status.lower() if status in {"PASS", "WARN", "FAIL"} else "warn"
        body_rows.append(
            compact_html(
                f"""
                <tr>
                  <td class="group">{escape(str(row.get("group", "")))}</td>
                  <td class="item">{escape(str(row.get("item", "")))}</td>
                  <td>{escape(str(row.get("current", "")))}</td>
                  <td>{escape(str(row.get("target", "")))}</td>
                  <td><span class="mars-qa-pill {tone}">{escape(status)}</span></td>
                  <td class="where">{escape(str(row.get("where", "")))}</td>
                </tr>
                """
            )
        )
    table = compact_html(
        f"""
        <div class="mars-qa-panel">
          <table class="mars-qa-table">
            <thead>
              <tr>
                <th>{escape(header[0])}</th>
                <th>{escape(header[1])}</th>
                <th>{escape(header[2])}</th>
                <th>{escape(header[3])}</th>
                <th>{escape(header[4])}</th>
                <th>{escape(header[5])}</th>
              </tr>
            </thead>
            <tbody>{"".join(body_rows)}</tbody>
          </table>
        </div>
        """
    )
    st.markdown(summary + table, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_product_images() -> dict[str, str]:
    products_path = ROOT / "data" / "processed" / "products.parquet"
    if not products_path.exists():
        return {}
    try:
        frame = pd.read_parquet(products_path, columns=["product_id", "image_path"])
    except Exception:
        return {}
    image_map: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        product_id = str(row.product_id or "").strip()
        image_path = row.image_path
        if product_id and pd.notna(image_path):
            text_path = str(image_path).strip()
            if text_path:
                image_map[product_id] = text_path
    return image_map


def resolve_image_source(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        return text
    path = Path(text)
    candidates = [path, ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def image_display_uri(image_ref: Any) -> str:
    source = resolve_image_source(image_ref)
    if not source:
        return ""
    if source.startswith(("http://", "https://")):
        return source
    path = Path(source)
    return image_data_uri(str(path), path.stat().st_mtime if path.exists() else 0.0)


def product_image_ref(row: dict[str, Any], image_lookup: dict[str, str]) -> str:
    product_id = str(row.get("product_id", "")).strip()
    return str(row.get("image_url") or image_lookup.get(product_id) or "").strip()


def product_preview_card_html(
    row: dict[str, Any],
    *,
    selected_label: str,
    image_uri: str,
    image_ref: str | None,
    lang: str = "en",
) -> str:
    product_id = str(row.get("product_id", "")).strip()
    name = str(row.get("name", product_id or "unknown")).strip()
    image_available = bool(image_uri)
    if image_available:
        media_html = (
            f'<img class="mars-preview-thumb" src="{escape(image_uri, quote=True)}" '
            f'alt="{escape(name, quote=True)}" loading="lazy" />'
        )
        state = ui_text(lang, "이미지 표시 가능", "image available")
    else:
        media_html = '<div class="mars-preview-missing"><i class="bi bi-image"></i></div>'
        state = (
            ui_text(lang, "이미지 경로 없음", "image path unavailable")
            if not image_ref
            else ui_text(lang, "이미지 파일 없음", "image file unavailable")
        )
    path_html = (
        f"<code>{escape(str(image_ref))}</code>"
        if image_ref and not image_available
        else f"<span>{escape(state)}</span>"
    )
    return compact_html(
        f"""
        <div class="mars-preview-card">
          <div class="mars-preview-media">{media_html}</div>
          <div class="mars-preview-caption">
            <strong>{escape(selected_label)}</strong>
            {path_html}
          </div>
        </div>
        """
    )


def render_product_preview(rows: list[dict[str, Any]], *, key: str, lang: str = "en") -> None:
    if not rows:
        return
    image_lookup = load_product_images()
    labels: list[str] = []
    rows_by_label: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        product_id = str(row.get("product_id", ""))
        name = str(row.get("name", product_id))
        label = f"{index}. {product_id} - {name}"
        labels.append(label)
        rows_by_label[label] = row
    selected = st.selectbox(
        ui_text(lang, "상품 이미지 미리보기", "Preview product image"),
        labels,
        key=f"{key}_image_select",
    )
    row = rows_by_label[selected]
    product_id = str(row.get("product_id", ""))
    image_ref = str(row.get("image_url") or image_lookup.get(product_id) or "").strip()
    image_uri = image_display_uri(image_ref)
    left, right = st.columns([0.95, 1.05], gap="medium")
    with left:
        st.markdown(
            product_preview_card_html(
                row,
                selected_label=selected,
                image_uri=image_uri,
                image_ref=image_ref or None,
                lang=lang,
            ),
            unsafe_allow_html=True,
        )
    with right:
        panel_html(
            ui_text(lang, "선택 상품 요약", "Selected product"),
            product_summary_html(
                row,
                image_available=bool(image_uri),
                image_ref=image_ref or None,
                lang=lang,
            ),
            class_name="preview-summary-panel",
        )


def render_search_feedback_controls(
    client: MarsApiClient,
    *,
    rows: list[dict[str, Any]],
    query: str,
    debug: dict[str, Any],
    limit: int = 5,
    lang: str = "en",
) -> None:
    if not rows or not debug.get("search_id"):
        return
    search_id = str(debug.get("search_id"))
    session_id = str(debug.get("session_id") or f"S-dashboard-{search_id}")
    user_id = "dashboard-search-user"
    st.markdown(f"#### {ui_text(lang, '검색 피드백', 'Search feedback')}")
    st.caption(
        ui_text(
            lang,
            "클릭 피드백은 검색 화면의 view/cart/purchase 이벤트로 기록되며, 검증 후 검색 행동 모델 자동 갱신에 활용할 수 있습니다.",
            "Click feedback is logged as search-surface view/cart/purchase events and can feed the automated search behavior-model refresh after validation.",
        )
    )
    for rank, row in enumerate(rows[:limit], start=1):
        product_id = str(row.get("product_id", ""))
        if not product_id:
            continue
        label = f"{rank}. {product_id} - {row.get('name', '')}"
        cols = st.columns([3, 1, 1, 1])
        cols[0].caption(label)
        for event_type, column in zip(("view", "cart", "purchase"), cols[1:], strict=True):
            button_label = {
                "view": ui_text(lang, "조회", "View"),
                "cart": ui_text(lang, "장바구니", "Cart"),
                "purchase": ui_text(lang, "구매", "Purchase"),
            }[event_type]
            if column.button(
                button_label,
                key=f"search_feedback_{search_id}_{product_id}_{event_type}",
                use_container_width=True,
            ):
                event_response = client.record_event(
                    user_id=user_id,
                    event_type=event_type,
                    product_id=product_id,
                    session_id=session_id,
                    query=query,
                    metadata={
                        "source_surface": "search",
                        "surface": "search",
                        "event_role": "user_action",
                        "search_id": search_id,
                        "rank": rank,
                        "result_name": str(row.get("name", "")),
                        "score": row.get("score"),
                    },
                )
                if event_response.is_demo or not event_response.data.get("accepted"):
                    st.warning(
                        f"{ui_text(lang, '피드백을 기록하지 못했습니다', 'Could not log')} {event_type} / {product_id}: "
                        f"{event_response.error or 'API fallback'}"
                    )
                else:
                    st.success(
                        ui_text(
                            lang,
                            f"{product_id}에 대한 {button_label} 피드백을 기록했습니다.",
                            f"Logged {event_type} feedback for {product_id}.",
                        )
                    )


def live_event_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = metrics.get("simulator", {}).get("timeline", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    preferred = [
        "timestamp",
        "event_type",
        "surface",
        "event_role",
        "user_id",
        "session_id",
        "product_id",
        "query",
        "rank",
        "ab_bucket",
        "strategy",
    ]
    columns = [column for column in preferred if column in frame.columns]
    if not columns:
        return frame.copy()
    return frame[columns].copy()


def localized_surface(value: Any, lang: str) -> str:
    text = str(value or "unknown")
    if lang == "ko":
        return SURFACE_LABELS_KO.get(text, text)
    return text.replace("_", " ").title()


def localized_event_type(value: Any, lang: str) -> str:
    text = normalize_event_type(value)
    if lang == "ko":
        return EVENT_LABELS_KO.get(text, text)
    return text.replace("_", " ").title()


def localized_event_color_map(lang: str) -> dict[str, str]:
    return {
        localized_event_type(event_type, lang): color
        for event_type, color in EVENT_COLOR_MAP.items()
    }


def normalize_event_type(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return EVENT_TYPE_ALIASES.get(text, text)


def ordered_event_labels(lang: str) -> list[str]:
    return [localized_event_type(event_type, lang) for event_type in EVENT_FLOW_TYPES]


def event_order(value: Any) -> int:
    event_type = normalize_event_type(value)
    try:
        return EVENT_FLOW_TYPES.index(event_type)
    except ValueError:
        return len(EVENT_FLOW_TYPES)


def complete_live_event_series(series: pd.DataFrame) -> pd.DataFrame:
    if series.empty or not {"minute", "event_type", "count"}.issubset(series.columns):
        return pd.DataFrame()
    frame = series[["minute", "event_type", "count"]].copy()
    frame["minute"] = pd.to_datetime(frame["minute"], errors="coerce", utc=True)
    frame["event_type"] = frame["event_type"].map(normalize_event_type)
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["minute"])
    frame = frame[frame["event_type"].isin(EVENT_FLOW_TYPES)]
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["minute", "event_type"], as_index=False)["count"].sum()
    minutes = sorted(grouped["minute"].unique())
    complete_index = pd.MultiIndex.from_product(
        [minutes, EVENT_FLOW_TYPES],
        names=["minute", "event_type"],
    )
    complete = (
        grouped.set_index(["minute", "event_type"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    complete["event_order"] = complete["event_type"].map(event_order)
    return complete.sort_values(["minute", "event_order"])[["minute", "event_type", "count"]]


def event_plot_frame(event_series: pd.DataFrame, lang: str) -> pd.DataFrame:
    frame = complete_live_event_series(event_series)
    if frame.empty:
        return frame
    labels = ordered_event_labels(lang)
    frame["event_label"] = frame["event_type"].map(lambda value: localized_event_type(value, lang))
    frame["event_label"] = pd.Categorical(frame["event_label"], categories=labels, ordered=True)
    frame["event_order"] = frame["event_type"].map(event_order)
    return frame.sort_values(["minute", "event_order"])


def live_event_trend_figure(event_series: pd.DataFrame, lang: str) -> go.Figure:
    plot_frame = event_plot_frame(event_series, lang)
    labels = ordered_event_labels(lang)
    fig = px.area(
        plot_frame,
        x="minute",
        y="count",
        color="event_label",
        line_group="event_label",
        category_orders={"event_label": labels},
        color_discrete_map=localized_event_color_map(lang),
        labels={
            "minute": ui_text(lang, "시간", "time"),
            "count": ui_text(lang, "이벤트 수", "events"),
            "event_label": ui_text(lang, "이벤트 유형", "event type"),
        },
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        legend=dict(traceorder="normal"),
        xaxis_title=ui_text(lang, "시간", "time"),
        yaxis_title=ui_text(lang, "이벤트 수", "events"),
    )
    return fig


def live_surface_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    live_surfaces = metrics.get("simulator", {}).get("live_surfaces", {})
    if not isinstance(live_surfaces, dict):
        return pd.DataFrame(rows)
    for surface, values in sorted(live_surfaces.items()):
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "surface": surface,
                "impressions": int(values.get("impressions", 0) or 0),
                "clicks": int(values.get("clicks", 0) or 0),
                "carts": int(values.get("carts", 0) or 0),
                "conversions": int(values.get("conversions", 0) or 0),
                "ctr": float(values.get("ctr", 0.0) or 0.0),
                "cart_rate": float(values.get("cart_rate", 0.0) or 0.0),
                "cvr": float(values.get("cvr", 0.0) or 0.0),
                "purchase_per_click": float(values.get("purchase_per_click", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def live_surface_display_frame(frame: pd.DataFrame, lang: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    display["surface"] = display["surface"].map(lambda value: localized_surface(value, lang))
    display["ctr"] = display["ctr"].map(format_pct)
    display["cart_rate"] = display["cart_rate"].map(format_pct)
    display["cvr"] = display["cvr"].map(format_pct)
    display["purchase_per_click"] = display["purchase_per_click"].map(format_pct)
    if lang == "ko":
        display = display.rename(
            columns={
                "surface": "구분",
                "impressions": "노출",
                "clicks": "조회",
                "carts": "장바구니",
                "conversions": "구매",
                "ctr": "CTR",
                "cart_rate": "장바구니율",
                "cvr": "CVR",
                "purchase_per_click": "구매/조회",
            }
        )
    return display


def live_surface_cards_html(frame: pd.DataFrame, lang: str) -> str:
    if frame.empty:
        return ""
    cards = []
    for index, row in frame.iterrows():
        tone = MARS_CHART_COLORS[index % len(MARS_CHART_COLORS)]
        cards.append(
            compact_html(
                f"""
                <div class="mars-live-card" style="--tone:{tone};">
                  <div class="live-card-head">
                    <span>{escape(localized_surface(row.get("surface"), lang))}</span>
                    <i class="bi bi-broadcast-pin"></i>
                  </div>
                  <strong>{format_int(row.get("impressions"))}</strong>
                  <small>{escape(ui_text(lang, "노출", "impressions"))}</small>
                  <div class="live-card-stats">
                    <b>{format_int(row.get("clicks"))}</b>
                    <span>{escape(ui_text(lang, "조회", "views"))}</span>
                    <b>{format_int(row.get("carts"))}</b>
                    <span>{escape(ui_text(lang, "장바구니", "carts"))}</span>
                    <b>{format_int(row.get("conversions"))}</b>
                    <span>{escape(ui_text(lang, "구매", "purchases"))}</span>
                  </div>
                  <div class="live-card-rate">
                    <span>CTR {format_pct(row.get("ctr"))}</span>
                    <span>{escape(ui_text(lang, "장바구니율", "Cart/View"))} {format_pct(row.get("cart_rate"))}</span>
                    <span>CVR {format_pct(row.get("cvr"))}</span>
                  </div>
                </div>
                """
            )
        )
    return f'<div class="mars-live-card-grid">{"".join(cards)}</div>'


def live_event_mix_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    series = live_event_series_frame(metrics)
    if not series.empty:
        mix = series.groupby("event_type", as_index=False)["count"].sum()
        mix["event_order"] = mix["event_type"].map(event_order)
        return mix.sort_values("event_order")[["event_type", "count"]]
    frame = live_event_frame(metrics)
    if frame.empty or "event_type" not in frame.columns:
        return pd.DataFrame()
    event_types = frame.apply(semantic_live_event_type, axis=1)
    event_types = event_types[event_types.isin(EVENT_FLOW_TYPES)]
    if event_types.empty:
        return pd.DataFrame()
    mix = event_types.value_counts().reindex(EVENT_FLOW_TYPES, fill_value=0).reset_index()
    mix.columns = ["event_type", "count"]
    return mix


def model_versions_display_frame(rows: Any, lang: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    def column(name: str, default: str = "") -> pd.Series:
        if name in frame.columns:
            return frame[name]
        return pd.Series([default] * len(frame), index=frame.index)

    def metadata_value(row: pd.Series, key: str, default: str = "") -> str:
        metadata = row.get("metadata", {})
        if isinstance(metadata, dict):
            value = metadata.get(key, default)
            return "" if value is None else str(value)
        return default

    created = pd.to_datetime(column("created_at"), errors="coerce")
    display = pd.DataFrame(
        {
            "version": column("version").astype(str),
            "created_at": created.dt.strftime("%m-%d %H:%M").fillna(""),
            "status": column("status").astype(str),
            "mode": frame.apply(lambda row: metadata_value(row, "mode", "full"), axis=1),
            "artifact": column("artifact_path").map(
                lambda value: compact_runtime_label(value, max_chars=24)
            ),
            "report": column("metrics_path").map(
                lambda value: compact_runtime_label(value, max_chars=24)
            ),
        }
    )
    if lang == "ko":
        display = display.rename(
            columns={
                "version": "버전",
                "created_at": "생성",
                "status": "상태",
                "mode": "모드",
                "artifact": "산출물",
                "report": "리포트",
            }
        )
    return display


def live_event_series_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    minute_rows = metrics.get("simulator", {}).get("minute_timeline", [])
    if isinstance(minute_rows, list) and minute_rows:
        minute_frame = pd.DataFrame(minute_rows)
        if {"minute", "event_type", "count"}.issubset(minute_frame.columns):
            return complete_live_event_series(minute_frame)
    frame = live_event_frame(metrics)
    if frame.empty:
        return pd.DataFrame()
    if {"date", "value"}.issubset(frame.columns):
        series = frame[["date", "value"]].copy()
        series["minute"] = pd.to_datetime(series["date"], errors="coerce")
        series["event_type"] = "event"
        series["count"] = series["value"]
        return complete_live_event_series(series)
    if "timestamp" not in frame.columns:
        return pd.DataFrame()
    timeline = frame.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce", utc=True)
    timeline = timeline.dropna(subset=["timestamp"])
    if timeline.empty:
        return pd.DataFrame()
    timeline["minute"] = timeline["timestamp"].dt.floor("min")
    if "event_type" not in timeline.columns:
        timeline["event_type"] = "event"
    timeline["event_type"] = timeline.apply(semantic_live_event_type, axis=1)
    timeline = timeline.dropna(subset=["event_type"])
    if timeline.empty:
        return pd.DataFrame()
    grouped = (
        timeline.groupby(["minute", "event_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return complete_live_event_series(grouped)


def semantic_live_event_type(row: pd.Series) -> str | None:
    event_type = normalize_event_type(row.get("event_type"))
    if event_type in {"view", "cart", "purchase"}:
        return event_type
    if event_type != "search":
        return None
    event_role = str(row.get("event_role") or "")
    product_id = row.get("product_id")
    rank = row.get("rank")
    if event_role == "user_action" or pd.isna(product_id) or product_id in {"", None}:
        return "search"
    if pd.isna(rank) or rank in {"", None}:
        return "search"
    return None


def event_snapshot_html(event_series: pd.DataFrame, lang: str) -> str:
    if event_series.empty:
        return ""
    latest = pd.to_datetime(event_series["minute"], errors="coerce").max()
    latest_label = (
        latest.strftime("%H:%M") if pd.notna(latest) else ui_text(lang, "현재", "current")
    )
    mix = complete_live_event_series(event_series)
    if not mix.empty:
        mix = mix.groupby("event_type", as_index=False)["count"].sum()
        mix["event_order"] = mix["event_type"].map(event_order)
        mix = mix.sort_values("event_order")
    total = int(mix["count"].sum()) if not mix.empty else 0
    pills = []
    for _, row in mix.iterrows():
        event_type = str(row.get("event_type", "event"))
        label = localized_event_type(event_type, lang)
        color = EVENT_COLOR_MAP.get(event_type, "#377dff")
        pills.append(
            compact_html(
                f"""
                <div class="mars-event-pill" style="--event-color:{color};">
                  <span>{escape(label)}</span>
                  <strong>{format_int(row.get("count"))}</strong>
                </div>
                """
            )
        )
    return compact_html(
        f"""
        <div class="mars-event-snapshot">
          <div>
            <span>{escape(ui_text(lang, "최근 활동 상태", "Current activity"))}</span>
            <strong class="event-total">{format_int(total)}</strong>
            <p>{escape(ui_text(lang, f"{latest_label} 기준 현재 구간의 행동 로그입니다. 시간 구간이 더 쌓이면 추이 차트로 자동 전환됩니다.", f"Current activity at {latest_label}. The panel switches to a trend once more time buckets arrive."))}</p>
          </div>
          <div class="mars-event-pill-grid">{"".join(pills)}</div>
        </div>
        """
    )


def live_event_display_mode(metrics: dict[str, Any]) -> str:
    event_series = live_event_series_frame(metrics)
    if not event_series.empty:
        return "trend" if event_series["minute"].nunique() >= 2 else "snapshot"
    if not live_surface_frame(metrics).empty:
        return "surface"
    if not live_event_mix_frame(metrics).empty:
        return "mix"
    return "empty"


def control_room_event_copy(metrics: dict[str, Any], lang: str) -> tuple[str, str]:
    mode = live_event_display_mode(metrics)
    if mode == "trend":
        return (
            ui_text(lang, "실시간 이벤트 추이", "Live Event Trend"),
            ui_text(
                lang,
                "분 단위로 쌓이는 검색·조회·장바구니·구매 흐름을 누적 면적으로 보여줍니다.",
                "Shows search, view, cart, and purchase flow by event type over time.",
            ),
        )
    if mode == "snapshot":
        return (
            ui_text(lang, "실시간 로그 상태", "Live Log Status"),
            ui_text(
                lang,
                "현재 시간 구간의 행동 로그 유입 여부와 이벤트 구성을 요약합니다. 충분한 시간 구간이 쌓이면 추이 차트로 자동 전환됩니다.",
                "Summarizes current behavior-log activity and event mix. It switches to a trend chart once enough time buckets arrive.",
            ),
        )
    if mode == "surface":
        return (
            ui_text(lang, "화면별 실시간 반응", "Live Response by Surface"),
            ui_text(
                lang,
                "검색 화면과 추천 화면에서 발생한 노출·조회·장바구니·구매를 비교합니다.",
                "Compares impressions, views, carts, and purchases across search and recommendation surfaces.",
            ),
        )
    if mode == "mix":
        return (
            ui_text(lang, "이벤트 타입 구성", "Event Type Mix"),
            ui_text(
                lang,
                "현재 수집된 행동 로그를 이벤트 유형별 비중으로 보여줍니다.",
                "Shows the current behavior logs by event type share.",
            ),
        )
    return (
        ui_text(lang, "실시간 로그 상태", "Live Log Status"),
        ui_text(
            lang,
            "아직 표시할 실시간 행동 로그가 없습니다. 로그가 들어오면 이 영역에 바로 표시됩니다.",
            "No live behavior logs are available yet. This panel updates as logs arrive.",
        ),
    )


def render_live_event_trend(metrics: dict[str, Any], lang: str, *, height: int = 360) -> None:
    event_series = live_event_series_frame(metrics)
    if not event_series.empty:
        if event_series["minute"].nunique() < 2:
            st.markdown(event_snapshot_html(event_series, lang), unsafe_allow_html=True)
            return
        fig = live_event_trend_figure(event_series, lang)
        st.plotly_chart(plotly_clean(fig, height=height), width="stretch")
        return

    surface_frame = live_surface_frame(metrics)
    if not surface_frame.empty:
        chart_frame = surface_frame.melt(
            id_vars=["surface"],
            value_vars=["impressions", "clicks", "carts", "conversions"],
            var_name="metric",
            value_name="count",
        )
        chart_frame["surface"] = chart_frame["surface"].map(
            lambda value: localized_surface(value, lang)
        )
        chart_frame["metric"] = chart_frame["metric"].map(
            lambda value: LIVE_METRIC_LABELS_KO.get(value, value) if lang == "ko" else value
        )
        fig = px.bar(
            chart_frame,
            x="surface",
            y="count",
            color="metric",
            barmode="group",
            color_discrete_sequence=MARS_CHART_COLORS,
            labels={
                "surface": ui_text(lang, "화면", "surface"),
                "count": ui_text(lang, "로그 수", "logs"),
                "metric": ui_text(lang, "지표", "metric"),
            },
        )
        fig.update_layout(
            xaxis_title=ui_text(lang, "구분", "surface"),
            yaxis_title=ui_text(lang, "로그 수", "logs"),
        )
        st.plotly_chart(plotly_clean(fig, height=height), width="stretch")
        return

    event_mix = live_event_mix_frame(metrics)
    if not event_mix.empty:
        event_mix = event_mix.copy()
        event_mix["event_label"] = event_mix["event_type"].map(
            lambda value: localized_event_type(value, lang)
        )
        event_mix["event_label"] = pd.Categorical(
            event_mix["event_label"],
            categories=ordered_event_labels(lang),
            ordered=True,
        )
        fig = px.pie(
            event_mix,
            names="event_label",
            values="count",
            hole=0.5,
            color="event_label",
            category_orders={"event_label": ordered_event_labels(lang)},
            color_discrete_map=localized_event_color_map(lang),
            labels={
                "event_label": ui_text(lang, "이벤트 유형", "event type"),
                "count": ui_text(lang, "이벤트 수", "events"),
            },
        )
        st.plotly_chart(plotly_clean(pie_readable(fig), height=height), width="stretch")
        return

    panel_html(
        ui_text(lang, "실시간 로그 상태", "Live Log Status"),
        f"<p>{escape(ui_text(lang, '아직 표시할 실시간 행동 로그가 없습니다.', 'No live behavior logs are available yet.'))}</p>",
    )


def persona_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    personas = metrics.get("simulator", {}).get("personas", {})
    if not isinstance(personas, dict):
        return pd.DataFrame()
    rows = [
        {"persona": str(name), "share": float(share or 0.0)} for name, share in personas.items()
    ]
    return pd.DataFrame(rows).sort_values("share", ascending=False)


def persona_cards_html(frame: pd.DataFrame, lang: str) -> str:
    if frame.empty:
        return ""
    cards = []
    for index, row in frame.head(6).iterrows():
        tone = MARS_CHART_COLORS[index % len(MARS_CHART_COLORS)]
        persona = str(row.get("persona", "unknown")).replace("_", " ")
        cards.append(
            compact_html(
                f"""
                <div class="mars-persona-chip" style="--tone:{tone};">
                  <span>{escape(persona)}</span>
                  <strong>{format_pct(row.get("share"))}</strong>
                  <small>{escape(ui_text(lang, "시뮬레이터 사용자 비중", "simulated user share"))}</small>
                </div>
                """
            )
        )
    return f'<div class="mars-persona-strip">{"".join(cards)}</div>'


def render_live_event_feed(
    metrics: dict[str, Any],
    *,
    expanded: bool,
    key: str,
    lang: str = "en",
) -> None:
    frame = live_event_frame(metrics)
    with st.expander(ui_text(lang, "라이브 이벤트 피드", "Live event feed"), expanded=expanded):
        st.caption(
            ui_text(
                lang,
                "최근 수집된 행동 로그입니다. 노출, 조회, 장바구니, 구매 반응을 시간순으로 확인할 수 있습니다.",
                "Recent behavior events. Review impressions, views, carts, and purchases in time order.",
            )
        )
        if frame.empty:
            st.info(
                ui_text(
                    lang,
                    "아직 기록된 라이브 API 이벤트가 없습니다.",
                    "No live API events have been recorded yet.",
                )
            )
            return
        st.dataframe(frame, width="stretch", hide_index=True, key=f"{key}_feed_table")


def normalize_ab_report(data: dict[str, Any]) -> pd.DataFrame:
    buckets = data.get("buckets", {})
    if isinstance(buckets, dict):
        rows = [{"bucket": name, **values} for name, values in buckets.items()]
    elif isinstance(buckets, list):
        rows = buckets
    else:
        rows = []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "conversions" not in frame.columns and "purchases" in frame.columns:
        frame["conversions"] = frame["purchases"]
    if "ctr" not in frame.columns:
        clicks = frame["clicks"] if "clicks" in frame.columns else frame.get("conversions", 0)
        frame["ctr"] = clicks / frame["impressions"].replace(0, pd.NA)
    if "cvr" not in frame.columns:
        frame["cvr"] = frame.get("conversions", 0) / frame["impressions"].replace(0, pd.NA)
    return frame.fillna(0)


def page_control_room(
    client: MarsApiClient,
    metrics: dict[str, Any],
    health: DashboardResponse,
    lang: str,
) -> None:
    page_header(tr("nav_control_title", lang), tr("control_subtitle", lang))
    system = metrics.get("system", {})
    search = metrics.get("search", {})
    reco = metrics.get("recommendation", {})
    ab_data = client.ab_report("mars_default").data
    uplift_by_metric = ab_data.get("uplift_by_metric", {}) if isinstance(ab_data, dict) else {}
    cvr_uplift = uplift_by_metric.get(
        "cvr", (ab_data.get("cvr") or {}).get("uplift", ab_data.get("uplift", 0))
    )
    _, _, total_events = system_event_counts(system)

    cards = (
        metric_card_html(
            ui_text(lang, "행동 로그", "Behavior Logs"),
            format_int(total_events),
            system_event_caption(system, lang),
            tone="blue",
            icon="bi-database",
        )
        + metric_card_html(
            ui_text(lang, "검색 NDCG@10", "Search NDCG@10"),
            f"{float(search.get('ndcg_at_10', 0)):.3f}",
            ui_text(lang, "목표 >= 0.50", "target >= 0.50"),
            tone="green",
            icon="bi-search",
        )
        + metric_card_html(
            ui_text(lang, "추천 HitRate@50", "Reco HitRate@50"),
            f"{float(reco.get('hit_rate_at_50', reco.get('hitrate_at_50', 0))):.3f}",
            ui_text(lang, "목표 >= 0.20", "target >= 0.20"),
            tone="purple",
            icon="bi-stars",
        )
        + metric_card_html(
            "CVR Lift",
            format_pct(cvr_uplift),
            treatment_lift_hint(cvr_uplift, lang),
            tone=lift_tone(cvr_uplift),
            icon=lift_icon(cvr_uplift),
        )
        + metric_card_html(
            ui_text(lang, "지속 학습 상태", "CT Status"),
            status_label(metrics.get("training", {}).get("status", "watching"), lang),
            ui_text(lang, "모델 레지스트리 모니터링", "registry monitored"),
            tone="orange",
            icon="bi-arrow-repeat",
        )
    )
    st.markdown(metric_grid_html(cards, columns=5), unsafe_allow_html=True)

    left, right = st.columns([1.45, 0.9])
    with left:
        event_title, event_caption = control_room_event_copy(metrics, lang)
        st.markdown(
            f'<div class="mars-block-label">{escape(event_title)}</div>', unsafe_allow_html=True
        )
        st.caption(event_caption)
        render_live_event_trend(metrics, lang, height=360)
    with right:
        readiness = metrics.get("artifact_readiness", {})
        rows = []
        for name, ready in readiness.items():
            rows.append(
                compact_html(
                    f"""
                    <div>
                      <span>{escape(artifact_label(name, lang))}</span>
                      {badge_html(status_label("ready" if ready else "missing", lang), tone="ok" if ready else "warn", icon="bi-check-circle" if ready else "bi-exclamation-triangle")}
                    </div>
                    """
                )
            )
        services = health.data.get("services", {}) if isinstance(health.data, dict) else {}
        redis_label = services.get("redis", "demo")
        rows.append(
            compact_html(
                f"""
                <div>
                  <span>{escape(ui_text(lang, "Redis 피처 스토어", "Redis Feature Store"))}</span>
                  {badge_html(status_label(redis_label, lang), tone="ok" if redis_label == "ready" else "warn", icon="bi-lightning-charge")}
                </div>
                """
            )
        )
        panel_html(
            ui_text(lang, "서비스 준비 상태", "Service Readiness"),
            f'<div class="mars-info-list">{"".join(rows)}</div>',
            class_name="chart-peer-panel",
        )

    bottom_left, bottom_mid, bottom_right = st.columns(3)
    with bottom_left:
        panel_html(
            ui_text(lang, "검색 성능", "Search Budget"),
            compact_html(
                f"""
                <div class="mars-info-list">
                  <div><span>{escape(ui_text(lang, "p50 지연", "p50 latency"))}</span><b>{format_ms(search.get("latency_p50_ms"))}</b></div>
                  <div><span>{escape(ui_text(lang, "p95 지연", "p95 latency"))}</span><b>{format_ms(search.get("latency_p95_ms"))}</b></div>
                  <div><span>MRR@10</span><b>{float(search.get("mrr_at_10", search.get("mrr", 0))):.3f}</b></div>
                </div>
                """
            ),
            class_name="ops-summary-panel",
        )
    with bottom_mid:
        panel_html(
            ui_text(lang, "추천 파이프라인", "Recommendation Pipeline"),
            compact_html(
                f"""
                <div class="mars-info-list">
                  <div><span>{escape(ui_text(lang, "후보 생성 p95", "candidate p95"))}</span><b>{format_ms(reco.get("candidate_p95_ms"))}</b></div>
                  <div><span>{escape(ui_text(lang, "랭킹 p95", "ranking p95"))}</span><b>{format_ms(reco.get("ranking_p95_ms"))}</b></div>
                  <div><span>{escape(ui_text(lang, "전체 p95", "total p95"))}</span><b>{format_ms(reco.get("total_p95_ms"))}</b></div>
                </div>
                """
            ),
            class_name="ops-summary-panel",
        )
    with bottom_right:
        training = metrics.get("training", {})
        new_logs = float(training.get("new_logs", 0) or 0)
        threshold = max(float(training.get("new_logs_threshold", 1) or 1), 1.0)
        progress = min(new_logs / threshold, 1.0)
        panel_html(
            ui_text(lang, "지속 학습", "Continuous Training"),
            compact_html(
                f"""
                <div class="mars-info-list">
                  <div><span>{escape(ui_text(lang, "신규 로그", "new logs"))}</span><b>{format_int(new_logs)}</b></div>
                  <div><span>{escape(ui_text(lang, "임계값", "threshold"))}</span><b>{format_int(threshold)}</b></div>
                </div>
                <div class="mars-progress"><span style="width:{progress * 100:.0f}%"></span></div>
                """
            ),
            class_name="ops-summary-panel",
        )


def page_experiments(client: MarsApiClient, lang: str) -> None:
    page_header(tr("experiments_title", lang), tr("experiments_subtitle", lang))
    experiment = experiment_key_picker(lang, key="experiments_experiment") or DEFAULT_EXPERIMENT_KEY
    response = client.ab_report(experiment)
    data = response.data
    buckets = normalize_ab_report(data)
    control = (
        buckets[buckets["bucket"].eq("control")].iloc[0].to_dict()
        if not buckets.empty and buckets["bucket"].eq("control").any()
        else {}
    )
    treatment = (
        buckets[buckets["bucket"].eq("treatment")].iloc[0].to_dict()
        if not buckets.empty and buckets["bucket"].eq("treatment").any()
        else {}
    )
    uplift_by_metric = data.get("uplift_by_metric", {}) if isinstance(data, dict) else {}
    ctr_lift = uplift_by_metric.get("ctr", (data.get("ctr") or {}).get("uplift", 0))
    cvr_lift = uplift_by_metric.get(
        "cvr", (data.get("cvr") or {}).get("uplift", data.get("uplift", 0))
    )
    cards = (
        metric_card_html(
            "CTR Lift",
            format_pct(ctr_lift),
            lift_hint(ctr_lift, lang, "클릭률", "click-through rate"),
            tone=lift_tone(ctr_lift),
            icon=lift_icon(ctr_lift),
        )
        + metric_card_html(
            "CVR Lift",
            format_pct(cvr_lift),
            lift_hint(cvr_lift, lang, "전환율", "conversion rate"),
            tone=lift_tone(cvr_lift),
            icon=lift_icon(cvr_lift),
        )
        + metric_card_html(
            "p-value",
            f"{float(data.get('p_value', 0)):.4f}",
            ui_text(lang, "통계 검정", "statistical test"),
            tone="purple",
            icon="bi-calculator",
        )
        + metric_card_html(
            ui_text(lang, "대조군 CVR", "Control CVR"),
            format_pct(control.get("cvr")),
            ui_text(lang, "기준 버킷", "baseline bucket"),
            tone="blue",
            icon="bi-circle-half",
        )
        + metric_card_html(
            ui_text(lang, "실험군 CVR", "Treatment CVR"),
            format_pct(treatment.get("cvr")),
            ui_text(lang, "후보 전략", "candidate strategy"),
            tone="purple",
            icon="bi-stars",
        )
    )
    st.markdown(metric_grid_html(cards, columns=5), unsafe_allow_html=True)

    left, right = st.columns([1.35, 0.9])
    with left:
        if not buckets.empty:
            st.markdown(
                chart_heading_html(
                    ui_text(lang, "노출 비중", "Impression Share"),
                    ui_text(
                        lang,
                        "대조군과 실험군의 유입 규모를 비교합니다.",
                        "Compares control and treatment traffic scale.",
                    ),
                ),
                unsafe_allow_html=True,
            )
            fig = px.pie(
                buckets,
                names="bucket",
                values="impressions",
                hole=0.48,
                color="bucket",
                color_discrete_map={
                    "control": "#377dff",
                    "treatment": "#12b76a",
                },
            )
            st.plotly_chart(plotly_clean(pie_readable(fig), height=360), width="stretch")
        else:
            st.warning(
                ui_text(
                    lang,
                    "이 실험 키에 해당하는 실시간 이벤트가 없습니다.",
                    "No live events were found for this experiment key.",
                )
            )
    with right:
        impressions = int(control.get("impressions", 0) or 0) + int(
            treatment.get("impressions", 0) or 0
        )
        clicks = int(control.get("clicks", 0) or 0) + int(treatment.get("clicks", 0) or 0)
        conversions = int(control.get("conversions", 0) or 0) + int(
            treatment.get("conversions", 0) or 0
        )
        st.markdown(
            '<div class="mars-chart-heading mars-chart-spacer"></div>', unsafe_allow_html=True
        )
        panel_html(
            ui_text(lang, "퍼널 스냅샷", "Funnel Snapshot"),
            compact_html(
                f"""
                <div class="mars-info-list big">
                  <div><span>{escape(ui_text(lang, "노출", "Impressions"))}</span><b>{format_int(impressions)}</b></div>
                  <div><span>{escape(ui_text(lang, "클릭", "Clicks"))}</span><b>{format_int(clicks)}</b></div>
                  <div><span>{escape(ui_text(lang, "구매", "Purchases"))}</span><b>{format_int(conversions)}</b></div>
                  <div><span>95% CI</span><b>{escape(str(data.get("confidence_interval_95", [])))}</b></div>
                </div>
                """
            ),
            class_name="chart-peer-panel",
        )
    if not buckets.empty:
        st.dataframe(buckets, width="stretch", hide_index=True)


def page_model_ops(metrics: dict[str, Any], lang: str) -> None:
    page_header(tr("nav_model_ops_title", lang), tr("model_ops_subtitle", lang))
    stages = [
        (ui_text(lang, "데이터", "Data"), "processed parquet", "bi-database"),
        (ui_text(lang, "검색 인덱스", "Search Index"), "CLIP + FAISS", "bi-search"),
        (ui_text(lang, "후보 생성", "Candidate"), "Two-Tower ANN", "bi-diagram-3"),
        (ui_text(lang, "랭킹", "Ranking"), "Wide&Deep", "bi-sort-down"),
        (ui_text(lang, "재랭킹", "Re-ranking"), "MAB slots", "bi-shuffle"),
        (ui_text(lang, "서빙", "Serving"), "FastAPI + Redis", "bi-lightning-charge"),
    ]
    pipeline = "".join(
        compact_html(
            f"""
            <div class="mars-pipeline-card">
              <i class="bi {icon}"></i>
              <b>{escape(name)}</b>
              <span>{escape(meta)}</span>
              {badge_html(status_label("ready", lang), tone="ok", icon="bi-check-circle")}
            </div>
            """
        )
        for name, meta, icon in stages
    )
    st.markdown(f'<div class="mars-pipeline">{pipeline}</div>', unsafe_allow_html=True)

    system = metrics.get("system", {})
    training = metrics.get("training", {})
    reco = metrics.get("recommendation", {})
    _, live_events, total_events = system_event_counts(system)
    left, mid, right = st.columns(3)
    with left:
        panel_html(
            ui_text(lang, "런타임 데이터", "Runtime Data"),
            compact_html(
                f"""
                <div class="mars-info-list">
                  <div><span>{escape(ui_text(lang, "상품", "Products"))}</span><b>{format_int(system.get("products"))}</b></div>
                  <div><span>{escape(ui_text(lang, "사용자", "Users"))}</span><b>{format_int(system.get("users"))}</b></div>
                  <div><span>{escape(ui_text(lang, "행동 로그", "Behavior Logs"))}</span><b>{format_int(total_events)}</b></div>
                  <div><span>{escape(ui_text(lang, "라이브 증가분", "Live Increment"))}</span><b>{format_int(live_events)}</b></div>
                </div>
                """
            ),
            class_name="ops-summary-panel",
        )
    with mid:
        new_logs = float(training.get("new_logs", 0) or 0)
        threshold = max(float(training.get("new_logs_threshold", 1) or 1), 1.0)
        panel_html(
            ui_text(lang, "재학습 준비도", "Retrain Readiness"),
            compact_html(
                f"""
                <div class="mars-info-list">
                  <div><span>{escape(ui_text(lang, "상태", "Status"))}</span><b>{escape(status_label(training.get("status", "watching"), lang))}</b></div>
                  <div><span>{escape(ui_text(lang, "신규 로그", "New logs"))}</span><b>{format_int(new_logs)}</b></div>
                  <div><span>{escape(ui_text(lang, "임계값", "Threshold"))}</span><b>{format_int(threshold)}</b></div>
                </div>
                <div class="mars-progress"><span style="width:{min(new_logs / threshold, 1.0) * 100:.0f}%"></span></div>
                """
            ),
            class_name="ops-summary-panel",
        )
    with right:
        panel_html(
            ui_text(lang, "지연 시간 예산", "Latency Budget"),
            compact_html(
                f"""
                <div class="mars-info-list">
                  <div><span>{escape(ui_text(lang, "후보 생성 p95", "Candidate p95"))}</span><b>{format_ms(reco.get("candidate_p95_ms"))}</b></div>
                  <div><span>{escape(ui_text(lang, "랭킹 p95", "Ranking p95"))}</span><b>{format_ms(reco.get("ranking_p95_ms"))}</b></div>
                  <div><span>{escape(ui_text(lang, "전체 p95", "Total p95"))}</span><b>{format_ms(reco.get("total_p95_ms"))}</b></div>
                </div>
                """
            ),
            class_name="ops-summary-panel",
        )

    chart_left, chart_right = st.columns([1.05, 0.95])
    with chart_left:
        personas = persona_frame(metrics)
        if not personas.empty:
            st.markdown(
                chart_heading_html(
                    ui_text(lang, "페르소나 분포", "Persona Distribution"),
                    ui_text(
                        lang, "시뮬레이터 사용자 군집 비중입니다.", "Simulator user persona share."
                    ),
                ),
                unsafe_allow_html=True,
            )
            fig = px.pie(
                personas,
                names="persona",
                values="share",
                hole=0.45,
                color_discrete_sequence=MARS_CHART_COLORS,
            )
            st.plotly_chart(
                plotly_clean(pie_readable(fig, textinfo="percent"), height=330),
                width="stretch",
            )
            st.markdown(persona_cards_html(personas, lang), unsafe_allow_html=True)
        else:
            panel_html(
                ui_text(lang, "페르소나 분포", "Persona Distribution"),
                f"<p>{escape(ui_text(lang, '페르소나 집계 데이터가 아직 없습니다.', 'Persona aggregate data is not available yet.'))}</p>",
            )
    with chart_right:
        event_mix = live_event_mix_frame(metrics)
        st.markdown(
            chart_heading_html(
                ui_text(lang, "라이브 행동 믹스", "Live Behavior Mix"),
                ui_text(
                    lang, "현재 수집 로그의 이벤트 유형 비중입니다.", "Current event type share."
                ),
            ),
            unsafe_allow_html=True,
        )
        if not event_mix.empty:
            event_mix = event_mix.copy()
            event_mix["event_label"] = event_mix["event_type"].map(
                lambda value: localized_event_type(value, lang)
            )
            event_mix["event_label"] = pd.Categorical(
                event_mix["event_label"],
                categories=ordered_event_labels(lang),
                ordered=True,
            )
            fig = px.pie(
                event_mix,
                names="event_label",
                values="count",
                hole=0.5,
                color="event_label",
                category_orders={"event_label": ordered_event_labels(lang)},
                color_discrete_map=localized_event_color_map(lang),
                labels={
                    "event_label": ui_text(lang, "이벤트 유형", "event type"),
                    "count": ui_text(lang, "이벤트 수", "events"),
                },
            )
            st.plotly_chart(plotly_clean(pie_readable(fig), height=330), width="stretch")
        else:
            st.info(
                ui_text(lang, "아직 라이브 행동 로그가 없습니다.", "No live behavior events yet.")
            )
    versions = pd.DataFrame(training.get("versions", []))
    st.markdown(
        chart_heading_html(
            ui_text(lang, "모델 버전", "Model Versions"),
            ui_text(
                lang,
                "현재 활성 모델과 이전 산출물 버전을 한 표에서 비교합니다.",
                "Compares the active model and archived artifact versions in one table.",
            ),
        ),
        unsafe_allow_html=True,
    )
    st.dataframe(model_versions_display_frame(versions, lang), width="stretch", hide_index=True)


def page_guide_localized(lang: str) -> None:
    page_header(tr("guide_title", lang), tr("guide_subtitle", lang))
    if lang == "ko":
        cards = [
            {
                "title": "통합 현황",
                "eyebrow": "운영 현황",
                "body": "서비스가 지금 어떤 상태인지 가장 먼저 보는 페이지입니다.",
                "bullets": [
                    "상단 KPI에서 이벤트 규모, 검색 품질, 추천 품질, CVR Lift를 확인합니다.",
                    "실시간 로그 상태와 추이로 최근 행동 로그 흐름이 끊기지 않는지 봅니다.",
                    "서비스 준비 상태에서 데이터, 인덱스, 추천기, Redis 상태를 점검합니다.",
                ],
                "image": "control-room.png",
            },
            {
                "title": "검색 품질",
                "eyebrow": "FAISS 검색",
                "body": "검색 지표와 실제 검색 결과, 클릭 피드백 기록을 한 화면에서 확인합니다.",
                "bullets": [
                    "MRR/NDCG/Recall과 p95 지연 시간이 목표를 넘는지 봅니다.",
                    "텍스트, 이미지, 하이브리드 검색을 직접 호출합니다.",
                    "조회/장바구니/구매 버튼으로 검색 행동 로그를 남길 수 있습니다.",
                ],
                "image": "search.png",
            },
            {
                "title": "추천 파이프라인",
                "eyebrow": "개인화 추천",
                "body": "사용자별 추천 결과와 후보 생성, 랭킹, 재랭킹 지연 시간을 점검합니다.",
                "bullets": [
                    "Recall, AUC, HitRate, Coverage가 목표를 만족하는지 확인합니다.",
                    "기본 A/B 선택지로 버킷별 실시간 추천 전략을 비교합니다.",
                    "추천 상품 이미지와 세션 컨텍스트를 함께 봅니다.",
                ],
                "image": "recommendation.png",
            },
            {
                "title": "실험 분석",
                "eyebrow": "A/B 테스트",
                "body": "대조군과 실험군의 성과 차이가 의미 있는지 확인합니다.",
                "bullets": [
                    "기본 A/B 버튼으로 현재 실험 리포트를 바로 불러옵니다.",
                    "CTR Lift, CVR Lift, p-value, 95% CI를 함께 봅니다.",
                    "버킷별 규모 차트로 노출, 클릭, 구매 규모가 충분한지 확인합니다.",
                ],
                "image": "experiments.png",
            },
            {
                "title": "모델 운영",
                "eyebrow": "운영 상태",
                "body": "데이터부터 서빙까지 파이프라인과 재학습 상태를 점검합니다.",
                "bullets": [
                    "파이프라인 카드에서 데이터, 검색 인덱스, 후보 생성, 랭킹, 서빙 상태를 봅니다.",
                    "재학습 준비도에서 새 로그 수와 재학습 임계값을 확인합니다.",
                    "모델 버전에서 현재 활성 모델과 이전 버전을 비교합니다.",
                ],
                "image": "model-ops.png",
            },
            {
                "title": "라이브 로그",
                "eyebrow": "행동 로그",
                "body": "실서비스처럼 들어오는 행동 로그와 지속 학습 트리거를 관찰합니다.",
                "bullets": [
                    "검색/추천 surface별 노출, 클릭, 전환 규모를 비교합니다.",
                    "이벤트 타입 구성과 분 단위 흐름을 색상별 차트로 확인합니다.",
                    "실시간 새로고침으로 행동 로그가 계속 들어오는지 확인합니다.",
                ],
                "image": "live-logs.png",
            },
            {
                "title": "제출 검증",
                "eyebrow": "요구사항 점검",
                "body": "명세서 요구사항이 현재 실행 환경에서 통과되는지 표로 확인합니다.",
                "bullets": [
                    "PASS/WARN/FAIL 상태를 그룹별로 훑어봅니다.",
                    "근거 컬럼에서 어떤 API, 산출물, 데이터 파일을 봐야 하는지 확인합니다.",
                    "데모 대체 데이터 안내가 뜨면 FastAPI 서버 연결 상태부터 점검합니다.",
                ],
                "image": "qa-gate.png",
            },
        ]
    else:
        cards = [
            {
                "title": "Control Room",
                "eyebrow": "Operations",
                "body": "Start here to understand the current runtime health.",
                "bullets": [
                    "Use the KPI strip for event scale, search quality, recommendation quality, and CVR lift.",
                    "Check Live Event Volume to confirm recent behavior logs are flowing.",
                    "Use Service Readiness for data, index, recommender, and Redis status.",
                ],
                "image": "control-room.png",
            },
            {
                "title": "Search Quality",
                "eyebrow": "FAISS retrieval",
                "body": "Inspect search metrics, live results, and feedback logging in one place.",
                "bullets": [
                    "Check MRR, NDCG, Recall, and p95 latency against the target.",
                    "Run text, image, and hybrid search requests directly.",
                    "Use view/cart/purchase buttons to log search behavior events.",
                ],
                "image": "search.png",
            },
            {
                "title": "Recommendation",
                "eyebrow": "Personalized ranking",
                "body": "Inspect user recommendations and stage latency from candidate to reranking.",
                "bullets": [
                    "Check Recall, AUC, HitRate, and Coverage against the target.",
                    "Use the Default A/B selector to compare live serving behavior.",
                    "Review product images and session context next to the results.",
                ],
                "image": "recommendation.png",
            },
            {
                "title": "Experiments",
                "eyebrow": "A/B testing",
                "body": "Check whether treatment meaningfully improves over control.",
                "bullets": [
                    "Load the current experiment with the Default A/B selector.",
                    "Read CTR Lift, CVR Lift, p-value, and 95% CI together.",
                    "Use Bucket Volume to confirm enough impressions, views, and purchases.",
                ],
                "image": "experiments.png",
            },
            {
                "title": "Model Ops",
                "eyebrow": "Model operations",
                "body": "Inspect the pipeline and continuous training state from data to serving.",
                "bullets": [
                    "Check Data, Search Index, Candidate, Ranking, and Serving pipeline cards.",
                    "Use Retrain Readiness to compare new logs against the threshold.",
                    "Review Model Versions for the active and archived models.",
                ],
                "image": "model-ops.png",
            },
            {
                "title": "Live Logs",
                "eyebrow": "Behavior stream",
                "body": "Watch live behavior events and continuous-training trigger readiness.",
                "bullets": [
                    "Compare impressions, views, carts, and purchases by search/recommendation surface.",
                    "Use colored charts for event type mix and minute-by-minute flow.",
                    "Turn on live refresh to verify behavior-log traffic.",
                ],
                "image": "live-logs.png",
            },
            {
                "title": "QA Gate",
                "eyebrow": "Submission check",
                "body": "Validate whether the spec requirements pass in the current runtime.",
                "bullets": [
                    "Scan PASS/WARN/FAIL by group.",
                    "Use the where column to locate the API, artifact, or data file behind each row.",
                    "If the demo fallback warning appears, check the FastAPI connection first.",
                ],
                "image": "qa-gate.png",
            },
        ]
    guide_html = "".join(
        guide_card_html(
            title=item["title"],
            eyebrow=item["eyebrow"],
            body=item["body"],
            bullets=item["bullets"],
            image_name=item["image"],
        )
        for item in cards
    )
    st.markdown(f'<div class="mars-guide-grid">{guide_html}</div>', unsafe_allow_html=True)


def page_search(client: MarsApiClient, metrics: dict[str, Any], lang: str) -> None:
    st.markdown(
        f'<div class="mars-section-title">{escape(ui_text(lang, "검색 품질", "Search Quality"))}</div>',
        unsafe_allow_html=True,
    )
    search_metrics = metrics.get("search", {})
    if search_metrics.get("quality_status") == "fail":
        st.warning(
            ui_text(
                lang,
                "검색 품질이 요구 목표 중 하나 이상을 만족하지 못했습니다. 최종 제출 전 MRR, NDCG, p95 지연 시간을 확인하세요.",
                "Search quality is below at least one required target. Check MRR, NDCG, and p95 latency before final submission.",
            )
        )
    quality_cards = (
        metric_card_html(
            "MRR@10",
            f"{float(search_metrics.get('mrr_at_10', search_metrics.get('mrr', 0))):.3f}",
            ui_text(lang, "목표 >= 0.55", "target >= 0.55"),
            tone="purple",
            icon="bi-bullseye",
        )
        + metric_card_html(
            "NDCG@10",
            f"{float(search_metrics.get('ndcg_at_10', 0)):.3f}",
            ui_text(lang, "목표 >= 0.50", "target >= 0.50"),
            tone="green",
            icon="bi-bar-chart-line",
        )
        + metric_card_html(
            "Recall@10",
            f"{float(search_metrics.get('recall_at_10', 0)):.3f}",
            ui_text(lang, "정답 포함률", "relevant item found"),
            tone="blue",
            icon="bi-check2-circle",
        )
        + metric_card_html(
            "p50",
            format_ms(search_metrics.get("latency_p50_ms")),
            ui_text(lang, "중앙 지연", "median latency"),
            tone="orange",
            icon="bi-speedometer",
        )
        + metric_card_html(
            "p95",
            format_ms(search_metrics.get("latency_p95_ms")),
            ui_text(lang, "목표 <= 200 ms", "target <= 200 ms"),
            tone="blue",
            icon="bi-lightning-charge",
        )
    )
    st.markdown(metric_grid_html(quality_cards, columns=5), unsafe_allow_html=True)
    detail_cards = (
        metric_card_html(
            ui_text(lang, "카테고리 적중@10", "Category Hit@10"),
            f"{float(search_metrics.get('category_hit_at_10', 0)):.3f}",
            ui_text(lang, "쿼리 카테고리 일치", "query category match"),
            tone="green",
            icon="bi-tags",
        )
        + metric_card_html(
            ui_text(lang, "평가 질의 수", "Evaluated Queries"),
            format_int(search_metrics.get("evaluated_queries")),
            ui_text(lang, "test split 기준", "held-out test split"),
            tone="blue",
            icon="bi-list-check",
        )
        + metric_card_html(
            ui_text(lang, "샘플 수", "Sample Size"),
            format_int(search_metrics.get("prediction_sample_size")),
            ui_text(lang, "평가 표본", "evaluation sample"),
            tone="purple",
            icon="bi-database-check",
        )
    )
    st.markdown(
        metric_grid_html(detail_cards, columns=3, class_name="compact"), unsafe_allow_html=True
    )
    st.caption(
        ui_text(
            lang,
            f"오프라인 소스: {search_metrics.get('source', 'unknown')}. 기본 평가: {search_metrics.get('primary_evaluation', 'unknown')}. MRR/NDCG는 정확한 상품 ID 기준이며, 카테고리 적중@10은 결과가 쿼리 카테고리와 맞는지 확인합니다.",
            f"Offline source: {search_metrics.get('source', 'unknown')}. Primary evaluation: {search_metrics.get('primary_evaluation', 'unknown')}. MRR/NDCG are exact product-id metrics; Category Hit@10 checks whether results match the query category.",
        )
    )
    production_metrics = search_metrics.get("production_with_qrels_prior") or {}
    if production_metrics:
        with st.expander(ui_text(lang, "평가 상세", "Evaluation detail"), expanded=False):
            st.caption(
                ui_text(
                    lang,
                    "아래 값은 분리된 qrels test split에서 계산됩니다. qrels train split은 supervised query behavior feature 생성에만 사용됩니다.",
                    "These values are computed on the held-out qrels test split. The qrels train split is used only for supervised query behavior features.",
                )
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "MRR@10": production_metrics.get("mrr_at_10"),
                            "NDCG@10": production_metrics.get("ndcg_at_10"),
                            "Recall@10": production_metrics.get("recall_at_10"),
                            "p95_ms": production_metrics.get("latency_p95_ms"),
                            "source": production_metrics.get("source"),
                        }
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    query_cols = st.columns([2, 1, 1])
    query = query_cols[0].text_input(ui_text(lang, "검색어", "Search query"), value="black socks")
    search_type_labels = {
        ui_text(lang, "텍스트", "text"): "text",
        ui_text(lang, "이미지", "image"): "image",
        ui_text(lang, "하이브리드", "hybrid"): "hybrid",
    }
    selected_search_type = query_cols[1].selectbox(
        ui_text(lang, "검색 방식", "Search type"),
        list(search_type_labels.keys()),
        index=0,
    )
    search_type = search_type_labels[str(selected_search_type)]
    top_k = query_cols[2].slider("Top K", min_value=5, max_value=50, value=10, step=5)
    image_url = None
    if search_type in {"image", "hybrid"}:
        image_url = st.text_input(
            ui_text(lang, "이미지 경로 또는 URL", "Image path or URL"), value=SAMPLE_IMAGE_URL
        )
    response = client.search(
        query=query,
        search_type=search_type,
        top_k=top_k,
        image_url=image_url,
    )
    source_badge(response, lang=lang)

    debug = response.data.get("debug", {})
    live_cards = (
        metric_card_html(
            ui_text(lang, "최근 질의 지연", "Last Query Latency"),
            format_ms(response.data.get("latency_ms")),
            ui_text(lang, "현재 API 응답", "live API response"),
            tone="blue",
            icon="bi-stopwatch",
        )
        + metric_card_html(
            ui_text(lang, "반환 결과", "Returned"),
            format_int(response.data.get("total_count")),
            ui_text(lang, "현재 top-k 후보", "current top-k candidates"),
            tone="green",
            icon="bi-box-seam",
        )
        + metric_card_html(
            ui_text(lang, "인덱스", "Index"),
            compact_runtime_label(debug.get("index_backend", "unknown")),
            ui_text(lang, "검색 백엔드", "retrieval backend"),
            tone="purple",
            icon="bi-hdd-network",
        )
        + metric_card_html(
            ui_text(lang, "인코더", "Encoder"),
            compact_runtime_label(debug.get("encoder_type", "unknown")),
            ui_text(lang, "질의 임베딩", "query embedding"),
            tone="orange",
            icon="bi-cpu",
        )
    )
    st.markdown(
        metric_grid_html(live_cards, columns=4, class_name="compact"), unsafe_allow_html=True
    )
    st.caption(
        ui_text(
            lang,
            "p50/p95는 오프라인 리포트 지표이고, 최근 질의 지연은 현재 API 응답 시간입니다.",
            "p50/p95 are offline report metrics; Last Query Latency is the live API response time.",
        )
    )

    rows = response.data.get("results", [])
    product_grid(
        rows,
        empty=ui_text(lang, "반환된 검색 결과가 없습니다.", "No search results returned."),
        lang=lang,
    )
    with st.expander(
        ui_text(lang, "검색 결과 원본 테이블", "Raw search result table"), expanded=False
    ):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    render_search_feedback_controls(client, rows=rows, query=query, debug=debug, lang=lang)
    render_product_preview(rows, key="search", lang=lang)


def page_recommendation(client: MarsApiClient, metrics: dict[str, Any], lang: str) -> None:
    st.markdown(
        f'<div class="mars-section-title">{escape(ui_text(lang, "추천 파이프라인", "Recommendation Pipeline"))}</div>',
        unsafe_allow_html=True,
    )
    reco = metrics.get("recommendation", {})
    quality_cards = (
        metric_card_html(
            "Recall@300",
            f"{float(reco.get('recall_at_300', 0)):.3f}",
            ui_text(lang, "목표 >= 0.30", "target >= 0.30"),
            tone="purple",
            icon="bi-diagram-3",
        )
        + metric_card_html(
            "AUC",
            f"{float(reco.get('auc', 0)):.3f}",
            ui_text(lang, "목표 >= 0.70", "target >= 0.70"),
            tone="blue",
            icon="bi-activity",
        )
        + metric_card_html(
            "HitRate@50",
            f"{float(reco.get('hit_rate_at_50', reco.get('hitrate_at_50', 0))):.3f}",
            ui_text(lang, "목표 >= 0.20", "target >= 0.20"),
            tone="green",
            icon="bi-bullseye",
        )
        + metric_card_html(
            "NDCG@50",
            f"{float(reco.get('ndcg_at_50', 0)):.3f}",
            ui_text(lang, "목표 >= 0.08", "target >= 0.08"),
            tone="orange",
            icon="bi-bar-chart",
        )
        + metric_card_html(
            ui_text(lang, "커버리지", "Coverage"),
            f"{float(reco.get('coverage', 0)):.3f}",
            ui_text(lang, "목표 >= 0.20", "target >= 0.20"),
            tone="purple",
            icon="bi-grid-3x3-gap",
        )
        + metric_card_html(
            "p95 Total",
            format_ms(reco.get("total_p95_ms")),
            ui_text(lang, "전체 서빙 지연", "total serving latency"),
            tone="blue",
            icon="bi-speedometer2",
        )
    )
    st.markdown(metric_grid_html(quality_cards, columns=6), unsafe_allow_html=True)
    st.caption(
        ui_text(
            lang,
            f"오프라인 평가 소스: {reco.get('source', 'unknown')}, 평가 사용자 수: {format_int(reco.get('evaluated_users'))}. p95 Total은 밀리초 단위이며, 실시간 요청 지연은 아래에 별도로 표시됩니다.",
            f"Offline evaluation source: {reco.get('source', 'unknown')} over {format_int(reco.get('evaluated_users'))} sampled users. p95 Total is milliseconds; live request latency is shown below.",
        )
    )

    user_cols = st.columns([1.7, 0.8, 1, 1.7])
    user_id = user_cols[0].text_input(ui_text(lang, "사용자 ID", "User ID"), value="U000020")
    top_n = user_cols[1].slider("Top N", min_value=5, max_value=50, value=10, step=5)
    session_id = user_cols[2].text_input(ui_text(lang, "세션 ID", "Session ID"), value="S-demo")
    with user_cols[3]:
        experiment_key = experiment_key_picker(
            lang,
            key="recommendation_experiment",
            allow_disabled=True,
        )
    st.caption(
        ui_text(
            lang,
            "기존 개발 사용자 예시는 U000020, U000038, U000052, U005256입니다. 알 수 없는 사용자는 인기 기반 콜드스타트 추천으로 대체됩니다. 기본 A/B를 선택하면 사용자 ID 기준으로 A/B 버킷과 서빙 전략이 자동 배정됩니다.",
            "Try existing dev users such as U000020, U000038, U000052, U005256. Unknown users fall back to popularity-based cold-start recommendations. Default A/B assigns the bucket and serving strategy by user ID.",
        )
    )
    response = client.recommend(
        user_id=user_id,
        top_n=top_n,
        session_id=session_id,
        experiment_key=experiment_key,
    )
    source_badge(response, lang=lang)

    latency = response.data.get("pipeline_latency", {})
    latency_cards = (
        metric_card_html(
            ui_text(lang, "실시간 전체", "Live Total"),
            format_ms(latency.get("total_ms")),
            ui_text(lang, "API 요청 기준", "live request"),
            tone="blue",
            icon="bi-stopwatch",
        )
        + metric_card_html(
            ui_text(lang, "후보 생성", "Candidate"),
            format_ms(latency.get("candidate_ms")),
            ui_text(lang, "후보 검색 단계", "candidate stage"),
            tone="purple",
            icon="bi-diagram-3",
        )
        + metric_card_html(
            ui_text(lang, "랭킹", "Ranking"),
            format_ms(latency.get("ranking_ms")),
            ui_text(lang, "랭커 추론", "ranker inference"),
            tone="green",
            icon="bi-sort-down",
        )
        + metric_card_html(
            ui_text(lang, "재랭킹", "Reranking"),
            format_ms(latency.get("reranking_ms")),
            ui_text(lang, "탐색/재정렬", "exploration rerank"),
            tone="orange",
            icon="bi-shuffle",
        )
    )
    st.markdown(
        metric_grid_html(latency_cards, columns=4, class_name="compact"), unsafe_allow_html=True
    )
    latency_df = pd.DataFrame(
        [
            {
                "stage": ui_text(lang, "후보 생성", "candidate"),
                "latency_ms": latency.get("candidate_ms", 0),
            },
            {"stage": ui_text(lang, "랭킹", "ranking"), "latency_ms": latency.get("ranking_ms", 0)},
            {
                "stage": ui_text(lang, "재랭킹", "reranking"),
                "latency_ms": latency.get("reranking_ms", 0),
            },
        ]
    )
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            chart_heading_html(
                ui_text(lang, "단계별 지연 시간", "Stage Latency"),
                ui_text(
                    lang,
                    "추천 요청이 후보 생성, 랭킹, 재랭킹에서 소비한 시간입니다.",
                    "Latency spent by each recommendation stage.",
                ),
            ),
            unsafe_allow_html=True,
        )
        fig = px.bar(
            latency_df,
            x="stage",
            y="latency_ms",
            color="stage",
            color_discrete_sequence=MARS_CHART_COLORS,
            labels={
                "stage": ui_text(lang, "단계", "stage"),
                "latency_ms": ui_text(lang, "지연 시간(ms)", "latency (ms)"),
            },
        )
        st.plotly_chart(plotly_clean(fig, height=320), width="stretch")
    with right:
        st.markdown(
            chart_heading_html(
                ui_text(lang, "세션 컨텍스트", "Session Context"),
                ui_text(
                    lang,
                    "현재 추천 요청에 반영된 사용자·세션 신호입니다.",
                    "User and session signals used for this recommendation request.",
                ),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            compact_html(
                f"""
                <div class="mars-panel context-body-panel">
                  {session_context_html(response.data.get("session_context", {}), lang=lang)}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    rec_rows = recommendation_table_frame(response.data.get("recommendations", []), lang=lang)
    product_grid(
        response.data.get("recommendations", []),
        empty=ui_text(lang, "반환된 추천 결과가 없습니다.", "No recommendations returned."),
        lang=lang,
    )
    if not rec_rows.empty:
        preferred = [
            "product_id",
            "name",
            "category",
            "price",
            "score",
            "reason",
            "is_exploration",
        ]
        rec_rows = rec_rows[[column for column in preferred if column in rec_rows.columns]]
    with st.expander(
        ui_text(lang, "추천 결과 원본 테이블", "Raw recommendation table"), expanded=False
    ):
        st.dataframe(rec_rows, width="stretch", hide_index=True)
    render_product_preview(
        response.data.get("recommendations", []), key="recommendation", lang=lang
    )


def page_requirement_check(
    client: MarsApiClient,
    metrics: dict[str, Any],
    health: DashboardResponse,
    lang: str,
) -> None:
    st.markdown(
        f'<div class="mars-section-title">{escape(tr("qa_title", lang))}</div>',
        unsafe_allow_html=True,
    )
    st.caption(tr("qa_caption", lang))

    system = metrics.get("system", {})
    search = metrics.get("search", {})
    reco = metrics.get("recommendation", {})
    data_quality = metrics.get("data_quality", {})
    training = metrics.get("training", {})
    services = health.data.get("services", {}) if isinstance(health.data, dict) else {}
    artifacts = metrics.get("artifact_readiness", {})
    mode = metrics.get("mode") or system.get("mode", "unknown")
    full_mode = mode == "full"
    if health.is_demo or health.data.get("status") == "demo" or mode == "demo":
        st.warning(tr("qa_fallback_warning", lang))

    text_search = client.search(query="black jacket", search_type="text", top_k=3).data
    image_search = client.search(search_type="image", top_k=3, image_url=SAMPLE_IMAGE_URL).data
    hybrid_search = client.search(
        query="sport sweatshirt",
        search_type="hybrid",
        top_k=3,
        image_url=SAMPLE_IMAGE_URL,
    ).data
    recommendation = client.recommend(
        user_id="U000020", top_n=5, session_id="S-requirement-check"
    ).data
    ab_report = client.ab_report("mars_default").data

    required_search_fields = {"search_type", "results", "latency_ms", "total_count"}
    required_search_result_fields = {"product_id", "name", "score", "price"}
    search_result = (text_search.get("results") or [{}])[0]
    required_reco_fields = {"user_id", "recommendations", "pipeline_latency", "session_context"}
    required_reco_result_fields = {"product_id", "score", "reason", "is_exploration"}
    reco_result = (recommendation.get("recommendations") or [{}])[0]
    latency = recommendation.get("pipeline_latency", {})

    rows: list[dict[str, Any]] = []

    def add(group: str, item: str, current: Any, target: str, status: str, where: str) -> None:
        rows.append(
            {
                "group": REQ_GROUPS_KO.get(group, group) if lang == "ko" else group,
                "item": REQ_ITEMS_KO.get(item, item) if lang == "ko" else item,
                "current": current,
                "target": target,
                "status": status,
                "where": where,
            }
        )

    add(
        "Runtime",
        "API health",
        health.data.get("status"),
        "ok",
        status_text(health.data.get("status") == "ok"),
        "/healthz",
    )
    add(
        "Runtime",
        "Redis feature store",
        services.get("redis"),
        "ready",
        status_text(services.get("redis") == "ready"),
        "/healthz",
    )
    add(
        "Runtime",
        "API and dashboard containers",
        "dashboard rendering",
        "docker compose up",
        "PASS",
        "docker compose ps",
    )
    add(
        "Runtime",
        "Artifact readiness",
        f"{sum(1 for ready in artifacts.values() if ready)}/{len(artifacts)} ready",
        "all ready",
        status_text(bool(artifacts) and all(bool(value) for value in artifacts.values())),
        "/api/metrics",
    )

    products = int(system.get("products", 0) or 0)
    users = int(system.get("users", 0) or 0)
    events = int(system.get("events", 0) or 0)
    add(
        "Simulator",
        "Products",
        format_int(products),
        "50,000 full / 5,000 dev",
        status_text(products >= 50000, warn=products >= 5000 and not full_mode),
        "data/processed/products.parquet",
    )
    add(
        "Simulator",
        "Users",
        format_int(users),
        "10,000 full / 1,000 dev",
        status_text(users >= 10000, warn=users >= 1000 and not full_mode),
        "data/processed/users.parquet",
    )
    add(
        "Simulator",
        "Events",
        format_int(events),
        "1,000,000 full / 50,000 dev",
        status_text(events >= 1000000, warn=events >= 50000 and not full_mode),
        "data/processed/events.parquet",
    )
    add(
        "Simulator",
        "6 personas",
        ", ".join(data_quality.get("persona_names", [])),
        "6 persona types",
        status_text(int(data_quality.get("persona_count", 0) or 0) >= 6),
        "config + users.parquet",
    )
    add(
        "Simulator",
        "Event types",
        ", ".join(data_quality.get("event_types", [])),
        "search/view/cart/purchase",
        status_text(
            {"search", "view", "cart", "purchase"}.issubset(
                set(data_quality.get("event_types", []))
            )
        ),
        "events.parquet",
    )
    add(
        "Simulator",
        "Train/valid/test split",
        str(data_quality.get("split", {})),
        "0.8 / 0.1 / 0.1",
        status_text(bool(data_quality.get("split_counts"))),
        "manifest.json",
    )

    add(
        "Search API",
        "Required response fields",
        ", ".join(text_search.keys()),
        ", ".join(sorted(required_search_fields)),
        status_text(
            required_search_fields.issubset(text_search.keys())
            and required_search_result_fields.issubset(search_result.keys())
        ),
        "POST /api/search",
    )
    add(
        "Search API",
        "Text search",
        format_ms(text_search.get("latency_ms")),
        "<= 200 ms",
        target_status(text_search.get("latency_ms"), 200, higher_is_better=False),
        "POST /api/search",
    )
    add(
        "Search API",
        "Image search",
        format_ms(image_search.get("latency_ms")),
        "<= 200 ms",
        target_status(image_search.get("latency_ms"), 200, higher_is_better=False),
        "POST /api/search",
    )
    add(
        "Search API",
        "Hybrid search",
        format_ms(hybrid_search.get("latency_ms")),
        "works",
        status_text(bool(hybrid_search.get("results"))),
        "POST /api/search",
    )
    search_mrr = search.get("mrr", search.get("mrr_at_10"))
    add(
        "Search Quality",
        "MRR@10",
        f"{float(search_mrr or 0):.3f}",
        ">= 0.55",
        target_status(search_mrr, 0.55),
        "artifacts/reports/metrics.json",
    )
    add(
        "Search Quality",
        "NDCG@10",
        f"{float(search.get('ndcg_at_10', 0)):.3f}",
        ">= 0.50",
        target_status(search.get("ndcg_at_10"), 0.50),
        "artifacts/reports/metrics.json",
    )
    add(
        "Search Quality",
        "Search quality status",
        f"{search.get('quality_status', 'unknown')} / {search.get('primary_evaluation', search.get('source', 'unknown'))}",
        "PASS when MRR, NDCG, and latency targets are all satisfied",
        status_text(search.get("quality_status") != "fail"),
        "evaluation.search_primary",
    )

    add(
        "Recommendation API",
        "Required response fields",
        ", ".join(recommendation.keys()),
        ", ".join(sorted(required_reco_fields)),
        status_text(
            required_reco_fields.issubset(recommendation.keys())
            and required_reco_result_fields.issubset(reco_result.keys())
        ),
        "GET /api/recommend",
    )
    add(
        "Recommendation API",
        "Candidate latency",
        format_ms(latency.get("candidate_ms")),
        "<= 100 ms target",
        target_status(latency.get("candidate_ms"), 100, higher_is_better=False),
        "pipeline_latency",
    )
    add(
        "Recommendation API",
        "Total latency",
        format_ms(latency.get("total_ms")),
        "<= 200 ms",
        target_status(latency.get("total_ms"), 200, higher_is_better=False),
        "pipeline_latency",
    )
    add(
        "Recommendation API",
        "MAB exploration slot",
        str(any(item.get("is_exploration") for item in recommendation.get("recommendations", []))),
        "at least 1 true",
        status_text(
            any(item.get("is_exploration") for item in recommendation.get("recommendations", []))
        ),
        "recommendations[].is_exploration",
    )
    add(
        "Recommendation Quality",
        "Recall@300",
        f"{float(reco.get('recall_at_300', 0)):.3f}",
        ">= 0.30",
        target_status(reco.get("recall_at_300"), 0.30),
        "metrics.json",
    )
    add(
        "Recommendation Quality",
        "AUC",
        f"{float(reco.get('auc', 0)):.3f}",
        ">= 0.70",
        target_status(reco.get("auc"), 0.70),
        "metrics.json",
    )
    add(
        "Recommendation Quality",
        "HitRate@50",
        f"{float(reco.get('hit_rate_at_50', reco.get('hitrate_at_50', 0))):.3f}",
        ">= 0.20",
        target_status(reco.get("hit_rate_at_50", reco.get("hitrate_at_50")), 0.20),
        "metrics.json",
    )
    add(
        "Recommendation Quality",
        "NDCG@50",
        f"{float(reco.get('ndcg_at_50', 0)):.3f}",
        ">= 0.08",
        target_status(reco.get("ndcg_at_50"), 0.08),
        "metrics.json",
    )
    reco_coverage = reco.get("coverage", reco.get("coverage_at_50"))
    add(
        "Recommendation Quality",
        "Coverage@50",
        f"{float(reco_coverage or 0):.3f}",
        ">= 0.20",
        target_status(reco_coverage, 0.20),
        "metrics.json",
    )

    add(
        "Feature Store",
        "Redis lookup latency",
        format_ms(system.get("redis_latency_ms")),
        "<= 10 ms",
        target_status(system.get("redis_latency_ms"), 10, higher_is_better=False),
        "/api/metrics",
    )
    ab_buckets = ab_report.get("buckets", {}) if isinstance(ab_report, dict) else {}
    ab_control = ab_buckets.get("control", {}) if isinstance(ab_buckets, dict) else {}
    ab_treatment = ab_buckets.get("treatment", {}) if isinstance(ab_buckets, dict) else {}
    add(
        "A/B Testing",
        "CTR by bucket",
        f"control={format_pct(ab_control.get('ctr'))}, treatment={format_pct(ab_treatment.get('ctr'))}",
        "CTR visible for both buckets",
        status_text("ctr" in ab_control and "ctr" in ab_treatment),
        "/api/ab/report",
    )
    add(
        "A/B Testing",
        "CVR by bucket",
        f"control={format_pct(ab_control.get('cvr'))}, treatment={format_pct(ab_treatment.get('cvr'))}",
        "CVR visible for both buckets",
        status_text("cvr" in ab_control and "cvr" in ab_treatment),
        "/api/ab/report",
    )
    add(
        "A/B Testing",
        "p-value and 95% CI",
        f"p={ab_report.get('p_value')} ci={ab_report.get('confidence_interval_95')}",
        "present",
        status_text("p_value" in ab_report and "confidence_interval_95" in ab_report),
        "/api/ab/report",
    )
    add(
        "Continuous Training",
        "Live CTR/CVR",
        f"CTR={format_pct(training.get('ctr'))}, CVR={format_pct(training.get('cvr'))}",
        "live monitoring rates visible",
        status_text("ctr" in training and "cvr" in training),
        "/api/metrics.training",
    )
    add(
        "Continuous Training",
        "Retrain decision",
        training.get("next_action", "unknown"),
        "threshold logic visible",
        status_text("should_retrain" in training),
        "/api/metrics",
    )
    add(
        "Continuous Training",
        "Model registry",
        format_int(len(training.get("versions", []))),
        ">= 1 version",
        status_text(len(training.get("versions", [])) >= 1),
        "artifacts/registry/models.json",
    )

    status_dataframe(rows, lang=lang)
    st.caption(tr("qa_footer", lang))


def render_training_panel(metrics: dict[str, Any], *, show_feed: bool, lang: str = "en") -> None:
    training = metrics.get("training", {})
    new_logs = float(training.get("new_logs", 0) or 0)
    threshold = max(float(training.get("new_logs_threshold", 1) or 1), 1.0)
    training_rate_source = training.get("cvr_source", training.get("ctr_source", "unknown"))
    training_rate_source_label = compact_source_label(training_rate_source, lang=lang)
    ctr_rate_source_label = compact_source_label(
        training.get("ctr_source", training_rate_source), lang=lang
    )
    log_source_label = display_log_source_label(training.get("log_source", "unknown"), lang=lang)
    cards = (
        metric_card_html(
            ui_text(lang, "상태", "Status"),
            status_label(training.get("status", "unknown"), lang),
            ui_text(lang, "행동 로그 모니터링", "behavior log monitor"),
            tone="blue",
            icon="bi-eye",
        )
        + metric_card_html(
            ui_text(lang, "신규 로그", "New logs"),
            format_int(new_logs),
            ui_text(lang, f"목표 {format_int(threshold)}", f"target {format_int(threshold)}"),
            tone="green",
            icon="bi-database-add",
        )
        + metric_card_html(
            "CTR",
            format_pct(training.get("ctr")),
            ui_text(
                lang,
                f"임계값 {format_pct(training.get('ctr_threshold'))}",
                f"threshold {format_pct(training.get('ctr_threshold'))}",
            ),
            tone="purple",
            icon="bi-cursor",
        )
        + metric_card_html(
            "CVR",
            format_pct(training.get("cvr")),
            ui_text(
                lang,
                f"소스 {training_rate_source_label}",
                f"source {training_rate_source_label}",
            ),
            tone="orange",
            icon="bi-bag-check",
        )
        + metric_card_html(
            "HitRate",
            format_pct(training.get("hit_rate")),
            ui_text(
                lang,
                f"임계값 {format_pct(training.get('hitrate_threshold'))}",
                f"threshold {format_pct(training.get('hitrate_threshold'))}",
            ),
            tone="green",
            icon="bi-bullseye",
        )
        + metric_card_html(
            ui_text(lang, "재학습 상태", "Retrain Status"),
            retrain_trigger_label(training, lang),
            retrain_trigger_hint(training, lang),
            tone="orange"
            if training.get("retrain_trigger_active", training.get("should_retrain", False))
            else "blue",
            icon="bi-arrow-repeat",
        )
    )
    st.markdown(metric_grid_html(cards, columns=6), unsafe_allow_html=True)
    progress = min(
        1.0,
        new_logs / threshold,
    )
    st.caption(
        ui_text(
            lang,
            f"지속 학습 모니터: {log_source_label} | 현재 로그 {format_int(training.get('current_log_count'))} | 마지막 체크포인트 {format_int(training.get('last_log_count'))} | CTR/CVR 소스 {ctr_rate_source_label}",
            f"CT monitor: {log_source_label} | current logs {format_int(training.get('current_log_count'))} | last checkpoint {format_int(training.get('last_log_count'))} | CTR/CVR source {ctr_rate_source_label}",
        )
    )
    st.progress(progress, text=training_action_label(training.get("next_action"), lang))
    if training.get("reasons"):
        st.warning(
            ui_text(lang, "재학습 사유: ", "Retrain reasons: ")
            + ", ".join(map(str, training.get("reasons", [])))
        )
    if training.get("alert_active"):
        st.info(
            ui_text(lang, "지표 알림: ", "Metric alerts: ")
            + ", ".join(map(str, training.get("alert_reasons", [])))
        )
    surface_frame = live_surface_frame(metrics)
    if not surface_frame.empty:
        st.markdown(
            compact_html(
                f"""
                <div class="mars-live-hero">
                  <span><i class="bi bi-broadcast-pin"></i>{escape(ui_text(lang, "실시간 행동 로그", "Live behavior logs"))}</span>
                  <h3>{escape(ui_text(lang, "검색/추천 화면별 반응을 바로 확인합니다.", "Monitor search and recommendation response by surface."))}</h3>
                  <p>{escape(ui_text(lang, "시뮬레이터와 대시보드 피드백 버튼이 남긴 노출, 클릭, 장바구니, 구매 이벤트를 집계합니다.", "Aggregates exposure, click, cart, and purchase events from the simulator and dashboard feedback controls."))}</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        st.markdown(live_surface_cards_html(surface_frame, lang), unsafe_allow_html=True)
        st.caption(
            ui_text(
                lang,
                "실시간 API와 시뮬레이터가 남긴 행동 로그를 검색/추천 화면별로 집계한 값입니다.",
                "Behavior events from the live API and simulator, grouped by search/recommendation surface.",
            )
        )
        chart_frame = surface_frame.melt(
            id_vars=["surface"],
            value_vars=["impressions", "clicks", "carts", "conversions"],
            var_name="metric",
            value_name="count",
        )
        chart_frame["surface_label"] = chart_frame["surface"].map(
            lambda value: localized_surface(value, lang)
        )
        chart_frame["metric_label"] = chart_frame["metric"].map(
            lambda value: LIVE_METRIC_LABELS_KO.get(value, value) if lang == "ko" else value
        )
        chart_left, chart_right = st.columns([1.25, 0.85])
        with chart_left:
            st.markdown(
                chart_heading_html(
                    ui_text(lang, "화면별 노출/조회/장바구니/구매", "Surface Response Volume"),
                    ui_text(
                        lang,
                        "검색과 추천 화면의 행동 로그 규모를 비교합니다.",
                        "Compares behavior volume across search and recommendation surfaces.",
                    ),
                ),
                unsafe_allow_html=True,
            )
            fig = px.bar(
                chart_frame,
                x="surface_label",
                y="count",
                color="metric_label",
                barmode="group",
                color_discrete_sequence=MARS_CHART_COLORS,
                labels={
                    "surface_label": ui_text(lang, "화면", "surface"),
                    "count": ui_text(lang, "로그 수", "logs"),
                    "metric_label": ui_text(lang, "지표", "metric"),
                },
            )
            st.plotly_chart(plotly_clean(fig, height=340), width="stretch")
        with chart_right:
            event_mix = live_event_mix_frame(metrics)
            if not event_mix.empty:
                st.markdown(
                    chart_heading_html(
                        ui_text(lang, "이벤트 타입 구성", "Event Type Mix"),
                        ui_text(
                            lang,
                            "현재 수집 로그의 이벤트 유형 비중입니다.",
                            "Current event type share.",
                        ),
                    ),
                    unsafe_allow_html=True,
                )
                event_mix = event_mix.copy()
                event_mix["event_label"] = event_mix["event_type"].map(
                    lambda value: localized_event_type(value, lang)
                )
                event_mix["event_label"] = pd.Categorical(
                    event_mix["event_label"],
                    categories=ordered_event_labels(lang),
                    ordered=True,
                )
                fig = px.pie(
                    event_mix,
                    names="event_label",
                    values="count",
                    hole=0.52,
                    color="event_label",
                    category_orders={"event_label": ordered_event_labels(lang)},
                    color_discrete_map=localized_event_color_map(lang),
                    labels={
                        "event_label": ui_text(lang, "이벤트 유형", "event type"),
                        "count": ui_text(lang, "이벤트 수", "events"),
                    },
                )
                st.plotly_chart(plotly_clean(pie_readable(fig), height=340), width="stretch")
        event_series = live_event_series_frame(metrics)
        if not event_series.empty:
            if event_series["minute"].nunique() < 2:
                st.markdown(
                    f'<div class="mars-block-label">{escape(ui_text(lang, "최근 활동 상태", "Current Activity"))}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    ui_text(
                        lang,
                        "현재 분 구간의 이벤트만 있어 추이 차트 대신 활동 상태를 표시합니다.",
                        "Only the current minute is available, so the dashboard shows activity status instead of a trend chart.",
                    )
                )
                st.markdown(event_snapshot_html(event_series, lang), unsafe_allow_html=True)
            else:
                st.markdown(
                    chart_heading_html(
                        ui_text(lang, "분 단위 라이브 이벤트 흐름", "Live Event Flow by Minute"),
                        ui_text(
                            lang,
                            "최근 분 단위 이벤트 수를 검색, 조회, 장바구니, 구매 순서의 누적 면적으로 표시합니다.",
                            "Minute-level event counts as a stacked area ordered by search, view, cart, and purchase.",
                        ),
                    ),
                    unsafe_allow_html=True,
                )
                fig = live_event_trend_figure(event_series, lang)
                st.plotly_chart(plotly_clean(fig, height=300), width="stretch")
        with st.expander(
            ui_text(lang, "화면 집계 원본 표", "Raw surface aggregate table"), expanded=False
        ):
            st.dataframe(
                live_surface_display_frame(surface_frame, lang),
                width="stretch",
                hide_index=True,
            )
    render_live_event_feed(metrics, expanded=show_feed, key="ct", lang=lang)
    st.markdown(f"#### {ui_text(lang, '모델 버전', 'Model Versions')}")
    st.dataframe(
        model_versions_display_frame(training.get("versions", []), lang),
        width="stretch",
        hide_index=True,
    )


@st.fragment(run_every="5s")
def live_training_panel(client: MarsApiClient, *, show_feed: bool, lang: str = "en") -> None:
    metrics_response = client.metrics()
    if metrics_response.is_demo:
        source_badge(metrics_response, lang=lang)
    render_training_panel(metrics_response.data, show_feed=show_feed, lang=lang)


def page_training(client: MarsApiClient, metrics: dict[str, Any], lang: str = "en") -> None:
    st.markdown(
        f'<div class="mars-section-title">{escape(ui_text(lang, "라이브 로그와 지속 학습", "Live Logs & Continuous Training"))}</div>',
        unsafe_allow_html=True,
    )
    controls = st.columns([1, 1, 1, 2])
    auto_refresh = controls[0].toggle(
        ui_text(lang, "실시간 새로고침", "Live refresh"), value=False, key="ct_live_refresh"
    )
    show_feed = controls[1].toggle(
        ui_text(lang, "라이브 피드 보기", "Show live feed"), value=False, key="ct_show_live_feed"
    )
    if controls[2].button(
        ui_text(lang, "라이브 실행 초기화", "Reset live run"),
        key="ct_reset_live_run",
        use_container_width=True,
    ):
        reset_response = client.reset_live_run()
        if reset_response.is_demo:
            st.warning(reset_response.error or "Could not reset live run.")
        else:
            data = reset_response.data
            st.success(
                ui_text(
                    lang,
                    f"Live log archived. Previous lines: {format_int(data.get('previous_lines'))}",
                    f"Live log archived. Previous lines: {format_int(data.get('previous_lines'))}",
                )
            )
            st.rerun()
    controls[3].caption(
        ui_text(
            lang,
            "실시간 새로고침은 이 패널만 5초마다 갱신합니다. Reset은 현재 라이브 로그를 archive로 옮기고 새 실행을 시작합니다.",
            "Live refresh updates only the CT panel every 5 seconds. Reset archives the current log and starts a new live run.",
        )
    )
    if auto_refresh:
        live_training_panel(client, show_feed=show_feed, lang=lang)
    else:
        render_training_panel(metrics, show_feed=show_feed, lang=lang)


def main() -> None:
    st.set_page_config(page_title="MARS Dashboard", page_icon="M", layout="wide")
    inject_css()

    active_slug = active_nav_slug()
    lang = active_language()
    section = NAV_BY_SLUG[active_slug]
    render_sidebar_nav(active_slug, lang)
    default_api_url = (
        os.getenv("MARS_API_BASE_URL") or os.getenv("API_BASE_URL") or "http://localhost:8000"
    )
    api_url = render_sidebar_controls(active_slug, lang, default_api_url)
    client = MarsApiClient(base_url=api_url, timeout=12.0)

    health = client.health()
    metrics_response = client.metrics()
    metrics = metrics_response.data

    render_sidebar_connection_status(health, metrics_response, lang)
    page_title = next(
        (tr(title_key, lang) for slug, _, title_key, _, _ in NAV_ITEMS if slug == active_slug),
        section,
    )
    topbar(
        section=page_title,
        health=health,
        metrics_response=metrics_response,
        metrics=metrics,
        lang=lang,
    )

    if section == "Control Room":
        page_control_room(client, metrics, health, lang)
    elif section == "Search":
        page_search(client, metrics, lang)
    elif section == "Recommendation":
        page_recommendation(client, metrics, lang)
    elif section == "Experiments":
        page_experiments(client, lang)
    elif section == "Model Ops":
        page_model_ops(metrics, lang)
    elif section == "Continuous Training":
        page_training(client, metrics, lang)
    elif section == "QA Gate":
        page_requirement_check(client, metrics, health, lang)
    else:
        page_guide_localized(lang)


if __name__ == "__main__":
    main()
