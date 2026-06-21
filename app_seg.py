import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates
import threading, os

def _start_ngrok():
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = "3FM0hUKdnzBXmYxM1kVfWbrNQsJ_3VUcesyw3kdQd4uAHA1Gc"
    tunnel = ngrok.connect(8501)
    print("\n" + "="*50)
    print(f"  공유 링크: {tunnel.public_url}")
    print("="*50 + "\n")

if not os.environ.get("_NGROK_STARTED"):
    os.environ["_NGROK_STARTED"] = "1"
    threading.Thread(target=_start_ngrok, daemon=True).start()

from logic import evaluate_crack
from color_utils import draw_severity_box, severity_to_hex, get_severity_gradient_legend
from crack_measure import (
    crack_width_px_percentile,
    crack_width_with_location,
    detect_coin_diameter_px,
    px_to_mm,
    two_points_to_diameter_px,
    COIN_DIAMETER_MM,
)

MODEL_PATH = "runs/segment/crack_seg_v1-4/weights/best.pt"

st.set_page_config(page_title="균열 탐지 시스템 (Segment)", layout="wide")
st.title("🔍 콘크리트 균열 탐지 및 안전 등급 평가 — Segment 모델")
st.caption("YOLOv8-seg 기반 균열 자동 탐지 + 마스크 기반 정밀 폭 측정 + 국토안전관리원 기준 등급 판정")

# ── 사이드바 설정 ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 분석 설정")
    location = st.selectbox("부재 종류", ["기둥", "보", "슬래브", "벽", "바닥"])
    damage_type = st.selectbox("손상 유형", ["균열", "처짐", "박리"])
    conf_threshold = st.slider("탐지 신뢰도 임계값", min_value=0.05, max_value=0.9,
                               value=0.1, step=0.05)
    st.markdown("---")
    st.info("📏 균열 옆에 **100원 동전**(지름 24mm)을 놓고, 정면에서 30~50cm 거리로 촬영해주세요.")
    st.caption("⚠️ 본 측정값은 참고용 추정치이며, 정밀 진단 시 크랙게이지를 통한 직접 측정을 병행해야 합니다.")
    st.markdown("---")
    st.markdown(get_severity_gradient_legend())

# ── 이미지 업로드 ──────────────────────────────────────
uploaded = st.file_uploader("이미지 업로드 (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("왼쪽에서 설정 후, 동전과 함께 촬영한 균열 이미지를 업로드하세요.")
    st.stop()

# 이미지 로드
pil_img = Image.open(uploaded).convert("RGB")
img_np = np.array(pil_img)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# ── YOLO 추론 (segment) ────────────────────────────────
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

with st.spinner("균열 탐지 중..."):
    results = model(img_bgr, conf=conf_threshold)[0]

boxes = results.boxes
masks = results.masks
num_cracks = len(boxes) if boxes is not None else 0

# ── 동전 자동 탐지 (px → mm 환산 기준) ──────────────────
if "coin_pts" not in st.session_state:
    st.session_state.coin_pts = []

coin_result = detect_coin_diameter_px(img_bgr)
coin_diameter_px = None
coin_center_for_draw = None
manual_mode = False

if coin_result is not None:
    coin_diameter_px = coin_result["diameter_px"]
    coin_center_for_draw = coin_result["center"]
    st.success(f"✅ 동전 자동 탐지 성공 — 지름 {coin_diameter_px:.1f}px → {COIN_DIAMETER_MM}mm 기준 적용")
else:
    manual_mode = True
    st.warning(
        "⚠️ 동전을 자동으로 찾지 못했습니다. 아래 이미지에서 "
        "**동전의 양 끝 2곳을 순서대로 클릭**해주세요."
    )

# ── 수동 보정 모드: 사용자 2점 클릭 ─────────────────────
if manual_mode:
    display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    click = streamlit_image_coordinates(Image.fromarray(display_rgb), key="coin_click")

    if click is not None:
        pt = (click["x"], click["y"])
        if len(st.session_state.coin_pts) < 2:
            if not st.session_state.coin_pts or st.session_state.coin_pts[-1] != pt:
                st.session_state.coin_pts.append(pt)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.caption(f"클릭한 점: {len(st.session_state.coin_pts)}/2")
    with col_b:
        if st.button("🔄 클릭 초기화"):
            st.session_state.coin_pts = []
            st.rerun()

    if len(st.session_state.coin_pts) == 2:
        coin_diameter_px = two_points_to_diameter_px(
            st.session_state.coin_pts[0], st.session_state.coin_pts[1]
        )
        st.success(f"✅ 수동 측정 완료 — 지름 {coin_diameter_px:.1f}px → {COIN_DIAMETER_MM}mm 기준 적용")
    else:
        st.info("동전 양 끝 2점을 클릭하면 측정이 진행됩니다.")
        st.stop()

# ── 균열 폭 계산 (마스크 기반 정밀 측정) ─────────────────
crack_widths_mm = []
crack_width_lines = []  # 각 균열별 실측 폭을 나타내는 (p1, p2) 좌표 — bbox 대신 이걸 그림
if num_cracks > 0 and masks is not None and coin_diameter_px:
    for mask_data in masks.data:
        mask_np = mask_data.cpu().numpy()
        mask_resized = cv2.resize(
            mask_np, (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        loc = crack_width_with_location(mask_resized, percentile=95.0)
        if loc:
            crack_widths_mm.append(px_to_mm(loc["width_px"], coin_diameter_px))
            crack_width_lines.append((loc["p1"], loc["p2"]))
        else:
            crack_widths_mm.append(0.0)
            crack_width_lines.append(None)
elif num_cracks > 0 and masks is None:
    # 마스크 없으면 bbox 단축변으로 fallback
    st.warning("마스크를 찾을 수 없어 바운딩박스 기준으로 폭을 추정합니다.")
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width_px = min(x2 - x1, y2 - y1)
        if coin_diameter_px:
            crack_widths_mm.append(px_to_mm(width_px, coin_diameter_px))
        crack_width_lines.append(None)

# 대표 균열폭: 최댓값 (보수적 평가)
if crack_widths_mm:
    crack_width = max(crack_widths_mm)
elif num_cracks == 0:
    crack_width = 0.0
    st.warning(f"탐지된 균열이 없습니다. 신뢰도 임계값({conf_threshold})을 더 낮춰보세요.")
else:
    crack_width = 0.0

# ── 등급 판정 ──────────────────────────────────────────
eval_result = evaluate_crack(crack_width, location, damage_type)
grade       = eval_result["grade"]
grade_desc  = eval_result["grade_desc"]
risk_label  = eval_result["risk_label"]
action      = eval_result["action"]
severity    = eval_result["severity_score"]
hex_color   = severity_to_hex(severity)
methods     = eval_result["methods"]

# ── 마스크 오버레이 + 박스 + 동전 그리기 ─────────────────
annotated = img_bgr.copy()

if num_cracks > 0 and masks is not None:
    overlay = annotated.copy()
    for i, mask_data in enumerate(masks.data):
        mask_np = mask_data.cpu().numpy()
        mask_resized = cv2.resize(
            mask_np, (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        from color_utils import severity_to_color
        color = severity_to_color(severity)
        overlay[mask_resized] = color
    annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)

if num_cracks > 0:
    for i, box in enumerate(boxes):
        coords = box.xyxy[0].tolist()
        conf   = float(box.conf[0])
        w_label = f"{crack_widths_mm[i]:.2f}mm" if i < len(crack_widths_mm) else "?"
        lbl    = f"{grade} {w_label} ({conf:.0%})"

        # bbox(전체를 감싸는 큰 사각형)는 더 이상 그리지 않는다 —
        # 균열처럼 가늘고 긴 형태에서는 bbox 두께가 실제 폭과 무관하게 항상 두껍게 나와
        # "이 사각형이 균열 폭이다"라는 오해를 유발하기 때문.
        # 대신: ① 마스크 외곽선만 얇게 그리고, ② 실제 폭을 측정한 지점에
        # 그 폭만큼의 짧은 선분을 직접 그려서 "여기서 이만큼 측정했다"를 보여준다.

        # ① 마스크 윤곽선 (해당 균열의 마스크만)
        if masks is not None and i < len(masks.data):
            mask_np_i = masks.data[i].cpu().numpy()
            mask_resized_i = cv2.resize(
                mask_np_i, (img_bgr.shape[1], img_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            contours, _ = cv2.findContours(mask_resized_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (0, 0, 255), 1)

        # ② 실측 폭 선분 (crack_width_with_location이 계산한 두 점)
        if i < len(crack_width_lines) and crack_width_lines[i] is not None:
            p1, p2 = crack_width_lines[i]
            cv2.line(annotated, p1, p2, (0, 0, 255), 2)
            # 선분 양 끝에 작은 캡(측정 지점임을 명확히 표시)
            cv2.circle(annotated, p1, 3, (0, 0, 255), -1)
            cv2.circle(annotated, p2, 3, (0, 0, 255), -1)
            label_pos = (min(p1[0], p2[0]), min(p1[1], p2[1]) - 8)
        else:
            x1, y1, _, _ = coords
            label_pos = (int(x1), int(y1) - 8)

        cv2.putText(
            annotated, lbl, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
        )

if coin_center_for_draw:
    cv2.circle(annotated, coin_center_for_draw, int(coin_result["radius_px"]), (255, 200, 0), 2)
    cv2.putText(annotated, "coin (24mm)",
                (coin_center_for_draw[0] - 30, coin_center_for_draw[1] - int(coin_result["radius_px"]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
elif manual_mode and len(st.session_state.coin_pts) == 2:
    cv2.line(annotated, st.session_state.coin_pts[0], st.session_state.coin_pts[1], (255, 200, 0), 2)

annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ── 레이아웃 출력 ──────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("탐지 결과 이미지 (마스크 오버레이)")
    st.image(annotated_rgb, use_container_width=True)
    st.caption(f"탐지된 균열 수: **{num_cracks}개** | 신뢰도 임계값: {conf_threshold}")

with col2:
    st.subheader("안전 등급 판정")

    st.markdown(
        f"""
        <div style='background:{hex_color}22; border-left:6px solid {hex_color};
                    padding:16px; border-radius:8px; margin-bottom:12px'>
            <h2 style='margin:0; color:{hex_color}'>{grade.upper()}</h2>
            <p style='margin:4px 0 0 0'>{grade_desc}</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(f"**위험도:** {risk_label}")
    st.markdown(f"**조치사항:** {action}")
    st.caption("⚠️ 참고용 추정치입니다. 정밀 진단 시 크랙게이지 병행 측정을 권장합니다.")

    st.markdown("---")
    st.subheader("보수·보강 공법")
    for m in methods:
        with st.expander(f"{m['순위']} — {m['공법']}"):
            st.write(f"**설명:** {m['설명']}")
            st.write(f"**개략공사비:** {m['개략공사비']}")
            st.write(f"**시공성:** {m['시공성']}")
            st.write(f"**장비:** {m['장비']}")

# ── 요약 지표 ──────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("탐지 균열 수", f"{num_cracks}개")
m2.metric("추정 균열폭 (마스크)", f"{crack_width:.2f} mm")
m3.metric("상태 등급", grade)
m4.metric("위험도", eval_result["risk"])