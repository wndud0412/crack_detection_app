"""
균열 폭 자동 측정 모듈

처리 순서 (보고서 4.5절 설계 + segment 모델 기반 정밀화):
  1. segment 마스크 → distance transform → 균열 폭(px) 계산
  2. 사진 속 동전(100원, 지름 24mm) → Hough Circle로 자동 탐지 → 지름(px) 계산
  3. mm_per_pixel = 24.0 / 동전_지름_px
  4. 균열폭_mm = 균열폭_px * mm_per_pixel

동전 자동탐지가 실패하면 None을 반환하여, 호출 측(app.py)에서
사용자가 동전 양 끝 2점을 직접 클릭하는 수동 보정 모드로 전환하도록 한다.
"""

import cv2
import numpy as np

COIN_DIAMETER_MM = 24.0  # 100원 동전 실측 지름 (보고서 4.5절 고정값)


# ──────────────────────────────────────────────────────────
# 1. 균열 폭(px) 계산 — segment 마스크 기반
# ──────────────────────────────────────────────────────────
def crack_width_px_from_mask(mask: np.ndarray) -> float | None:
    """
    YOLO segment 마스크(0/1 또는 0/255, 단일 채널)로부터
    균열의 최대 두께(px)를 distance transform으로 계산한다.

    원리:
      - 마스크 내부 각 픽셀에서 "가장 가까운 배경(마스크 밖)까지의 거리"를 계산
      - 균열처럼 가늘고 긴 형태에서는, 중심선 위 픽셀의 거리값 × 2 가 그 지점의 두께
      - 마스크 전체에서 이 값의 최댓값을 균열의 대표 폭으로 사용
        (균열 폭은 보통 위치마다 다르므로, 최댓값을 쓰면 가장 심한 지점을 보수적으로 평가)

    Returns:
        최대 두께(px). 마스크가 비어있으면 None.
    """
    if mask is None:
        return None

    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255

    if cv2.countNonZero(m) == 0:
        return None

    # distance transform: 각 전경 픽셀 → 가장 가까운 배경까지 거리
    dist = cv2.distanceTransform(m, cv2.DIST_L2, maskSize=5)
    max_dist = float(dist.max())

    if max_dist <= 0:
        return None

    # 거리값은 "중심에서 가장자리까지"이므로, 두께(폭)는 2배
    width_px = max_dist * 2.0
    return width_px


def crack_width_px_percentile(mask: np.ndarray, percentile: float = 95.0) -> float | None:
    """
    최댓값 대신 상위 percentile 값을 쓰는 버전 (노이즈/라벨 끝부분 이상치에 덜 민감).
    기본 95퍼센타일을 사용하여 극단적 튐 값을 완화한다.
    """
    if mask is None:
        return None

    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255

    if cv2.countNonZero(m) == 0:
        return None

    dist = cv2.distanceTransform(m, cv2.DIST_L2, maskSize=5)
    nonzero_dist = dist[dist > 0]
    if nonzero_dist.size == 0:
        return None

    p_dist = float(np.percentile(nonzero_dist, percentile))
    return p_dist * 2.0


def crack_width_with_location(mask: np.ndarray, percentile: float = 95.0) -> dict | None:
    """
    균열 폭(px)과 함께, 그 폭이 측정된 실제 위치 정보까지 반환한다.

    distance transform 값이 (percentile 기준으로) 가장 큰 픽셀을 "측정 중심점"으로 잡고,
    그 지점에서 균열의 국소 방향에 "수직"인 방향으로 실제 두께만큼의 선분 양 끝점을 계산한다.
    이렇게 하면 bbox 같은 큰 사각형이 아니라, 균열 위에 짧은 선 하나로
    "여기서 이만큼 폭을 측정했다"를 정확히 시각화할 수 있다.

    원리:
      1. distance transform으로 각 픽셀의 중심거리(반두께) 계산
      2. percentile 값에 가장 가까운 distance를 갖는 전경 픽셀을 측정점으로 선택
         (여러 개면 그중 distance가 가장 큰 점 = 균열이 가장 두꺼운 대표 지점)
      3. 그 점 주변 작은 윈도우에서 마스크의 주성분(PCA)으로 균열의 국소 진행 방향을 추정
      4. 진행 방향에 수직인 방향으로 ±(measured_width/2)만큼 이동한 두 점을 선분 끝점으로 계산

    Returns:
        {
            "width_px": float,           # 측정된 폭(px) = percentile 기준 두께
            "center": (x, y),            # 측정 중심점 (이미지 좌표, 정수)
            "p1": (x1, y1),               # 폭을 나타내는 선분의 시작점
            "p2": (x2, y2),               # 폭을 나타내는 선분의 끝점
        }
        측정 불가 시 None.
    """
    if mask is None:
        return None

    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255

    if cv2.countNonZero(m) == 0:
        return None

    dist = cv2.distanceTransform(m, cv2.DIST_L2, maskSize=5)
    nonzero_mask = dist > 0
    if not np.any(nonzero_mask):
        return None

    nonzero_dist = dist[nonzero_mask]
    target_radius = float(np.percentile(nonzero_dist, percentile))
    width_px = target_radius * 2.0

    # percentile 반지름값에 가장 가까운 픽셀들을 찾고, 그중 하나를 측정 중심점으로 선택
    diff = np.abs(dist - target_radius)
    diff[~nonzero_mask] = np.inf
    cy, cx = np.unravel_index(np.argmin(diff), diff.shape)
    center = (int(cx), int(cy))

    # 측정 중심점 주변 로컬 윈도우에서 균열의 진행 방향(주성분) 추정
    win = max(int(target_radius * 4), 9)
    y0, y1 = max(0, cy - win), min(m.shape[0], cy + win + 1)
    x0, x1 = max(0, cx - win), min(m.shape[1], cx + win + 1)
    local = m[y0:y1, x0:x1]

    ys, xs = np.nonzero(local)
    if len(xs) >= 2:
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        pts -= pts.mean(axis=0)
        # 공분산 행렬의 주고유벡터 = 균열이 뻗어나가는 방향
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]  # 진행 방향 단위벡터
        # 진행 방향에 수직인 방향 = 폭 방향
        normal = np.array([-direction[1], direction[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len
        else:
            normal = np.array([0.0, 1.0])
    else:
        # 윈도우 안에 픽셀이 거의 없으면 수직 방향을 기본값으로 사용
        normal = np.array([0.0, 1.0])

    half = target_radius
    p1 = (int(round(cx - normal[0] * half)), int(round(cy - normal[1] * half)))
    p2 = (int(round(cx + normal[0] * half)), int(round(cy + normal[1] * half)))

    return {
        "width_px": width_px,
        "center": center,
        "p1": p1,
        "p2": p2,
    }


# ──────────────────────────────────────────────────────────
# 2. 동전 자동 탐지 — Hough Circle Transform
# ──────────────────────────────────────────────────────────
def detect_coin_diameter_px(image_bgr: np.ndarray) -> dict | None:
    """
    이미지에서 동전(원형 객체)을 자동 탐지하여 지름(px)을 반환한다.

    Hough Circle은 조명/배경에 민감하므로, 여러 파라미터 조합을 시도하고
    가장 그럴듯한(이미지 대비 적당한 크기의) 원 하나를 선택한다.
    실패 시 None을 반환 — 호출 측에서 수동 클릭 모드로 전환해야 함.

    Returns:
        성공 시 {"center": (x, y), "radius_px": r, "diameter_px": 2r}
        실패 시 None
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape[:2]
    img_min_dim = min(h, w)

    # 동전이 이미지에서 차지할 법한 반지름 범위를 이미지 크기 비례로 추정
    min_r = max(8, int(img_min_dim * 0.01))
    max_r = int(img_min_dim * 0.25)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=img_min_dim // 4,
        param1=100,
        param2=40,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)

    # 여러 원이 잡히면, 가장 원형도가 높고 적당한 크기인 것을 선택
    # (단순화: 첫 번째로 검출된 원 중 가장 신뢰도 높은(=가장 먼저 반환된) 원 사용)
    x, y, r = circles[0]

    return {
        "center": (int(x), int(y)),
        "radius_px": float(r),
        "diameter_px": float(r) * 2.0,
    }


# ──────────────────────────────────────────────────────────
# 3. px → mm 환산
# ──────────────────────────────────────────────────────────
def px_to_mm(width_px: float, coin_diameter_px: float,
             coin_real_mm: float = COIN_DIAMETER_MM) -> float:
    """
    보고서 4.5절 코드 6과 동일한 환산식.
    mm_per_pixel = 동전 실측 지름(mm) / 동전 픽셀 지름(px)
    """
    if coin_diameter_px <= 0:
        raise ValueError("동전 지름(px)은 0보다 커야 합니다.")
    mm_per_pixel = coin_real_mm / coin_diameter_px
    return round(width_px * mm_per_pixel, 3)


def two_points_to_diameter_px(p1: tuple, p2: tuple) -> float:
    """사용자가 클릭한 동전 양 끝 2점 사이의 거리(px)를 계산."""
    (x1, y1), (x2, y2) = p1, p2
    return float(np.hypot(x2 - x1, y2 - y1))