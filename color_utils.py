import cv2
import numpy as np

def severity_to_color(severity_score):
    """
    severity_score: 1(경미) ~ 5(심각)
    반환: BGR 색상 (OpenCV용)

    1 → 초록색 (양호)
    2 → 연두색
    3 → 노란색 (주의)
    4 → 주황색
    5 → 빨간색 (위험)
    """
    color_map = {
        1: (0, 200, 0),      # 진한 초록
        2: (0, 220, 120),    # 연두
        3: (0, 220, 220),    # 노랑
        4: (0, 140, 255),    # 주황
        5: (0, 0, 255),      # 빨강
    }
    return color_map.get(severity_score, (0, 220, 220))


def severity_to_hex(severity_score):
    """
    Streamlit UI용 HEX 색상 코드
    """
    hex_map = {
        1: "#00C800",   # 초록
        2: "#00DC78",   # 연두
        3: "#FFDC00",   # 노랑
        4: "#FF8C00",   # 주황
        5: "#FF0000",   # 빨강
    }
    return hex_map.get(severity_score, "#FFDC00")


def draw_severity_box(image, box_coords, severity_score, label=""):
    """
    이미지에 심각도별 색상으로 박스를 그려줌
    image: numpy array (원본 이미지)
    box_coords: (x1, y1, x2, y2)
    severity_score: 1~5
    """
    x1, y1, x2, y2 = [int(v) for v in box_coords]
    color = severity_to_color(severity_score)

    # 박스 그리기 (두께 3)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    # 라벨 배경 + 텍스트
    if label:
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            color, -1
        )
        cv2.putText(
            image, label, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2
        )

    return image


def get_severity_gradient_legend():
    """
    UI에 표시할 범례 텍스트
    """
    return """
    🟢 1단계 (a등급) — 양호
    🟢 2단계 (b등급) — 보통
    🟡 3단계 (c등급) — 주의
    🟠 4단계 (d등급) — 불량
    🔴 5단계 (e등급) — 위험
    """