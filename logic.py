def evaluate_crack(crack_width_mm, location, damage_type="균열"):
    """
    crack_width_mm: 균열 폭 (mm, 참고용 추정치)
    location: 부재 종류 ("기둥", "보", "슬래브", "벽", "바닥")
    damage_type: 손상 유형 ("균열", "처짐", "박리")
    """

    # ─────────────────────────────────────
    # ① 상태 등급 판정 (국토안전관리원 세부지침 기준 — 참고용)
    # ─────────────────────────────────────
    if crack_width_mm < 0.2:
        grade = "a등급"
        grade_desc = "양호 — 결함 없음 또는 경미한 손상"
        severity_score = 1   # 색상 계산용 (1~5)
    elif crack_width_mm < 0.3:
        grade = "b등급"
        grade_desc = "보통 — 경미한 결함"
        severity_score = 2
    elif crack_width_mm < 1.0:
        grade = "c등급"
        grade_desc = "주의 — 보수 검토 필요"
        severity_score = 3
    elif crack_width_mm < 2.0:
        grade = "d등급"
        grade_desc = "불량 — 즉시 보수 필요"
        severity_score = 4
    else:
        grade = "e등급"
        grade_desc = "위험 — 즉각 사용 중단 검토"
        severity_score = 5

    # ─────────────────────────────────────
    # ② 위험도 평가 (부재 위치 + 등급 조합)
    # ─────────────────────────────────────
    structural = ["기둥", "보", "슬래브"]

    if location in structural and grade in ["d등급", "e등급"]:
        risk = "HIGH"
        risk_label = "🔴 HIGH"
        risk_detail = "구조 부재의 심각한 손상 — 붕괴 위험 가능성"
        action = "즉시 전문가 정밀안전진단 의뢰 및 사용 제한 검토"
    elif location in structural and grade == "c등급":
        risk = "MEDIUM-HIGH"
        risk_label = "🟠 MEDIUM-HIGH"
        risk_detail = "구조 부재의 주의 수준 손상"
        action = "빠른 시일 내 보수 및 전문가 검토 필요"
    elif location in structural:
        risk = "MEDIUM"
        risk_label = "🟡 MEDIUM"
        risk_detail = "구조 부재이나 경미한 수준"
        action = "정기 관찰 및 진행 여부 모니터링"
    elif grade in ["d등급", "e등급"]:
        risk = "MEDIUM-HIGH"
        risk_label = "🟠 MEDIUM-HIGH"
        risk_detail = "비구조 부재이나 심각한 손상"
        action = "보수 후 원인 파악 필요 (누수·철근부식 가능성 확인)"
    elif grade == "c등급":
        risk = "MEDIUM"
        risk_label = "🟡 MEDIUM"
        risk_detail = "비구조 부재의 주의 수준 손상"
        action = "보수 후 지속 관찰 필요"
    else:
        risk = "LOW"
        risk_label = "🟢 LOW"
        risk_detail = "경미한 수준"
        action = "정기 관찰 유지"

    # ─────────────────────────────────────
    # ③ 보수·보강 공법 의사결정 지원
    #    (부재 종류 + 손상 유형 + 등급 → 1순위/2순위 공법 비교)
    # ─────────────────────────────────────
    methods = get_repair_methods(location, damage_type, grade)

    return {
        "grade": grade,
        "grade_desc": grade_desc,
        "severity_score": severity_score,
        "risk": risk,
        "risk_label": risk_label,
        "risk_detail": risk_detail,
        "action": action,
        "methods": methods,   # 1순위/2순위 공법 비교 리스트
    }


def get_repair_methods(location, damage_type, grade):
    """
    부재 종류 + 손상 유형 + 등급에 따라
    1순위, 2순위 보수·보강 공법을 비용/시공성과 함께 추천
    """

    # 균열 - 경미~보통 (a~b등급)
    if damage_type == "균열" and grade in ["a등급", "b등급"]:
        return [
            {
                "순위": "1순위",
                "공법": "표면처리공법",
                "설명": "균열 표면에 실란트 또는 도막방수재 도포",
                "개략공사비": "낮음 (㎡당 1~2만원 수준)",
                "시공성": "쉬움 — 1일 이내 시공 가능",
                "장비": "표면 실란트, 도막방수재, 붓/롤러",
            },
            {
                "순위": "2순위",
                "공법": "관찰 후 미시공",
                "설명": "경미한 수준이므로 정기 관찰만 진행",
                "개략공사비": "없음",
                "시공성": "해당없음",
                "장비": "정기점검 체크리스트",
            },
        ]

    # 균열 - 주의 (c등급)
    elif damage_type == "균열" and grade == "c등급":
        return [
            {
                "순위": "1순위",
                "공법": "에폭시 수지 주입공법",
                "설명": "균열 내부에 에폭시 수지를 저압 주입하여 충전",
                "개략공사비": "중간 (m당 3~5만원 수준)",
                "시공성": "보통 — 양생 포함 2~3일 소요",
                "장비": "에폭시 주입기, 주입 포트, 크랙게이지",
            },
            {
                "순위": "2순위",
                "공법": "표면처리공법 + 모니터링",
                "설명": "우선 표면 처리 후 진행 속도 추적 관찰",
                "개략공사비": "낮음 (㎡당 1~2만원 수준)",
                "시공성": "쉬움 — 1일 이내",
                "장비": "표면 실란트, 정기점검 체크리스트",
            },
        ]

    # 균열 - 불량~위험 (d~e등급), 구조부재
    elif damage_type == "균열" and grade in ["d등급", "e등급"] and location in ["기둥", "보"]:
        return [
            {
                "순위": "1순위",
                "공법": "강판보강공법",
                "설명": "강판을 부재 외부에 부착하여 내력 보강",
                "개략공사비": "높음 (m당 15~25만원 수준)",
                "시공성": "어려움 — 전문 시공팀 필요, 5일 이상",
                "장비": "강판, 앵커볼트, 에폭시 접착제, 천공기",
            },
            {
                "순위": "2순위",
                "공법": "탄소섬유시트 보강공법",
                "설명": "탄소섬유 시트를 부착하여 인장 보강",
                "개략공사비": "높음 (m당 20~30만원 수준, 강판보다 경량)",
                "시공성": "어려움 — 표면처리 후 부착, 3~4일",
                "장비": "탄소섬유시트, 에폭시 함침수지, 롤러",
            },
        ]

    # 균열 - 불량~위험 (d~e등급), 슬래브
    elif damage_type == "균열" and grade in ["d등급", "e등급"] and location == "슬래브":
        return [
            {
                "순위": "1순위",
                "공법": "탄소섬유보강공법",
                "설명": "슬래브 하부에 탄소섬유시트 부착으로 휨 보강",
                "개략공사비": "높음 (㎡당 8~12만원 수준)",
                "시공성": "어려움 — 동바리 설치 후 시공, 4~5일",
                "장비": "탄소섬유시트, 에폭시 함침수지",
            },
            {
                "순위": "2순위",
                "공법": "강판보강공법",
                "설명": "강판 부착으로 슬래브 보강",
                "개략공사비": "높음 (㎡당 10~15만원 수준)",
                "시공성": "어려움 — 중량물 취급, 5일 이상",
                "장비": "강판, 앵커볼트, 에폭시 접착제",
            },
        ]

    # 처짐 손상
    elif damage_type == "처짐":
        return [
            {
                "순위": "1순위",
                "공법": "H형강 지지공법",
                "설명": "H형강으로 부재 하부 지지 보강",
                "개략공사비": "높음",
                "시공성": "어려움 — 구조 검토 선행 필요",
                "장비": "H형강, 받침대, 용접장비",
            },
            {
                "순위": "2순위",
                "공법": "강판보강공법",
                "설명": "강판 부착으로 처짐 부재 보강",
                "개략공사비": "높음",
                "시공성": "어려움 — 전문 시공 필요",
                "장비": "강판, 앵커볼트",
            },
        ]

    # 박리 손상
    else:
        return [
            {
                "순위": "1순위",
                "공법": "단면복구공법",
                "설명": "박리 부위 제거 후 폴리머 모르타르로 단면 복구",
                "개략공사비": "중간",
                "시공성": "보통 — 2~3일",
                "장비": "폴리머 모르타르, 철근 방청제, 그라인더",
            },
            {
                "순위": "2순위",
                "공법": "표면처리공법",
                "설명": "경미한 경우 표면 마감재로 보호",
                "개략공사비": "낮음",
                "시공성": "쉬움 — 1일 이내",
                "장비": "표면 마감재",
            },
        ]


if __name__ == "__main__":
    test_cases = [
        (0.1,  "기둥",  "균열"),
        (0.25, "보",    "균열"),
        (0.5,  "슬래브","균열"),
        (1.5,  "기둥",  "균열"),
        (2.5,  "보",    "균열"),
        (1.0,  "벽",    "박리"),
        (0.8,  "바닥",  "처짐"),
    ]

    for width, loc, dtype in test_cases:
        result = evaluate_crack(width, loc, dtype)
        print(f"\n{'='*55}")
        print(f"  부재: {loc} | 균열폭: {width}mm | 손상유형: {dtype}")
        print(f"  상태등급: {result['grade']} — {result['grade_desc']}")
        print(f"  위험도: {result['risk_label']}")
        print(f"  조치사항: {result['action']}")
        print(f"  ── 보수 공법 ──")
        for m in result["methods"]:
            print(f"  [{m['순위']}] {m['공법']}: {m['설명']}")
            print(f"         비용: {m['개략공사비']} | 시공성: {m['시공성']}")