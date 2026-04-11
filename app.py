import os
import tempfile
from typing import Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="음성 분석 센터", layout="wide")
st.title("🎭 연기과 음성 자가진단 시스템")


# =========================
# 유틸 함수
# =========================
def save_uploaded_file_to_temp(uploaded_file) -> str:
    """
    업로드된 파일을 임시 파일로 저장하고 경로를 반환합니다.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def load_audio_safely(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    오디오 파일을 안전하게 로드합니다.
    성공 시: (y, sr, None)
    실패 시: (None, None, 에러메시지)
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        if y is None or len(y) == 0:
            return None, None, "오디오 데이터가 비어 있습니다."

        return y, sr, None

    except Exception as e:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".m4a":
            return (
                None,
                None,
                "m4a 파일을 읽는 과정에서 문제가 발생했습니다. "
                "현재 환경에서는 wav 또는 mp3 파일이 가장 안정적으로 분석됩니다.\n\n"
                f"상세 오류: {e}"
            )

        return None, None, f"오디오 파일을 읽지 못했습니다.\n\n상세 오류: {e}"


def analyze_voice(y: np.ndarray, sr: int) -> dict:
    """
    STFT 기반으로 주파수 대역 분포를 분석합니다.
    이 값은 절대적인 발성 평가가 아니라, 현재 음성의 대역 경향을 참고용으로 보여주는 값입니다.
    """
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    mean_stft = np.mean(stft, axis=1)

    def band_energy(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return 0.0
        return float(np.mean(mean_stft[mask]))

    low_val = band_energy(100, 500)       # 하부 울림 경향
    mid_val = band_energy(500, 1500)      # 전방 전달 경향
    high_val = band_energy(2500, 4500)    # 상부 울림 경향

    total = low_val + mid_val + high_val

    if total <= 0:
        low_score = mid_score = high_score = 0
    else:
        low_score = int(round((low_val / total) * 100))
        mid_score = int(round((mid_val / total) * 100))
        high_score = int(round((high_val / total) * 100))

        # 합계 100 보정
        diff = 100 - (low_score + mid_score + high_score)
        low_score += diff

    return {
        "raw": {
            "low_val": low_val,
            "mid_val": mid_val,
            "high_val": high_val,
        },
        "score": {
            "low": max(0, low_score),
            "mid": max(0, mid_score),
            "high": max(0, high_score),
        },
    }


def build_local_feedback(student_id: str, low: int, mid: int, high: int) -> str:
    """
    훈련 제안 없이, 분석 요약 / 강점 / 보완 필요 지점만 출력합니다.
    """
    lines = []

    lines.append(f"{student_id} 학생의 음성 분석 결과입니다.")
    lines.append(
        f"하부 울림 경향 {low}점, 전방 전달 경향 {mid}점, 상부 울림 경향 {high}점으로 나타납니다."
    )
    lines.append("")
    lines.append("이 수치는 절대평가가 아니라, 현재 음성의 주파수 대역 분포를 바탕으로 한 참고 지표입니다.")
    lines.append("")

    # 분석 요약
    lines.append("분석 요약")
    if low >= mid and low >= high:
        lines.append("전체적으로 하부 중심의 울림 경향이 상대적으로 두드러집니다.")
    elif mid >= low and mid >= high:
        lines.append("전체적으로 전방 전달 경향이 비교적 두드러집니다.")
    else:
        lines.append("전체적으로 상부 울림 활용 경향이 상대적으로 두드러집니다.")
    lines.append("")

    # 강점
    lines.append("강점")
    strength_written = False

    if low >= 45:
        lines.append("하부 에너지 기반은 비교적 안정적으로 형성된 상태로 보입니다.")
        strength_written = True

    if mid >= 30:
        lines.append("말소리의 전방 전달 가능성은 일정 부분 확보된 상태로 보입니다.")
        strength_written = True

    if high >= 20:
        lines.append("상부 울림 영역도 완전히 닫혀 있지는 않은 것으로 보입니다.")
        strength_written = True

    if not strength_written:
        lines.append("특정 대역이 과도하게 무너지지 않은 점은 확인됩니다.")

    lines.append("")

    # 보완 필요 지점
    lines.append("보완 필요 지점")
    weakness_written = False

    if low < 35:
        lines.append("하부 에너지 기반이 충분하지 않아 소리가 가볍게 들릴 가능성이 있습니다.")
        weakness_written = True

    if mid < 25:
        lines.append("전방 전달 경향이 약해 발음의 명료도나 전달력이 떨어질 가능성이 있습니다.")
        weakness_written = True

    if high < 12:
        lines.append("상부 울림 활용은 다소 제한적으로 나타납니다.")
        weakness_written = True

    if not weakness_written:
        lines.append("세 대역이 극단적으로 치우치지는 않아 전체 균형은 비교적 무난한 편으로 보입니다.")

    return "\n".join(lines)


# =========================
# 사이드바
# =========================
with st.sidebar:
    st.header("👤 학생 정보")
    student_id = st.text_input("학번을 입력하세요")
    st.caption("권장 파일 형식: wav, mp3")
    st.caption("m4a는 환경에 따라 바로 읽히지 않을 수 있습니다.")


# =========================
# 본문
# =========================
if not student_id:
    st.info("왼쪽 사이드바에 학번을 입력한 뒤 음성 파일을 올려주세요.")
else:
    uploaded_file = st.file_uploader(
        "음성 파일을 올려주세요",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        temp_path = None

        try:
            with st.spinner("음성 파일을 분석 중입니다..."):
                temp_path = save_uploaded_file_to_temp(uploaded_file)
                y, sr, load_error = load_audio_safely(temp_path)

                if load_error:
                    st.error(load_error)
                    st.stop()

                result = analyze_voice(y, sr)
                low = result["score"]["low"]
                mid = result["score"]["mid"]
                high = result["score"]["high"]

            st.divider()
            st.subheader(f"📊 {student_id} 학생의 분석 결과")

            col1, col2, col3 = st.columns(3)
            col1.metric("하부 울림 경향", f"{low}점")
            col2.metric("전방 전달 경향", f"{mid}점")
            col3.metric("상부 울림 경향", f"{high}점")

            chart_data = pd.DataFrame(
                {
                    "분석 항목": ["하부 울림", "전방 전달", "상부 울림"],
                    "점수": [low, mid, high],
                }
            )

            st.bar_chart(chart_data, x="분석 항목", y="점수")

            st.divider()
            st.subheader("📝 분석 해석")
            feedback = build_local_feedback(student_id, low, mid, high)
            st.text(feedback)

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass