import os
import tempfile
from typing import Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import streamlit as st
from pydub import AudioSegment  # m4a 변환을 위해 추가

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="음성 분석 센터", layout="wide")
st.title("🎭 연기과 음성 자가진단 시스템")

# =========================
# 유틸 함수
# =========================
def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def load_audio_safely(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        # [핵심 수정] m4a 파일인 경우 pydub를 사용하여 강제 변환 후 로드
        if ext in [".m4a", ".aac"]:
            audio = AudioSegment.from_file(file_path)
            # 분석 가능한 샘플 데이터로 변환
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            sr = audio.frame_rate
            # 스테레오라면 모노로 통합
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            # librosa 규격에 맞게 정규화 (1.0 ~ -1.0)
            y = samples / (2**15)
        else:
            # wav, mp3는 기존 방식 유지
            y, sr = librosa.load(file_path, sr=None, mono=True)

        if y is None or len(y) == 0:
            return None, None, "오디오 데이터가 비어 있습니다."

        return y, sr, None

    except Exception as e:
        return None, None, f"파일을 읽지 못했습니다. (확장자 확인 요망)\n\n상세 오류: {e}"

def analyze_voice(y: np.ndarray, sr: int) -> dict:
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    mean_stft = np.mean(stft, axis=1)

    def band_energy(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return 0.0
        return float(np.mean(mean_stft[mask]))

    low_val = band_energy(100, 500)
    mid_val = band_energy(500, 1500)
    high_val = band_energy(2500, 4500)

    total = low_val + mid_val + high_val

    if total <= 0:
        low_score = mid_score = high_score = 0
    else:
        low_score = int(round((low_val / total) * 100))
        mid_score = int(round((mid_val / total) * 100))
        high_score = int(round((high_val / total) * 100))
        diff = 100 - (low_score + mid_score + high_score)
        low_score += diff

    return {
        "score": {"low": max(0, low_score), "mid": max(0, mid_score), "high": max(0, high_score)}
    }

def build_local_feedback(student_id: str, low: int, mid: int, high: int) -> str:
    lines = [f"{student_id} 학생의 음성 분석 결과입니다.", 
             f"하부 울림 {low}점, 전방 전달 {mid}점, 상부 울림 {high}점으로 나타납니다.", ""]
    
    lines.append("분석 요약")
    if low >= mid and low >= high: lines.append("전체적으로 하부 중심의 울림 경향이 두드러집니다.")
    elif mid >= low and mid >= high: lines.append("전체적으로 전방 전달 경향이 두드러집니다.")
    else: lines.append("전체적으로 상부 울림 활용 경향이 두드러집니다.")
    
    lines.append("\n보안 필요 지점")
    if low < 35: lines.append("- 하부 에너지 기반을 더 보완하면 소리에 무게감이 생깁니다.")
    if mid < 25: lines.append("- 전방 전달력을 키우면 대사 전달력이 좋아집니다.")
    if high < 12: lines.append("- 상부 울림을 활용하면 소리의 선명도가 높아집니다.")
    
    return "\n".join(lines)

# =========================
# 사이드바 & 본문
# =========================
with st.sidebar:
    st.header("👤 학생 정보")
    student_id = st.text_input("학번을 입력하세요")

if not student_id:
    st.info("왼쪽 사이드바에 학번을 입력한 뒤 음성 파일을 올려주세요.")
else:
    uploaded_file = st.file_uploader("음성 파일을 올려주세요", type=["wav", "mp3", "m4a"])

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
                low, mid, high = result["score"]["low"], result["score"]["mid"], result["score"]["high"]

            st.divider()
            st.subheader(f"📊 {student_id} 학생의 분석 결과")
            col1, col2, col3 = st.columns(3)
            col1.metric("하부 울림", f"{low}점")
            col2.metric("전방 전달", f"{mid}점")
            col3.metric("상부 울림", f"{high}점")

            st.bar_chart(pd.DataFrame({"항목": ["하부", "전방", "상부"], "점수": [low, mid, high]}), x="항목", y="점수")
            st.divider()
            st.subheader("📝 분석 해석")
            st.text(build_local_feedback(student_id, low, mid, high))

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
