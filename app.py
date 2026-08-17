import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import parselmouth
from audio_recorder_streamlit import audio_recorder

# 1. 페이지 기본 설정 및 디자인
st.set_page_config(page_title="연기 발성 5대 공명 자가진단 시스템", page_icon="🎙️", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 2.0rem; font-weight: 700; color: #1E1E1E; text-align: center; }
    .sub-title { font-size: 0.95rem; color: #666666; text-align: center; margin-bottom: 20px; }
    .guide-box { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #1F77B4; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎙️ 연기 발성 5대 공명 진단 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Praat 음향학 분석 엔진 기반 · 실시간 음성 진단</div>', unsafe_allow_html=True)
st.divider()

# 2. 가이드 및 성별 선택
st.markdown("""
    <div class="guide-box">
        <b>📌 테스트 진행 방법</b><br>
        1. 성별을 먼저 선택해 주세요.<br>
        2. 아래 마이크 아이콘을 누르고, 편안한 톤으로 <b>"에---"</b> 소리를 3~5초간 일정하게 내주세요.<br>
        3. 녹음 버튼을 한 번 더 누르면 녹음이 종료되고 바로 분석됩니다.
    </div>
""", unsafe_allow_html=True)

gender = st.radio("성별 선택", options=["남성 (Male)", "여성 (Female)"], index=0, horizontal=True)

st.write("")
st.subheader("🎙️ 실시간 음성 녹음")

# 3. 마이크 녹음 위젯
audio_bytes = audio_recorder(
    text="버튼을 눌러 녹음을 시작하세요",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="3x",
)

# 4. Praat 기반 분석 엔진
def analyze_resonance_praat(audio_bytes, is_female=False):
    with open("temp_audio_input.wav", "wb") as f:
        f.write(audio_bytes)
    
    sound = parselmouth.Sound("temp_audio_input.wav")
    max_formant = 5500.0 if is_female else 5000.0
    formant = sound.to_formant_burg(max_formant=max_formant)
    
    y, sr = librosa.load("temp_audio_input.wav", sr=None)
    y = librosa.util.normalize(y)
    
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    mean_spectrum = np.mean(stft, axis=1)

    shift = 1.18 if is_female else 1.0

    # 5대 공명 대역 (Hz)
    chest_e = np.mean(mean_spectrum[(freqs >= 80) & (freqs <= 450 * shift)])
    palate_e = np.mean(mean_spectrum[(freqs > 450 * shift) & (freqs <= 1200 * shift)])
    teeth_e = np.mean(mean_spectrum[(freqs > 1200 * shift) & (freqs <= 2400 * shift)])
    nasal_e = np.mean(mean_spectrum[(freqs > 2400 * shift) & (freqs <= 3500 * shift)])
    head_e = np.mean(mean_spectrum[(freqs > 3500 * shift) & (freqs <= 5200 * shift)])

    # 감도 가중치 보정
    raw_scores = np.array([chest_e * 0.7, palate_e * 1.8, teeth_e * 2.8, nasal_e * 4.2, head_e * 5.5])
    norm_scores = (raw_scores / (np.sum(raw_scores) + 1e-6)) * 100
    return norm_scores.astype(int)

# 5. 녹음 완료 후 즉시 분석 실행
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    
    with st.spinner("Praat 음향학 엔진으로 5대 공명을 정밀 분석 중입니다..."):
        try:
            scores = analyze_resonance_praat(audio_bytes, is_female=("여성" in gender))
            categories = ['가슴(Chest)', '입천장(Palate)', '이빨/전방(Teeth)', '비강(Nasal)', '두개골(Head)']
            
            st.success("✅ 진단이 완료되었습니다!")
            
            # 5축 레이더 차트
            fig = go.Figure(go.Scatterpolar(
                r=list(scores) + [scores[0]],
                theta=categories + [categories[0]],
                fill='toself', fillcolor='rgba(31, 119, 180, 0.3)',
                line=dict(color='#1F77B4', width=3)
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(max(scores)+10, 40)])), showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # 세부 수치표
            cols = st.columns(5)
            labels = ["가슴", "입천장", "이빨/전방", "비강", "두개골"]
            for idx, col in enumerate(cols):
                col.metric(labels[idx], f"{scores[idx]}점")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
