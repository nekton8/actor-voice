import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import parselmouth
from audio_recorder_streamlit import audio_recorder
import streamlit.components.v1 as components

# 1. 페이지 기본 설정 및 스타일
st.set_page_config(page_title="연기 발성 5대 공명 진단 시스템", page_icon="🎙️", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1E1E1E; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 0.95rem; color: #666666; text-align: center; margin-bottom: 25px; }
    .guide-card { background-color: #F0F4F8; padding: 20px; border-radius: 12px; border-left: 6px solid #1F77B4; margin-bottom: 20px; }
    .step-header { font-size: 1.2rem; font-weight: 600; color: #0F4C81; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'gender' not in st.session_state:
    st.session_state.gender = "남성 (Male)"
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

# 상단 헤더
st.markdown('<div class="main-title">🎙️ 연기 발성 5대 공명 진단 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Praat 음향학 분석 엔진 기반 · 단계별 실시간 공명 진단</div>', unsafe_allow_html=True)
st.progress(st.session_state.step / 3)
st.divider()

# ==========================================
# PAGE 1: 성별 선택
# ==========================================
if st.session_state.step == 1:
    st.markdown('<div class="step-header">STEP 1. 사용자 기본 정보 입력</div>', unsafe_allow_html=True)
    
    st.write("정확한 공명 포먼트 분석을 위해 성별을 먼저 선택해 주세요.")
    gender_choice = st.radio(
        "성별 선택",
        options=["남성 (Male)", "여성 (Female)"],
        index=0 if st.session_state.gender == "남성 (Male)" else 1,
        horizontal=True
    )
    
    st.write("")
    if st.button("다음 단계로 이동 ➡️", use_container_width=True, type="primary"):
        st.session_state.gender = gender_choice
        st.session_state.step = 2
        st.rerun()

# ==========================================
# PAGE 2: 실시간 녹음 및 남은 시간 카운트다운
# ==========================================
elif st.session_state.step == 2:
    st.markdown('<div class="step-header">STEP 2. 발성 녹음 진행</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="guide-card">
            <b>📌 정확한 분석을 위한 3가지 수칙</b><br><br>
            1. <b>마이크 거리:</b> 스마트폰/마이크를 입에서 <b>주먹 하나 거리(약 15cm)</b> 띄우세요.<br>
            2. <b>발성 방법:</b> 가장 편안한 톤으로 <b>"에---"</b> 소리를 끊기지 않게 일정하게 내세요.<br>
            3. <b>녹음 시간:</b> 아래 남은 시간이 <b>0초가 될 때까지 5초간 발성을 유지</b>해 주세요.
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🎙️ 실시간 음성 녹음")
    st.caption("마이크를 누르면 녹음 상태(노란색) 변화를 감지하여 5초 남은 시간이 동작합니다.")

    # 마이크 위젯 (녹음 유지 5초)
    recorded_audio = audio_recorder(
        text="마이크를 눌러 녹음 시작 / 중지",
        recording_color="#e8b62c",
        neutral_color="#1F77B4",
        icon_name="microphone",
        icon_size="4x",
        pause_threshold=5.0,
    )

    # 마이크 상태를 감지해서 실시간 타이머를 출력하는 프론트엔드 주입
    timer_script = """
    <div id="recording-timer" style="
        font-size: 26px; 
        font-weight: bold; 
        color: #E74C3C; 
        text-align: center; 
        margin-top: 10px;
        min-height: 40px;
    "></div>

    <script>
    let intervalId = null;
    let timerContainer = document.getElementById('recording-timer');

    function checkMicState() {
        // 부모 창의 audio_recorder 아이콘 상태 모니터링
        const parentDoc = window.parent.document;
        const micIcon = parentDoc.querySelector('svg[data-icon="microphone"]');
        
        if (micIcon) {
            const isRecording = micIcon.style.fill === 'rgb(232, 182, 44)' || micIcon.getAttribute('fill') === '#e8b62c';
            
            if (isRecording && !intervalId) {
                let timeLeft = 5;
                timerContainer.innerHTML = "🔴 녹음 중... 남은 시간: " + timeLeft + "초";
                
                intervalId = setInterval(() => {
                    timeLeft--;
                    if (timeLeft >= 0) {
                        timerContainer.innerHTML = "🔴 녹음 중... 남은 시간: " + timeLeft + "초";
                    } else {
                        timerContainer.innerHTML = "✅ 5초 녹음 완료!";
                        timerContainer.style.color = "#27AE60";
                        clearInterval(intervalId);
                    }
                }, 1000);
            } else if (!isRecording && intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
        }
    }

    setInterval(checkMicState, 300);
    </script>
    """
    components.html(timer_script, height=60)

    if recorded_audio:
        st.session_state.audio_bytes = recorded_audio

    # 녹음 완료 시 플레이어 및 버튼 노출
    if st.session_state.audio_bytes is not None:
        st.write("")
        st.success("✅ 녹음이 성공적으로 완료되었습니다! 미리 들어보시고 분석을 진행해 주세요.")
        st.audio(st.session_state.audio_bytes, format="audio/wav")
        
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 다시 녹음하기", use_container_width=True):
                st.session_state.audio_bytes = None
                st.rerun()
        with col2:
            if st.button("📊 분석 결과 보기 ➡️", use_container_width=True, type="primary"):
                st.session_state.step = 3
                st.rerun()
                
    st.write("")
    if st.button("⬅️ 이전 단계 (성별 변경)"):
        st.session_state.step = 1
        st.rerun()

# ==========================================
# PAGE 3: 5대 공명 분석 결과 (100% 기준)
# ==========================================
elif st.session_state.step == 3:
    st.markdown('<div class="step-header">STEP 3. 5대 공명 진단 결과 리포트</div>', unsafe_allow_html=True)
    
    def analyze_resonance_praat(audio_bytes, is_female=False):
        with open("temp_audio_input.wav", "wb") as f:
            f.write(audio_bytes)
        
        sound = parselmouth.Sound("temp_audio_input.wav")
        max_f = 5500.0 if is_female else 5000.0
        formant = sound.to_formant_burg(maximum_formant=max_f)
        
        y, sr = librosa.load("temp_audio_input.wav", sr=None)
        y = librosa.util.normalize(y)
        
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        mean_spectrum = np.mean(stft, axis=1)

        shift = 1.18 if is_female else 1.0

        # 5대 공명 구간 (Hz)
        chest_e = np.mean(mean_spectrum[(freqs >= 80) & (freqs <= 450 * shift)])
        palate_e = np.mean(mean_spectrum[(freqs > 450 * shift) & (freqs <= 1200 * shift)])
        teeth_e = np.mean(mean_spectrum[(freqs > 1200 * shift) & (freqs <= 2400 * shift)])
        nasal_e = np.mean(mean_spectrum[(freqs > 2400 * shift) & (freqs <= 3500 * shift)])
        head_e = np.mean(mean_spectrum[(freqs > 3500 * shift) & (freqs <= 5200 * shift)])

        raw_scores = np.array([chest_e * 0.7, palate_e * 1.8, teeth_e * 2.8, nasal_e * 4.2, head_e * 5.5])
        norm_scores = (raw_scores / (np.sum(raw_scores) + 1e-6)) * 100
        return norm_scores.astype(int)

    with st.spinner("Praat 음향학 엔진으로 5대 공명 포먼트를 정밀 분석 중입니다..."):
        try:
            is_female = "여성" in st.session_state.gender
            scores = analyze_resonance_praat(st.session_state.audio_bytes, is_female=is_female)
            categories = ['가슴(Chest)', '입천장(Palate)', '이빨/전방(Teeth)', '비강(Nasal)', '두개골(Head)']
            
            fig = go.Figure(go.Scatterpolar(
                r=list(scores) + [scores[0]],
                theta=categories + [categories[0]],
                fill='toself', fillcolor='rgba(31, 119, 180, 0.3)',
                line=dict(color='#1F77B4', width=3)
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(max(scores)+10, 40)])),
                showlegend=False,
                height=380,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("📊 **5대 공명 점수 비율 (총합 100% 기준)**")
            cols = st.columns(5)
            labels = ["가슴", "입천장", "이빨/전방", "비강", "두개골"]
            for idx, col in enumerate(cols):
                col.metric(labels[idx], f"{scores[idx]} %")

            st.divider()
            max_idx = np.argmax(scores)
            st.info(f"💡 현재 발성에서 가장 발달된 공명은 **[{labels[max_idx]} 공명 ({scores[max_idx]}%)]**입니다.")

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

    st.write("")
    if st.button("🔄 처음부터 다시 진단하기", use_container_width=True, type="primary"):
        st.session_state.step = 1
        st.session_state.audio_bytes = None
        st.rerun()
