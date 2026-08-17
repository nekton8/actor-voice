import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import parselmouth
import streamlit.components.v1 as components

# 1. 페이지 디자인
st.set_page_config(page_title="연기 발성 5대 공명 진단 시스템", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 2.1rem; font-weight: 700; text-align: center; }
    .step-header { font-size: 1.2rem; font-weight: 600; color: #0F4C81; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'step' not in st.session_state: st.session_state.step = 1
if 'audio_bytes' not in st.session_state: st.session_state.audio_bytes = None

st.markdown('<div class="main-title">🎙️ 연기 발성 5대 공명 진단 시스템</div>', unsafe_allow_html=True)
st.divider()

# ==========================================
# PAGE 2: 타이머 녹음 (STEP 2)
# ==========================================
if st.session_state.step == 2:
    st.markdown('<div class="step-header">STEP 2. 5초 실시간 녹음</div>', unsafe_allow_html=True)
    
    # 타이머 및 녹음 제어 커스텀 컴포넌트
    html_recorder = """
    <div id="recorder" style="text-align: center; padding: 20px;">
        <button id="btn" onclick="start()" style="width:160px; height:160px; border-radius:50%; background:#E74C3C; color:white; font-size:24px; font-weight:bold; border:none; cursor:pointer;">🎙️<br>녹음 시작</button>
        <div id="status" style="font-size:32px; font-weight:bold; color:#E74C3C; margin-top:20px;">05초</div>
    </div>
    <script>
    let stream, recorder, chunks=[];
    async function start() {
        stream = await navigator.mediaDevices.getUserMedia({audio:true});
        recorder = new MediaRecorder(stream);
        recorder.ondataavailable = e => chunks.push(e.data);
        recorder.onstop = () => {
            let blob = new Blob(chunks, {type:'audio/wav'});
            let reader = new FileReader();
            reader.onload = () => window.parent.postMessage({type:'streamlit:setComponentValue', value:reader.result.split(',')[1]}, '*');
            reader.readAsDataURL(blob);
        };
        recorder.start();
        let sec = 5;
        let intv = setInterval(() => {
            sec--;
            document.getElementById('status').innerText = "0" + sec + "초";
            if(sec <= 0) { clearInterval(intv); recorder.stop(); document.getElementById('btn').innerText="완료"; }
        }, 1000);
    }
    </script>
    """
    val = components.html(html_recorder, height=300)
    
    # 컴포넌트 값 수신
    if val:
        import base64
        st.session_state.audio_bytes = base64.b64decode(val)
        st.rerun()

    if st.session_state.audio_bytes:
        st.success("✅ 녹음 완료!")
        st.audio(st.session_state.audio_bytes, format="audio/wav")
        if st.button("📊 분석 결과 보기"):
            st.session_state.step = 3
            st.rerun()
