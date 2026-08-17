import base64
import io
import tempfile

import librosa
import numpy as np
import parselmouth
import soundfile as sf
import streamlit as st
from parselmouth.praat import call


st.set_page_config(
    page_title="공명 훈련",
    page_icon="🎙️",
    layout="centered",
)


st.markdown(
    """
<style>
.block-container {
    max-width: 760px;
    padding-top: 1.0rem;
    padding-bottom: 1.5rem;
}

.main-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    line-height: 1.15;
    margin: 0 0 0.2rem 0;
}

.sub-title {
    font-size: 0.94rem;
    color: #666;
    line-height: 1.4;
    margin: 0 0 1.0rem 0;
}

.step-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: #7a7a7a;
    letter-spacing: 0.07em;
    margin: 0 0 0.1rem 0;
}

.step-title {
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 0.65rem 0;
}

.guide-box {
    background: #f5f6f8;
    border-radius: 13px;
    padding: 13px 16px;
    margin: 0 0 10px 0;
    font-size: 0.92rem;
    line-height: 1.5;
}

.complete-box {
    background: #eef9f2;
    border: 1px solid #cae8d4;
    padding: 12px 15px;
    border-radius: 13px;
    margin: 8px 0 10px 0;
}

.complete-title {
    font-size: 1rem;
    font-weight: 800;
}

.complete-time {
    font-size: 0.88rem;
    color: #2d7b4b;
    margin-top: 2px;
}

.result-good,
.result-mid,
.result-bad {
    border-radius: 16px;
    padding: 22px;
    margin: 12px 0 16px 0;
}

.result-good {
    background: #eaf7ef;
    border: 1px solid #b8e1c6;
}

.result-mid {
    background: #fff7e6;
    border: 1px solid #efd28e;
}

.result-bad {
    background: #fff0f0;
    border: 1px solid #efc0c0;
}

.result-title {
    font-size: 1.45rem;
    font-weight: 800;
}

.result-value {
    font-size: 2.7rem;
    font-weight: 900;
    line-height: 1.05;
    margin: 7px 0;
}

.result-description {
    font-size: 0.95rem;
    line-height: 1.55;
}

.compare-card {
    border: 1px solid #e1e4e8;
    border-radius: 14px;
    padding: 16px 8px;
    text-align: center;
    min-height: 100px;
}

.compare-label {
    font-size: 0.82rem;
    color: #6f747b;
    margin-bottom: 3px;
}

.compare-value {
    font-size: 1.7rem;
    font-weight: 850;
}

.compare-change-up {
    font-size: 0.82rem;
    font-weight: 700;
    color: #25834a;
    margin-top: 2px;
}

.compare-change-down {
    font-size: 0.82rem;
    font-weight: 700;
    color: #c94545;
    margin-top: 2px;
}

.reason-box {
    border: 1px solid #e1e4e8;
    border-radius: 14px;
    padding: 15px 16px;
    margin: 8px 0 15px 0;
}

.tip-box {
    background: #f5f6f8;
    border-radius: 14px;
    padding: 15px 16px;
    line-height: 1.55;
    margin: 8px 0 15px 0;
}

.confidence-box {
    border-radius: 12px;
    padding: 11px 13px;
    margin: 7px 0 15px 0;
    font-size: 0.9rem;
    background: #f8f9fa;
    border: 1px solid #e5e7eb;
}

@media (max-width: 600px) {
    .block-container {
        padding-top: 0.65rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-bottom: 1rem;
    }

    .main-title {
        font-size: 1.7rem;
        margin-bottom: 0.12rem;
    }

    .sub-title {
        font-size: 0.86rem;
        margin-bottom: 0.75rem;
    }

    .step-label {
        font-size: 0.68rem;
    }

    .step-title {
        font-size: 1.28rem;
        margin-bottom: 0.5rem;
    }

    .guide-box {
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    .result-good,
    .result-mid,
    .result-bad {
        padding: 17px;
        margin-top: 8px;
        border-radius: 13px;
    }

    .result-title {
        font-size: 1.2rem;
    }

    .result-value {
        font-size: 2.2rem;
    }

    .result-description {
        font-size: 0.88rem;
    }

    .compare-card {
        min-height: 88px;
        padding: 12px 5px;
    }

    .compare-value {
        font-size: 1.45rem;
    }

    .reason-box,
    .tip-box,
    .confidence-box {
        padding: 12px;
        font-size: 0.88rem;
    }

    h3 {
        font-size: 1.08rem !important;
        margin-top: 0.9rem !important;
        margin-bottom: 0.4rem !important;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.45rem;
    }

    button {
        min-height: 40px !important;
    }

    div[data-testid="stAudio"] {
        margin-top: 0 !important;
        margin-bottom: 0.45rem !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


defaults = {
    "page": "select",
    "focus": "가슴",
    "baseline_audio": None,
    "target_audio": None,
    "baseline_recorder_id": 0,
    "target_recorder_id": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


RESONANCES = {
    "가슴": {
        "icon": "🫁",
        "description": "가슴 쪽의 깊고 풍부한 울림",
        "guide": "음을 억지로 낮추지 말고 가슴 쪽 울림을 풍부하게 만들어 보세요.",
    },
    "입천장": {
        "icon": "👄",
        "description": "입천장과 구강 공간을 활용한 울림",
        "guide": "입 안의 공간을 확보하고 입천장 쪽으로 울림이 퍼지는 느낌을 찾아보세요.",
    },
    "이빨·전방": {
        "icon": "🦷",
        "description": "소리가 앞쪽으로 또렷하게 모이는 울림",
        "guide": "소리를 밀어내지 말고 윗니와 입 앞쪽에 울림이 모이는 느낌을 찾아보세요.",
    },
    "비강": {
        "icon": "👃",
        "description": "코 주변과 얼굴 중앙에서 느껴지는 울림",
        "guide": "코로 억지로 보내기보다 얼굴 중앙에 진동이 생기는 느낌을 찾아보세요.",
    },
    "두개골": {
        "icon": "💀",
        "description": "머리 위쪽으로 확장되는 가볍고 높은 울림",
        "guide": "목에 힘을 주지 말고 소리가 머리 위쪽으로 가볍게 확장되는 느낌을 찾아보세요.",
    },
}


MODELS = {
    "가슴": {
        "primary_key": "band_80_500",
        "primary_label": "저역 에너지",
        "primary_direction": 1,
        "primary_weight": 0.75,
        "secondary_key": "centroid",
        "secondary_label": "소리의 에너지 중심 하향",
        "secondary_direction": -1,
        "secondary_weight": 0.25,
        "tip_good": "지금 만든 가슴 쪽 울림의 느낌을 기억하고 같은 음높이로 다시 재현해 보세요.",
        "tip_mid": "음을 낮추지 말고 가슴 쪽 울림을 조금 더 풍부하게 만들어 보세요.",
        "tip_bad": "기준 발성과 같은 높이를 유지한 채 가슴 쪽 진동을 다시 찾아보세요.",
    },
    "입천장": {
        "primary_key": "band_500_1500",
        "primary_label": "중저역·구강 에너지",
        "primary_direction": 1,
        "primary_weight": 0.70,
        "secondary_key": "band_1500_3000",
        "secondary_label": "중고역 에너지",
        "secondary_direction": 1,
        "secondary_weight": 0.30,
        "tip_good": "입 안의 공간감을 유지하면서 같은 울림을 다시 재현해 보세요.",
        "tip_mid": "턱과 혀에 힘을 빼고 입천장 쪽 공간감을 조금 더 만들어 보세요.",
        "tip_bad": "소리를 밀지 말고 입 안의 공간을 넓게 만든 뒤 다시 시도해 보세요.",
    },
    "이빨·전방": {
        "primary_key": "band_1500_3000",
        "primary_label": "전방 명료도 대역",
        "primary_direction": 1,
        "primary_weight": 0.75,
        "secondary_key": "centroid",
        "secondary_label": "소리의 에너지 중심 전진",
        "secondary_direction": 1,
        "secondary_weight": 0.25,
        "tip_good": "윗니와 입 앞쪽에 모인 선명한 울림을 같은 힘으로 다시 재현해 보세요.",
        "tip_mid": "볼륨을 키우기보다 소리의 초점을 조금 더 앞쪽에 모아보세요.",
        "tip_bad": "목에서 밀어내지 말고 윗니 뒤쪽으로 소리의 초점을 옮겨보세요.",
    },
    "비강": {
        "primary_key": "band_80_500",
        "primary_label": "저역 비강 관련 에너지",
        "primary_direction": 1,
        "primary_weight": 0.55,
        "secondary_key": "band_500_1500",
        "secondary_label": "중저역 에너지 감소",
        "secondary_direction": -1,
        "secondary_weight": 0.45,
        "tip_good": "얼굴 중앙에서 느껴지는 진동을 유지하면서 과도한 콧소리는 피하세요.",
        "tip_mid": "코로 소리를 밀기보다 코 주변의 진동만 조금 더 선명하게 찾아보세요.",
        "tip_bad": "입과 코의 통로를 억지로 막지 말고 얼굴 중앙의 가벼운 진동부터 다시 찾아보세요.",
    },
    "두개골": {
        "primary_key": "band_3000_5000",
        "primary_label": "상부 고역 에너지",
        "primary_direction": 1,
        "primary_weight": 0.75,
        "secondary_key": "centroid",
        "secondary_label": "소리의 에너지 중심 상승",
        "secondary_direction": 1,
        "secondary_weight": 0.25,
        "tip_good": "목의 힘을 빼고 가벼운 상부 울림을 같은 음높이에서 다시 재현해 보세요.",
        "tip_mid": "소리를 세게 올리기보다 가볍고 선명한 상부 배음을 조금 더 만들어 보세요.",
        "tip_bad": "목에 힘을 주지 말고 가벼운 소리로 머리 위쪽의 울림을 다시 찾아보세요.",
    },
}


RECORDER_HTML = """
<div class="recorder">
    <button id="recordButton" class="record-button">
        <div class="mic-icon">🎙</div>
    </button>
    <div id="statusText" class="status">녹음 시작</div>
    <div id="timerText" class="timer">00:00.0</div>
    <div class="progress-wrap">
        <div id="progressBar" class="progress"></div>
    </div>
    <div id="helpText" class="help">누르면 5초간 자동 녹음됩니다.</div>
</div>
"""


RECORDER_CSS = """
:host {
    display: block;
    width: 100%;
    height: 100%;
    overflow: hidden;
}

.recorder {
    width: 100%;
    height: 100%;
    box-sizing: border-box;
    border: 1px solid #e1e4e8;
    border-radius: 17px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 12px 10px;
    overflow: hidden;
    background: var(--st-background-color);
    font-family: var(--st-font);
}

.record-button {
    width: 78px;
    height: 78px;
    flex: 0 0 78px;
    border: none;
    border-radius: 50%;
    background: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow:
        0 4px 14px rgba(0,0,0,.13),
        inset 0 0 0 1px rgba(0,0,0,.06);
    transition: transform .15s ease, background .15s ease;
}

.record-button:active {
    transform: scale(.95);
}

.mic-icon {
    font-size: 34px;
    line-height: 1;
}

.record-button.recording {
    background: #e53935;
    animation: pulse 1.05s infinite;
}

.record-button.recording .mic-icon {
    font-size: 0;
}

.record-button.recording .mic-icon::after {
    content: "■";
    font-size: 25px;
    color: white;
}

.record-button.done {
    background: #e8f6ed;
}

.record-button.done .mic-icon {
    font-size: 0;
}

.record-button.done .mic-icon::after {
    content: "✓";
    font-size: 36px;
    font-weight: 800;
    color: #269451;
}

.status {
    margin-top: 8px;
    font-size: 14px;
    line-height: 1.15;
    font-weight: 750;
    color: var(--st-text-color);
}

.status.recording {
    color: #d32f2f;
}

.status.done {
    color: #25834a;
}

.timer {
    margin-top: 3px;
    font-size: 25px;
    line-height: 1.1;
    font-weight: 850;
    font-variant-numeric: tabular-nums;
    color: var(--st-text-color);
}

.progress-wrap {
    width: min(300px, 80%);
    height: 5px;
    margin-top: 8px;
    background: #eceff2;
    border-radius: 999px;
    overflow: hidden;
}

.progress {
    width: 0%;
    height: 100%;
    background: #e53935;
    border-radius: 999px;
}

.help {
    margin-top: 6px;
    font-size: 10.5px;
    line-height: 1.15;
    color: #7a7f87;
    text-align: center;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(229,57,53,.28); }
    70% { box-shadow: 0 0 0 12px rgba(229,57,53,0); }
    100% { box-shadow: 0 0 0 0 rgba(229,57,53,0); }
}
"""


RECORDER_JS = r"""
export default function(component) {
    const { parentElement, setStateValue } = component;

    const button = parentElement.querySelector("#recordButton");
    const statusText = parentElement.querySelector("#statusText");
    const timerText = parentElement.querySelector("#timerText");
    const progressBar = parentElement.querySelector("#progressBar");
    const helpText = parentElement.querySelector("#helpText");

    const RECORD_MS = 5000;

    let recording = false;
    let stream = null;
    let audioContext = null;
    let source = null;
    let processor = null;
    let silentGain = null;
    let buffers = [];
    let sampleRate = 44100;
    let startTime = 0;
    let timerFrame = null;
    let stopTimer = null;

    function formatTime(ms) {
        const sec = Math.min(ms, RECORD_MS) / 1000;
        return "00:" + sec.toFixed(1).padStart(4, "0");
    }

    function mergeBuffers(parts) {
        let total = 0;

        for (const part of parts) {
            total += part.length;
        }

        const merged = new Float32Array(total);
        let offset = 0;

        for (const part of parts) {
            merged.set(part, offset);
            offset += part.length;
        }

        return merged;
    }

    function writeString(view, offset, text) {
        for (let i = 0; i < text.length; i++) {
            view.setUint8(offset + i, text.charCodeAt(i));
        }
    }

    function encodeWav(samples, sr) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sr, true);
        view.setUint32(28, sr * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, "data");
        view.setUint32(40, samples.length * 2, true);

        let offset = 44;

        for (let i = 0; i < samples.length; i++) {
            let sample = Math.max(-1, Math.min(1, samples[i]));
            sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, sample, true);
            offset += 2;
        }

        return new Blob(
            [view],
            { type: "audio/wav" }
        );
    }

    function blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onloadend = () => {
                const value = reader.result;
                resolve(
                    value.substring(
                        value.indexOf(",") + 1
                    )
                );
            };

            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    function updateTimer() {
        if (!recording) {
            return;
        }

        const elapsed = performance.now() - startTime;
        const capped = Math.min(elapsed, RECORD_MS);

        timerText.textContent = formatTime(capped);
        progressBar.style.width =
            (capped / RECORD_MS * 100) + "%";

        timerFrame =
            requestAnimationFrame(updateTimer);
    }

    async function cleanup() {
        try {
            if (processor) {
                processor.disconnect();
            }
        } catch {}

        try {
            if (source) {
                source.disconnect();
            }
        } catch {}

        try {
            if (silentGain) {
                silentGain.disconnect();
            }
        } catch {}

        if (stream) {
            for (const track of stream.getTracks()) {
                track.stop();
            }
        }

        if (audioContext) {
            try {
                await audioContext.close();
            } catch {}
        }

        processor = null;
        source = null;
        silentGain = null;
        stream = null;
        audioContext = null;
    }

    async function stopRecording() {
        if (!recording) {
            return;
        }

        recording = false;

        clearTimeout(stopTimer);
        cancelAnimationFrame(timerFrame);

        timerText.textContent = "00:05.0";
        progressBar.style.width = "100%";

        button.classList.remove("recording");
        statusText.classList.remove("recording");

        statusText.textContent = "저장 중...";
        helpText.textContent = "잠시만 기다려주세요.";

        const samples = mergeBuffers(buffers);
        const wavBlob = encodeWav(
            samples,
            sampleRate
        );

        await cleanup();

        const audioBase64 =
            await blobToBase64(wavBlob);

        button.classList.add("done");
        statusText.classList.add("done");

        statusText.textContent = "녹음 완료";
        helpText.textContent =
            "5초 녹음이 저장되었습니다.";

        setStateValue(
            "audio_b64",
            audioBase64
        );
    }

    async function startRecording() {
        if (recording) {
            return;
        }

        try {
            stream =
                await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false,
                        channelCount: 1
                    }
                });

            audioContext =
                new (
                    window.AudioContext
                    ||
                    window.webkitAudioContext
                )();

            sampleRate =
                audioContext.sampleRate;

            source =
                audioContext.createMediaStreamSource(
                    stream
                );

            processor =
                audioContext.createScriptProcessor(
                    4096,
                    1,
                    1
                );

            silentGain =
                audioContext.createGain();

            silentGain.gain.value = 0;

            buffers = [];

            processor.onaudioprocess =
                (event) => {

                    if (!recording) {
                        return;
                    }

                    const input =
                        event.inputBuffer
                        .getChannelData(0);

                    buffers.push(
                        new Float32Array(input)
                    );
                };

            source.connect(processor);
            processor.connect(silentGain);
            silentGain.connect(
                audioContext.destination
            );

            recording = true;

            button.classList.remove("done");
            button.classList.add("recording");

            statusText.classList.remove("done");
            statusText.classList.add("recording");

            statusText.textContent =
                "● 녹음 중";

            helpText.textContent =
                "5초 후 자동으로 종료됩니다.";

            timerText.textContent =
                "00:00.0";

            progressBar.style.width =
                "0%";

            startTime =
                performance.now();

            updateTimer();

            stopTimer =
                setTimeout(
                    stopRecording,
                    RECORD_MS
                );

        } catch (error) {

            console.error(error);

            statusText.textContent =
                "마이크 사용 불가";

            helpText.textContent =
                "브라우저의 마이크 권한을 허용해주세요.";
        }
    }

    button.onclick = () => {
        if (!recording) {
            startRecording();
        }
    };

    return () => {

        clearTimeout(stopTimer);
        cancelAnimationFrame(timerFrame);

        if (stream) {
            for (const track of stream.getTracks()) {
                track.stop();
            }
        }

        if (audioContext) {
            try {
                audioContext.close();
            } catch {}
        }
    };
}
"""


recorder_component = st.components.v2.component(
    "resonance_mobile_recorder_v3",
    html=RECORDER_HTML,
    css=RECORDER_CSS,
    js=RECORDER_JS,
)


def five_second_recorder(key):

    result = recorder_component(
        default={
            "audio_b64": None
        },
        on_audio_b64_change=lambda: None,
        key=key,
        width="stretch",
        height=205,
    )

    audio_b64 = getattr(
        result,
        "audio_b64",
        None
    )

    if not audio_b64:
        return None

    try:
        return base64.b64decode(
            audio_b64
        )
    except Exception:
        return None


def audio_duration(audio_bytes):

    data, sr = sf.read(
        io.BytesIO(
            audio_bytes
        )
    )

    return float(
        len(data) / sr
    )


def safe_value(value):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def percent_change(old, new):

    if (
        not np.isfinite(old)
        or
        not np.isfinite(new)
        or
        abs(old) < 1e-10
    ):
        return np.nan

    return (
        (new - old)
        /
        abs(old)
        *
        100
    )


def band_ratio(
    power,
    freqs,
    low,
    high
):

    band_mask = (
        (freqs >= low)
        &
        (freqs < high)
    )

    total_mask = (
        (freqs >= 80)
        &
        (freqs < 5000)
    )

    total = (
        np.sum(
            power[
                total_mask
            ]
        )
        +
        1e-12
    )

    return float(
        np.sum(
            power[
                band_mask
            ]
        )
        /
        total
        *
        100
    )


def analyze_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=True
    ) as tmp:

        tmp.write(
            audio_bytes
        )

        tmp.flush()

        y, sr = librosa.load(
            tmp.name,
            sr=None,
            mono=True
        )

        y, _ = librosa.effects.trim(
            y,
            top_db=35
        )

        duration = (
            len(y)
            /
            sr
        )

        if duration < 2:

            raise ValueError(
                "유효한 발성 시간이 너무 짧습니다."
            )

        y = librosa.util.normalize(
            y
        )

        sound = parselmouth.Sound(
            tmp.name
        )

        pitch = call(
            sound,
            "To Pitch",
            0.0,
            60.0,
            500.0
        )

        f0 = call(
            pitch,
            "Get mean",
            0,
            0,
            "Hertz"
        )

        f0 = safe_value(
            f0
        )

        spectrum = np.abs(
            librosa.stft(
                y,
                n_fft=2048,
                hop_length=512
            )
        )

        power = (
            spectrum ** 2
        )

        mean_power = np.mean(
            power,
            axis=1
        )

        freqs = (
            librosa.fft_frequencies(
                sr=sr,
                n_fft=2048
            )
        )

        centroid = float(
            np.mean(
                librosa.feature
                .spectral_centroid(
                    y=y,
                    sr=sr
                )
            )
        )

        return {
            "duration": duration,
            "f0": f0,
            "centroid": centroid,

            "band_80_500":
                band_ratio(
                    mean_power,
                    freqs,
                    80,
                    500
                ),

            "band_500_1500":
                band_ratio(
                    mean_power,
                    freqs,
                    500,
                    1500
                ),

            "band_1500_3000":
                band_ratio(
                    mean_power,
                    freqs,
                    1500,
                    3000
                ),

            "band_3000_5000":
                band_ratio(
                    mean_power,
                    freqs,
                    3000,
                    5000
                ),
        }


def feature_change(
    baseline,
    target,
    key,
    direction
):

    raw = percent_change(
        baseline[key],
        target[key]
    )

    if not np.isfinite(raw):
        return 0.0

    return float(
        np.clip(
            raw * direction,
            -50,
            50
        )
    )


def confidence_from_pitch(
    f0_change
):

    if not np.isfinite(
        f0_change
    ):

        return (
            "낮음",
            "기본 음높이를 안정적으로 측정하지 못했습니다."
        )

    if f0_change <= 12:

        return (
            "높음",
            f"두 발성의 음높이 차이가 "
            f"{f0_change:.1f}%로 안정적입니다."
        )

    if f0_change <= 25:

        return (
            "보통",
            f"음높이가 {f0_change:.1f}% 달라 "
            "공명 변화에 일부 영향을 줄 수 있습니다."
        )

    return (
        "낮음",
        f"음높이가 {f0_change:.1f}% 달라 "
        "결과에 음높이 영향이 비교적 큽니다."
    )


def judge_resonance(
    focus,
    baseline,
    target
):

    model = MODELS[
        focus
    ]

    primary = feature_change(
        baseline,
        target,
        model[
            "primary_key"
        ],
        model[
            "primary_direction"
        ],
    )

    secondary = feature_change(
        baseline,
        target,
        model[
            "secondary_key"
        ],
        model[
            "secondary_direction"
        ],
    )

    score = (
        primary
        *
        model[
            "primary_weight"
        ]
        +
        secondary
        *
        model[
            "secondary_weight"
        ]
    )

    score = float(
        np.clip(
            score,
            -50,
            50
        )
    )

    f0_change = abs(
        percent_change(
            baseline[
                "f0"
            ],
            target[
                "f0"
            ]
        )
    )

    confidence, confidence_msg = (
        confidence_from_pitch(
            f0_change
        )
    )

    if score >= 10:

        status = "good"
        title = "잘 되고 있습니다"
        css = "result-good"

    elif score >= 3:

        status = "mid"
        title = "방향은 맞습니다"
        css = "result-mid"

    elif score > -3:

        status = "neutral"
        title = "뚜렷한 변화가 없습니다"
        css = "result-mid"

    else:

        status = "bad"
        title = (
            f"{focus} 공명 특성이 감소했습니다"
        )
        css = "result-bad"

    if status == "good":

        message = (
            f"기준 발성보다 {focus} 공명과 관련된 "
            "음향 특성이 뚜렷하게 강화되었습니다."
        )

        tip = model[
            "tip_good"
        ]

    elif status == "mid":

        message = (
            f"{focus} 공명과 관련된 변화가 "
            "나타나고 있습니다."
        )

        tip = model[
            "tip_mid"
        ]

    elif status == "neutral":

        message = (
            f"기준 발성과 비교했을 때 "
            f"{focus} 공명의 뚜렷한 강화는 "
            "확인되지 않았습니다."
        )

        tip = model[
            "tip_mid"
        ]

    else:

        message = (
            f"기준 발성보다 {focus} 공명과 관련된 "
            "음향 특성이 감소했습니다."
        )

        tip = model[
            "tip_bad"
        ]

    return {
        "status": status,
        "title": title,
        "css": css,
        "message": message,
        "score": score,
        "primary": primary,
        "secondary": secondary,
        "f0_change": f0_change,
        "confidence": confidence,
        "confidence_msg": confidence_msg,
        "tip": tip,
    }


if st.session_state.page == "select":

    title = "🎙️ 공명 훈련"

    subtitle = (
        "훈련할 공명을 선택하고 "
        "나의 기준 발성과 비교합니다."
    )

else:

    focus = (
        st.session_state.focus
    )

    title = (
        f"{RESONANCES[focus]['icon']} "
        f"{focus} 공명 훈련"
    )

    subtitle = (
        f"기준 발성과 비교하여 "
        f"{focus} 공명의 변화를 확인합니다."
    )


st.markdown(
    f'<div class="main-title">{title}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="sub-title">{subtitle}</div>',
    unsafe_allow_html=True,
)


if st.session_state.page == "select":

    st.markdown(
        '<div class="step-label">STEP 1</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="step-title">'
        '훈련할 공명을 선택하세요'
        '</div>',
        unsafe_allow_html=True,
    )

    options = list(
        RESONANCES.keys()
    )

    focus = st.radio(
        "공명 선택",
        options,
        label_visibility="collapsed",
    )

    info = (
        RESONANCES[
            focus
        ]
    )

    st.markdown(
        f"""
<div class="guide-box">
<b>{info['icon']} {focus} 공명</b><br>
{info['description']}
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        f"{focus} 공명 훈련 시작 ➡️",
        type="primary",
        use_container_width=True
    ):

        st.session_state.focus = (
            focus
        )

        st.session_state.page = (
            "baseline_record"
        )

        st.rerun()


elif st.session_state.page == "baseline_record":

    st.markdown(
        '<div class="step-label">STEP 2</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="step-title">'
        '기준 발성 녹음'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="guide-box">
<b>편안하게 /아/를 발성하세요.</b><br>
특정 공명을 의도하지 말고 자연스럽게 발성합니다.
버튼을 누르면 <b>5초간 자동 녹음</b>됩니다.
</div>
""",
        unsafe_allow_html=True,
    )

    audio = five_second_recorder(
        "baseline_"
        +
        str(
            st.session_state
            .baseline_recorder_id
        )
    )

    if audio:

        st.session_state.baseline_audio = (
            audio
        )

        st.session_state.page = (
            "baseline_review"
        )

        st.rerun()

    if st.button(
        "⬅️ 공명 다시 선택"
    ):

        st.session_state.page = (
            "select"
        )

        st.rerun()


elif st.session_state.page == "baseline_review":

    duration = audio_duration(
        st.session_state
        .baseline_audio
    )

    st.markdown(
        '<div class="step-label">STEP 2</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="step-title">'
        '기준 발성 확인'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="complete-box">
<div class="complete-title">✅ 녹음 완료</div>
<div class="complete-time">
녹음 시간 · {duration:.1f}초
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.audio(
        st.session_state
        .baseline_audio,
        format="audio/wav"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        if st.button(
            "🔄 다시 녹음",
            use_container_width=True
        ):

            st.session_state.baseline_audio = (
                None
            )

            st.session_state.baseline_recorder_id += (
                1
            )

            st.session_state.page = (
                "baseline_record"
            )

            st.rerun()

    with col2:

        if st.button(
            "이 녹음으로 진행 ➡️",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = (
                "target_record"
            )

            st.rerun()


elif st.session_state.page == "target_record":

    focus = (
        st.session_state.focus
    )

    info = (
        RESONANCES[
            focus
        ]
    )

    st.markdown(
        '<div class="step-label">STEP 3</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="step-title">'
        f'{focus} 공명 발성'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="guide-box">
<b>{focus} 공명을 의도해 /아/를 발성하세요.</b><br>
기준 발성과 같은 음높이를 유지하세요.
{info['guide']}
</div>
""",
        unsafe_allow_html=True,
    )

    audio = five_second_recorder(
        "target_"
        +
        str(
            st.session_state
            .target_recorder_id
        )
    )

    if audio:

        st.session_state.target_audio = (
            audio
        )

        st.session_state.page = (
            "target_review"
        )

        st.rerun()


elif st.session_state.page == "target_review":

    focus = (
        st.session_state.focus
    )

    duration = audio_duration(
        st.session_state
        .target_audio
    )

    st.markdown(
        '<div class="step-label">STEP 3</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="step-title">'
        f'{focus} 공명 발성 확인'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="complete-box">
<div class="complete-title">✅ 녹음 완료</div>
<div class="complete-time">
녹음 시간 · {duration:.1f}초
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.audio(
        st.session_state
        .target_audio,
        format="audio/wav"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        if st.button(
            "🔄 다시 녹음",
            use_container_width=True
        ):

            st.session_state.target_audio = (
                None
            )

            st.session_state.target_recorder_id += (
                1
            )

            st.session_state.page = (
                "target_record"
            )

            st.rerun()

    with col2:

        if st.button(
            "분석하기 ➡️",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = (
                "result"
            )

            st.rerun()


elif st.session_state.page == "result":

    focus = (
        st.session_state.focus
    )

    st.markdown(
        '<div class="step-label">STEP 4</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="step-title">'
        f'{focus} 공명 결과'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:

        with st.spinner(
            "기준 발성과 비교하고 있습니다..."
        ):

            baseline = analyze_audio(
                st.session_state
                .baseline_audio
            )

            target = analyze_audio(
                st.session_state
                .target_audio
            )

            result = judge_resonance(
                focus,
                baseline,
                target
            )

        score = (
            result[
                "score"
            ]
        )

        score_text = (
            f"+{score:.1f}%"
            if score >= 0
            else
            f"{score:.1f}%"
        )

        st.markdown(
            f"""
<div class="{result['css']}">
<div class="result-title">
{result['title']}
</div>
<div class="result-value">
{score_text}
</div>
<div class="result-description">
{result['message']}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        current_index = max(
            0,
            100 + score
        )

        st.markdown(
            f"### {focus} 공명 변화 지수"
        )

        col1, col2 = (
            st.columns(2)
        )

        with col1:

            st.markdown(
                """
<div class="compare-card">
<div class="compare-label">
기준 발성
</div>
<div class="compare-value">
100
</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col2:

            change_class = (
                "compare-change-up"
                if score >= 0
                else
                "compare-change-down"
            )

            arrow = (
                "↑"
                if score >= 0
                else
                "↓"
            )

            st.markdown(
                f"""
<div class="compare-card">
<div class="compare-label">
{focus} 공명 발성
</div>
<div class="compare-value">
{current_index:.1f}
</div>
<div class="{change_class}">
{arrow} {score:+.1f}
</div>
</div>
""",
                unsafe_allow_html=True,
            )

        model = (
            MODELS[
                focus
            ]
        )

        st.markdown(
            "### 왜 이렇게 판단했나요?"
        )

        st.markdown(
            '<div class="reason-box">',
            unsafe_allow_html=True,
        )

        primary = (
            result[
                "primary"
            ]
        )

        secondary = (
            result[
                "secondary"
            ]
        )

        primary_icon = (
            "✅"
            if primary > 3
            else
            "❌"
            if primary < -3
            else
            "➖"
        )

        secondary_icon = (
            "✅"
            if secondary > 3
            else
            "❌"
            if secondary < -3
            else
            "➖"
        )

        st.write(
            primary_icon
            +
            f" **{model['primary_label']}** 변화: "
            f"**{primary:+.1f}%**"
        )

        st.write(
            secondary_icon
            +
            f" **{model['secondary_label']}** 변화: "
            f"**{secondary:+.1f}%**"
        )

        pitch_icon = (
            "✅"
            if result[
                "confidence"
            ] == "높음"
            else
            "⚠️"
        )

        st.write(
            pitch_icon
            +
            " "
            +
            result[
                "confidence_msg"
            ]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "### 결과 신뢰도"
        )

        st.markdown(
            f"""
<div class="confidence-box">
<b>{result['confidence']}</b><br>
{result['confidence_msg']}
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            "### 다음 발성에서 해볼 것"
        )

        st.markdown(
            f"""
<div class="tip-box">
{result['tip']}
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander(
            "연구자용 상세 데이터"
        ):

            st.write(
                f"선택 공명: {focus}"
            )

            st.write(
                f"기준 F0: "
                f"{baseline['f0']:.1f} Hz / "
                f"훈련 F0: "
                f"{target['f0']:.1f} Hz"
            )

            st.write(
                f"음높이 차이: "
                f"{result['f0_change']:.2f}%"
            )

            bands = [
                (
                    "80–500 Hz",
                    "band_80_500"
                ),
                (
                    "500–1500 Hz",
                    "band_500_1500"
                ),
                (
                    "1500–3000 Hz",
                    "band_1500_3000"
                ),
                (
                    "3000–5000 Hz",
                    "band_3000_5000"
                ),
            ]

            for label, key in bands:

                st.write(
                    f"{label} · "
                    f"기준 "
                    f"{baseline[key]:.2f}% / "
                    f"훈련 "
                    f"{target[key]:.2f}% / "
                    f"변화 "
                    f"{target[key] - baseline[key]:+.2f}%p"
                )

            st.write(
                f"Spectral Centroid · "
                f"기준 "
                f"{baseline['centroid']:.1f} Hz / "
                f"훈련 "
                f"{target['centroid']:.1f} Hz"
            )

            st.write(
                f"1차 {focus} 공명 변화 지수: "
                f"{result['score']:+.2f}"
            )

            st.caption(
                "현재 5개 공명 판정식은 "
                "개인 기준 발성 대비 상대 변화로 만든 "
                "1차 휴리스틱입니다. "
                "직접 청지각 테스트 결과를 바탕으로 "
                "가중치와 임계값을 조정해야 합니다."
            )

        st.divider()

        col1, col2 = (
            st.columns(2)
        )

        with col1:

            if st.button(
                f"🎙️ {focus} 다시 해보기",
                use_container_width=True
            ):

                st.session_state.target_audio = (
                    None
                )

                st.session_state.target_recorder_id += (
                    1
                )

                st.session_state.page = (
                    "target_record"
                )

                st.rerun()

        with col2:

            if st.button(
                "다른 공명 선택",
                type="primary",
                use_container_width=True
            ):

                st.session_state.baseline_audio = (
                    None
                )

                st.session_state.target_audio = (
                    None
                )

                st.session_state.baseline_recorder_id += (
                    1
                )

                st.session_state.target_recorder_id += (
                    1
                )

                st.session_state.page = (
                    "select"
                )

                st.rerun()

    except Exception as error:

        st.error(
            f"분석 중 오류가 발생했습니다: {error}"
        )

        if st.button(
            f"🎙️ {focus} 다시 녹음"
        ):

            st.session_state.target_audio = (
                None
            )

            st.session_state.target_recorder_id += (
                1
            )

            st.session_state.page = (
                "target_record"
            )

            st.rerun()
