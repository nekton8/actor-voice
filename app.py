import base64
import io
import tempfile

import librosa
import numpy as np
import pandas as pd
import parselmouth
import streamlit as st
from parselmouth.praat import call


st.set_page_config(
    page_title="공명 캘리브레이션",
    page_icon="🎙️",
    layout="centered",
)


st.markdown(
    """
<style>
.block-container {
    max-width: 720px;
    padding-top: 0.8rem;
    padding-bottom: 1.5rem;
}

.main-title {
    font-size: 1.9rem;
    font-weight: 850;
    letter-spacing: -0.04em;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #6c6c6c;
    font-size: 0.92rem;
    line-height: 1.45;
    margin-bottom: 1rem;
}

.progress-card {
    border: 1px solid #e1e5e9;
    background: #fafbfc;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

.progress-title {
    font-size: 0.8rem;
    color: #73777d;
    font-weight: 700;
}

.progress-value {
    font-size: 1.45rem;
    font-weight: 850;
    margin-top: 2px;
}

.guide-box {
    background: #f5f6f8;
    border-radius: 13px;
    padding: 12px 15px;
    margin: 0 0 10px 0;
    font-size: 0.9rem;
    line-height: 1.48;
}

.success-box {
    background: #edf8f1;
    border: 1px solid #c9e7d3;
    border-radius: 13px;
    padding: 13px 15px;
    margin: 10px 0;
}

.done-box {
    background: #eef4ff;
    border: 1px solid #cad8ef;
    border-radius: 15px;
    padding: 18px;
    margin: 10px 0 16px 0;
}

.metric-mini {
    text-align: center;
    border: 1px solid #e3e5e8;
    border-radius: 12px;
    padding: 11px 5px;
}

.metric-label {
    font-size: 0.72rem;
    color: #747980;
}

.metric-value {
    font-size: 1.25rem;
    font-weight: 800;
}

@media (max-width: 600px) {

    .block-container {
        padding-top: 0.55rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-bottom: 1rem;
    }

    .main-title {
        font-size: 1.55rem;
    }

    .sub-title {
        font-size: 0.84rem;
        margin-bottom: 0.7rem;
    }

    .progress-card {
        padding: 11px 13px;
        margin-bottom: 8px;
    }

    .progress-value {
        font-size: 1.25rem;
    }

    .guide-box {
        padding: 10px 12px;
        font-size: 0.84rem;
        line-height: 1.4;
        margin-bottom: 8px;
    }

    .success-box {
        padding: 10px 12px;
        font-size: 0.86rem;
    }

    .done-box {
        padding: 15px;
    }

    button {
        min-height: 40px !important;
    }

    div[data-testid="stAudio"] {
        margin-top: 0 !important;
        margin-bottom: 0.4rem !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


RESONANCES = [
    "가슴",
    "입천장",
    "이빨·전방",
    "비강",
    "두개골",
]


RESONANCE_INFO = {
    "가슴": {
        "icon": "🫁",
        "guide": "다른 공명은 최대한 배제하고 가슴 공명만 분명하게 만들어 /아/를 발성하세요.",
    },
    "입천장": {
        "icon": "👄",
        "guide": "다른 공명은 최대한 배제하고 입천장·구강 공간의 울림만 분명하게 만들어 /아/를 발성하세요.",
    },
    "이빨·전방": {
        "icon": "🦷",
        "guide": "다른 공명은 최대한 배제하고 윗니와 입 앞쪽에 소리가 모이도록 /아/를 발성하세요.",
    },
    "비강": {
        "icon": "👃",
        "guide": "다른 공명은 최대한 배제하고 코 주변과 얼굴 중앙의 울림만 분명하게 만들어 /아/를 발성하세요.",
    },
    "두개골": {
        "icon": "💀",
        "guide": "다른 공명은 최대한 배제하고 머리 위쪽에서 느껴지는 울림만 분명하게 만들어 /아/를 발성하세요.",
    },
}


REPEATS = 5


defaults = {
    "focus_index": 0,
    "repeat_index": 0,
    "records": [],
    "audio_bytes": None,
    "recorder_id": 0,
    "page": "record",
}


for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


RECORDER_HTML = """
<div class="recorder">

    <button id="recordButton" class="record-button">
        <div class="mic-icon">🎙</div>
    </button>

    <div id="statusText" class="status">
        녹음 시작
    </div>

    <div id="timerText" class="timer">
        00:00.0
    </div>

    <div class="progress-wrap">
        <div id="progressBar" class="progress"></div>
    </div>

    <div id="helpText" class="help">
        누르면 5초간 자동 녹음됩니다.
    </div>

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

    transition:
        transform .15s ease,
        background .15s ease;
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
    color: #7a7f87;
    text-align: center;
}

@keyframes pulse {

    0% {
        box-shadow: 0 0 0 0 rgba(229,57,53,.28);
    }

    70% {
        box-shadow: 0 0 0 12px rgba(229,57,53,0);
    }

    100% {
        box-shadow: 0 0 0 0 rgba(229,57,53,0);
    }
}
"""


RECORDER_JS = r"""
export default function(component) {

    const {
        parentElement,
        setStateValue
    } = component;

    const button =
        parentElement.querySelector("#recordButton");

    const statusText =
        parentElement.querySelector("#statusText");

    const timerText =
        parentElement.querySelector("#timerText");

    const progressBar =
        parentElement.querySelector("#progressBar");

    const helpText =
        parentElement.querySelector("#helpText");

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

        const sec =
            Math.min(ms, RECORD_MS) / 1000;

        return "00:" +
            sec.toFixed(1).padStart(4, "0");
    }


    function mergeBuffers(parts) {

        let total = 0;

        for (const part of parts) {
            total += part.length;
        }

        const merged =
            new Float32Array(total);

        let offset = 0;

        for (const part of parts) {

            merged.set(
                part,
                offset
            );

            offset += part.length;
        }

        return merged;
    }


    function writeString(
        view,
        offset,
        text
    ) {

        for (
            let i = 0;
            i < text.length;
            i++
        ) {

            view.setUint8(
                offset + i,
                text.charCodeAt(i)
            );
        }
    }


    function encodeWav(
        samples,
        sr
    ) {

        const buffer =
            new ArrayBuffer(
                44 + samples.length * 2
            );

        const view =
            new DataView(buffer);

        writeString(
            view,
            0,
            "RIFF"
        );

        view.setUint32(
            4,
            36 + samples.length * 2,
            true
        );

        writeString(
            view,
            8,
            "WAVE"
        );

        writeString(
            view,
            12,
            "fmt "
        );

        view.setUint32(
            16,
            16,
            true
        );

        view.setUint16(
            20,
            1,
            true
        );

        view.setUint16(
            22,
            1,
            true
        );

        view.setUint32(
            24,
            sr,
            true
        );

        view.setUint32(
            28,
            sr * 2,
            true
        );

        view.setUint16(
            32,
            2,
            true
        );

        view.setUint16(
            34,
            16,
            true
        );

        writeString(
            view,
            36,
            "data"
        );

        view.setUint32(
            40,
            samples.length * 2,
            true
        );

        let offset = 44;

        for (
            let i = 0;
            i < samples.length;
            i++
        ) {

            let sample =
                Math.max(
                    -1,
                    Math.min(
                        1,
                        samples[i]
                    )
                );

            sample =
                sample < 0
                ?
                sample * 0x8000
                :
                sample * 0x7FFF;

            view.setInt16(
                offset,
                sample,
                true
            );

            offset += 2;
        }

        return new Blob(
            [view],
            {
                type: "audio/wav"
            }
        );
    }


    function blobToBase64(blob) {

        return new Promise(
            (resolve, reject) => {

                const reader =
                    new FileReader();

                reader.onloadend =
                    () => {

                        const value =
                            reader.result;

                        resolve(
                            value.substring(
                                value.indexOf(",") + 1
                            )
                        );
                    };

                reader.onerror =
                    reject;

                reader.readAsDataURL(blob);
            }
        );
    }


    function updateTimer() {

        if (!recording) {
            return;
        }

        const elapsed =
            performance.now() - startTime;

        const capped =
            Math.min(
                elapsed,
                RECORD_MS
            );

        timerText.textContent =
            formatTime(
                capped
            );

        progressBar.style.width =
            (
                capped /
                RECORD_MS *
                100
            )
            +
            "%";

        timerFrame =
            requestAnimationFrame(
                updateTimer
            );
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

            for (
                const track
                of
                stream.getTracks()
            ) {

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

        clearTimeout(
            stopTimer
        );

        cancelAnimationFrame(
            timerFrame
        );

        timerText.textContent =
            "00:05.0";

        progressBar.style.width =
            "100%";

        button.classList.remove(
            "recording"
        );

        statusText.classList.remove(
            "recording"
        );

        statusText.textContent =
            "저장 중...";

        helpText.textContent =
            "잠시만 기다려주세요.";

        const merged =
            mergeBuffers(
                buffers
            );

        const targetFrames =
            Math.round(
                sampleRate * 5
            );

        const samples =
            new Float32Array(
                targetFrames
            );

        samples.set(
            merged.subarray(
                0,
                Math.min(
                    merged.length,
                    targetFrames
                )
            )
        );

        const wavBlob =
            encodeWav(
                samples,
                sampleRate
            );

        await cleanup();

        const audioBase64 =
            await blobToBase64(
                wavBlob
            );

        button.classList.add(
            "done"
        );

        statusText.classList.add(
            "done"
        );

        statusText.textContent =
            "녹음 완료";

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
                    window.AudioContext ||
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

            silentGain.gain.value =
                0;

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
                        new Float32Array(
                            input
                        )
                    );
                };

            source.connect(
                processor
            );

            processor.connect(
                silentGain
            );

            silentGain.connect(
                audioContext.destination
            );

            recording = true;

            button.classList.remove(
                "done"
            );

            button.classList.add(
                "recording"
            );

            statusText.classList.remove(
                "done"
            );

            statusText.classList.add(
                "recording"
            );

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

            console.error(
                error
            );

            statusText.textContent =
                "마이크 사용 불가";

            helpText.textContent =
                "브라우저의 마이크 권한을 허용해주세요.";
        }
    }


    button.onclick =
        () => {

            if (!recording) {
                startRecording();
            }
        };


    return () => {

        clearTimeout(
            stopTimer
        );

        cancelAnimationFrame(
            timerFrame
        );

        if (stream) {

            for (
                const track
                of
                stream.getTracks()
            ) {

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
    "calibration_recorder_v2",
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


def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:

        pass

    return np.nan


def band_ratio(
    power,
    frequencies,
    low,
    high
):

    band_mask = (
        (frequencies >= low)
        &
        (frequencies < high)
    )

    total_mask = (
        (frequencies >= 80)
        &
        (frequencies < 5000)
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


def spectral_tilt(
    power,
    frequencies
):

    mask = (
        (frequencies >= 100)
        &
        (frequencies <= 5000)
    )

    freq = frequencies[
        mask
    ]

    pwr = power[
        mask
    ]

    valid = (
        (freq > 0)
        &
        (pwr > 0)
    )

    freq = freq[
        valid
    ]

    pwr = pwr[
        valid
    ]

    if len(freq) < 10:
        return np.nan

    x = np.log10(
        freq
    )

    y = (
        10
        *
        np.log10(
            pwr + 1e-12
        )
    )

    slope = np.polyfit(
        x,
        y,
        1
    )[0]

    return float(
        slope
    )


def mean_formant(
    formant,
    number,
    duration
):

    start = max(
        0.15,
        duration * 0.15
    )

    end = min(
        duration - 0.15,
        duration * 0.85
    )

    if end <= start:
        return np.nan

    times = np.linspace(
        start,
        end,
        20
    )

    values = []

    for t in times:

        try:

            value = call(
                formant,
                "Get value at time",
                number,
                float(t),
                "Hertz",
                "Linear",
            )

            value = safe_float(
                value
            )

            if np.isfinite(
                value
            ):

                values.append(
                    value
                )

        except Exception:

            pass

    if not values:
        return np.nan

    return float(
        np.median(
            values
        )
    )


def analyze_audio(
    audio_bytes
):

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

        raw_duration = (
            len(y)
            /
            sr
        )

        y_trim, _ = (
            librosa.effects.trim(
                y,
                top_db=35
            )
        )

        duration = (
            len(y_trim)
            /
            sr
        )

        if duration < 2:

            raise ValueError(
                "유효한 발성 시간이 너무 짧습니다."
            )

        y_norm = librosa.util.normalize(
            y_trim
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

        f0 = safe_float(
            f0
        )

        formant = call(
            sound,
            "To Formant (burg)",
            0.0,
            5,
            5500,
            0.025,
            50
        )

        sound_duration = float(
            call(
                sound,
                "Get total duration"
            )
        )

        f1 = mean_formant(
            formant,
            1,
            sound_duration
        )

        f2 = mean_formant(
            formant,
            2,
            sound_duration
        )

        f3 = mean_formant(
            formant,
            3,
            sound_duration
        )

        harmonicity = call(
            sound,
            "To Harmonicity (cc)",
            0.01,
            75,
            0.1,
            1.0
        )

        hnr = call(
            harmonicity,
            "Get mean",
            0,
            0
        )

        hnr = safe_float(
            hnr
        )

        spectrum = np.abs(
            librosa.stft(
                y_norm,
                n_fft=2048,
                hop_length=512
            )
        )

        power = (
            spectrum
            **
            2
        )

        mean_power = np.mean(
            power,
            axis=1
        )

        frequencies = (
            librosa.fft_frequencies(
                sr=sr,
                n_fft=2048
            )
        )

        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y_norm,
                    sr=sr
                )
            )
        )

        tilt = spectral_tilt(
            mean_power,
            frequencies
        )

        return {

            "raw_duration_sec":
                raw_duration,

            "voice_duration_sec":
                duration,

            "f0_hz":
                f0,

            "f1_hz":
                f1,

            "f2_hz":
                f2,

            "f3_hz":
                f3,

            "hnr_db":
                hnr,

            "spectral_centroid_hz":
                centroid,

            "spectral_tilt":
                tilt,

            "band_80_500_pct":
                band_ratio(
                    mean_power,
                    frequencies,
                    80,
                    500
                ),

            "band_500_1500_pct":
                band_ratio(
                    mean_power,
                    frequencies,
                    500,
                    1500
                ),

            "band_1500_3000_pct":
                band_ratio(
                    mean_power,
                    frequencies,
                    1500,
                    3000
                ),

            "band_3000_5000_pct":
                band_ratio(
                    mean_power,
                    frequencies,
                    3000,
                    5000
                ),
        }


def current_focus():

    return RESONANCES[
        st.session_state.focus_index
    ]


def total_recorded():

    return len(
        st.session_state.records
    )


def total_target():

    return (
        len(RESONANCES)
        *
        REPEATS
    )


def next_measurement():

    st.session_state.audio_bytes = None

    st.session_state.recorder_id += 1

    if (
        st.session_state.repeat_index
        <
        REPEATS - 1
    ):

        st.session_state.repeat_index += 1

    else:

        st.session_state.repeat_index = 0

        st.session_state.focus_index += 1

    if (
        st.session_state.focus_index
        >=
        len(RESONANCES)
    ):

        st.session_state.page = (
            "complete"
        )

    else:

        st.session_state.page = (
            "record"
        )


def reset_all():

    st.session_state.focus_index = 0

    st.session_state.repeat_index = 0

    st.session_state.records = []

    st.session_state.audio_bytes = None

    st.session_state.recorder_id += 1

    st.session_state.page = "record"


st.markdown(
    """
<div class="main-title">
🎙️ 남성 공명 캘리브레이션
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="sub-title">
5가지 공명을 각각 5회씩 녹음합니다.
현재 단계에서는 성공 여부를 판정하지 않고
공명별 음향 특징을 수집합니다.
</div>
""",
    unsafe_allow_html=True,
)


if st.session_state.page == "record":

    focus = current_focus()

    info = RESONANCE_INFO[
        focus
    ]

    repeat_no = (
        st.session_state.repeat_index
        +
        1
    )

    completed = total_recorded()

    total = total_target()

    st.progress(
        completed / total
    )

    st.markdown(
        f"""
<div class="progress-card">
<div class="progress-title">
전체 진행
</div>

<div class="progress-value">
{completed} / {total}
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="progress-card">
<div class="progress-title">
현재 측정
</div>

<div class="progress-value">
{info['icon']} {focus} · {repeat_no} / {REPEATS}
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="guide-box">
<b>/아/를 약 5초간 유지하세요.</b><br>
{info['guide']}<br>
가능하면 매번 비슷한 음높이와 비슷한 크기로 발성하세요.
</div>
""",
        unsafe_allow_html=True,
    )

    audio = five_second_recorder(
        key=(
            "calibration_"
            +
            str(
                st.session_state.recorder_id
            )
        )
    )

    if audio:

        st.session_state.audio_bytes = (
            audio
        )

        st.session_state.page = (
            "review"
        )

        st.rerun()


elif st.session_state.page == "review":

    focus = current_focus()

    repeat_no = (
        st.session_state.repeat_index
        +
        1
    )

    st.markdown(
        f"""
<div class="progress-card">
<div class="progress-title">
녹음 확인
</div>

<div class="progress-value">
{focus} · {repeat_no} / {REPEATS}
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.audio(
        st.session_state.audio_bytes,
        format="audio/wav"
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        if st.button(
            "🔄 다시 녹음",
            use_container_width=True
        ):

            st.session_state.audio_bytes = (
                None
            )

            st.session_state.recorder_id += (
                1
            )

            st.session_state.page = (
                "record"
            )

            st.rerun()

    with col2:

        if st.button(
            "이 녹음 저장",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "음향 분석 중..."
                ):

                    features = analyze_audio(
                        st.session_state.audio_bytes
                    )

                record = {
                    "resonance":
                        focus,

                    "repeat":
                        repeat_no,

                    **features,
                }

                st.session_state.records.append(
                    record
                )

                st.session_state.page = (
                    "saved"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"분석 오류: {e}"
                )


elif st.session_state.page == "saved":

    latest = (
        st.session_state.records[
            -1
        ]
    )

    focus = latest[
        "resonance"
    ]

    repeat_no = latest[
        "repeat"
    ]

    st.markdown(
        f"""
<div class="success-box">
<b>✅ {focus} {repeat_no}/5 저장 완료</b><br>
음향 데이터가 정상적으로 기록되었습니다.
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    with col1:

        st.markdown(
            f"""
<div class="metric-mini">
<div class="metric-label">
F0
</div>

<div class="metric-value">
{latest['f0_hz']:.0f}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
<div class="metric-mini">
<div class="metric-label">
F1
</div>

<div class="metric-value">
{latest['f1_hz']:.0f}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
<div class="metric-mini">
<div class="metric-label">
F2
</div>

<div class="metric-value">
{latest['f2_hz']:.0f}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if st.button(
        "다음 측정 ➡️",
        type="primary",
        use_container_width=True
    ):

        next_measurement()

        st.rerun()


elif st.session_state.page == "complete":

    df = pd.DataFrame(
        st.session_state.records
    )

    st.markdown(
        f"""
<div class="done-box">

<b style="font-size:1.2rem;">
✅ 캘리브레이션 완료
</b>

<br><br>

가슴 · 5회<br>
입천장 · 5회<br>
이빨·전방 · 5회<br>
비강 · 5회<br>
두개골 · 5회

<br><br>

총 <b>{len(df)}개</b>의 음성 데이터가 수집되었습니다.

</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader(
        "측정 데이터"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = (
        df.to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )

    st.download_button(
        "📥 CSV 다운로드",
        data=csv_bytes,
        file_name="male_resonance_calibration.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "CSV를 다운로드한 뒤 이 대화에 올려주세요. "
        "그 데이터를 바탕으로 5개 공명을 실제로 구분하는 변수를 분석합니다."
    )

    if st.button(
        "전체 다시 측정",
        use_container_width=True
    ):

        reset_all()

        st.rerun()
