import base64
import tempfile

import librosa
import numpy as np
import pandas as pd
import parselmouth
import streamlit as st
from parselmouth.praat import call


st.set_page_config(
    page_title="남성 공명 훈련 베타",
    page_icon="🎙️",
    layout="centered",
)


st.markdown(
    """
<style>
.block-container {
    max-width: 720px;
    padding-top: 0.6rem;
    padding-bottom: 1.2rem;
}
.main-title {
    font-size: 1.75rem;
    font-weight: 850;
    letter-spacing: -0.04em;
    line-height: 1.15;
    margin-bottom: 0.15rem;
}
.sub-title {
    color: #666;
    font-size: 0.88rem;
    line-height: 1.45;
    margin-bottom: 0.7rem;
}
.beta {
    display:inline-block;
    font-size:0.72rem;
    font-weight:800;
    padding:3px 7px;
    border-radius:999px;
    background:#eef4ff;
    color:#365f9d;
    margin-bottom:8px;
}
.card {
    border:1px solid #e2e5e8;
    border-radius:14px;
    padding:13px 14px;
    margin-bottom:8px;
    background:#fafbfc;
}
.target-title {
    font-size:1.45rem;
    font-weight:850;
}
.guide {
    background:#f5f6f8;
    border-radius:13px;
    padding:10px 12px;
    font-size:0.86rem;
    line-height:1.45;
    margin-bottom:8px;
}
.result-good {
    background:#eaf7ef;
    border:1px solid #b8e1c6;
    border-radius:15px;
    padding:16px;
    margin:8px 0 12px;
}
.result-mid {
    background:#fff8e8;
    border:1px solid #ead7a7;
    border-radius:15px;
    padding:16px;
    margin:8px 0 12px;
}
.result-bad {
    background:#fff0f0;
    border:1px solid #efc0c0;
    border-radius:15px;
    padding:16px;
    margin:8px 0 12px;
}
.result-hold {
    background:#f2f3f5;
    border:1px solid #d4d7db;
    border-radius:15px;
    padding:16px;
    margin:8px 0 12px;
}
.result-big {
    font-size:1.65rem;
    font-weight:900;
    margin:4px 0;
}
.small {
    color:#6d7278;
    font-size:0.82rem;
    line-height:1.4;
}
.bar-row {
    margin:8px 0 11px;
}
.bar-label {
    display:flex;
    justify-content:space-between;
    font-size:0.87rem;
    font-weight:700;
    margin-bottom:3px;
}
.bar-bg {
    height:10px;
    background:#eceff2;
    border-radius:999px;
    overflow:hidden;
}
.bar-fill {
    height:100%;
    background:#5b6f91;
    border-radius:999px;
}
.tip {
    background:#f7f8fa;
    border-radius:12px;
    padding:11px 12px;
    font-size:0.87rem;
    line-height:1.45;
    margin-top:8px;
}
@media(max-width:600px) {
    .block-container {
        padding-top:0.45rem;
        padding-left:0.8rem;
        padding-right:0.8rem;
        padding-bottom:1rem;
    }
    .main-title { font-size:1.45rem; }
    .sub-title { font-size:0.82rem; }
    .card { padding:10px 11px; }
    .target-title { font-size:1.28rem; }
    .guide { padding:9px 10px; font-size:0.81rem; }
    .result-good,
    .result-mid,
    .result-bad,
    .result-hold {
        padding:13px;
    }
    button {
        min-height:42px !important;
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
]


INFO = {
    "가슴": {
        "icon": "🫁",
        "guide": (
            "다른 공명은 최대한 배제하고 "
            "가슴 쪽 울림을 분명하게 만들어 /아/를 발성하세요."
        ),
        "tip": (
            "가슴 쪽 울림이 더 분명하게 느껴지도록 "
            "다시 발성해보세요."
        ),
    },

    "입천장": {
        "icon": "👄",
        "guide": (
            "다른 공명은 최대한 배제하고 "
            "입천장·구강 공간의 울림을 분명하게 만들어 /아/를 발성하세요."
        ),
        "tip": (
            "입천장과 구강 공간에 울림이 모이는 감각을 "
            "다시 확인해보세요."
        ),
    },

    "이빨·전방": {
        "icon": "🦷",
        "guide": (
            "다른 공명은 최대한 배제하고 "
            "윗니와 입 앞쪽에 소리가 모이도록 /아/를 발성하세요."
        ),
        "tip": (
            "소리가 윗니와 입 앞쪽으로 더 선명하게 모이도록 "
            "다시 발성해보세요."
        ),
    },

    "비강": {
        "icon": "👃",
        "guide": (
            "다른 공명은 최대한 배제하고 "
            "코 주변과 얼굴 중앙의 울림을 분명하게 만들어 /아/를 발성하세요."
        ),
        "tip": (
            "코 주변과 얼굴 중앙의 울림이 더 분명한지 확인하며 "
            "다시 발성해보세요."
        ),
    },
}


# F0는 측정하지만 분류에는 사용하지 않음
FEATURES = [
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "hnr_db",
    "spectral_centroid_hz",
    "spectral_tilt",
    "band_80_500_pct",
    "band_500_1500_pct",
    "band_1500_3000_pct",
    "band_3000_5000_pct",
]


# =========================================================
# 4공명 기준 모델
# 두개골 완전 제외
# =========================================================

CENTROIDS = {

    "가슴": {
        "f1_hz": 449.81686886925206,
        "f2_hz": 1038.30057387432,
        "f3_hz": 2564.769347715297,
        "hnr_db": 23.732902532301974,
        "spectral_centroid_hz": 1686.6313884693777,
        "spectral_tilt": -37.4864777108758,
        "band_80_500_pct": 93.78470357259114,
        "band_500_1500_pct": 5.996769759390089,
        "band_1500_3000_pct": 0.20982275282343227,
        "band_3000_5000_pct": 0.008704635030072544,
    },

    "입천장": {
        "f1_hz": 454.05227257076325,
        "f2_hz": 920.0919103430733,
        "f3_hz": 2594.9459247549926,
        "hnr_db": 33.92425150488846,
        "spectral_centroid_hz": 1249.6360049461298,
        "spectral_tilt": -31.7010433314199,
        "band_80_500_pct": 88.70456205095563,
        "band_500_1500_pct": 10.88910801070077,
        "band_1500_3000_pct": 0.39302450844219755,
        "band_3000_5000_pct": 0.013303857820574135,
    },

    "이빨·전방": {
        "f1_hz": 406.2235400787908,
        "f2_hz": 2123.982267170569,
        "f3_hz": 2918.0947226576372,
        "hnr_db": 24.45499149753048,
        "spectral_centroid_hz": 2109.2467772720097,
        "spectral_tilt": -17.040266756071862,
        "band_80_500_pct": 86.66817813449435,
        "band_500_1500_pct": 3.1495792269706726,
        "band_1500_3000_pct": 6.982677539189656,
        "band_3000_5000_pct": 3.199559701813592,
    },

    "비강": {
        "f1_hz": 255.4618998905658,
        "f2_hz": 1661.51094371261,
        "f3_hz": 2260.9144028566243,
        "hnr_db": 40.76973829274046,
        "spectral_centroid_hz": 1049.7815441814612,
        "spectral_tilt": -27.336919138839512,
        "band_80_500_pct": 99.63513692220052,
        "band_500_1500_pct": 0.23184568186600998,
        "band_1500_3000_pct": 0.11247787831558116,
        "band_3000_5000_pct": 0.020538966740584978,
    },
}


# 각 공명 반복 측정에서 실제로 나타난 변동폭
POOLED_STD = {
    "f1_hz": 33.1148377360414,
    "f2_hz": 93.48950964018955,
    "f3_hz": 82.9851860269652,
    "hnr_db": 1.9782225366868083,
    "spectral_centroid_hz": 230.7192962474501,
    "spectral_tilt": 3.0352007554915277,
    "band_80_500_pct": 4.053594853647912,
    "band_500_1500_pct": 4.066813548561021,
    "band_1500_3000_pct": 1.3700699238121068,
    "band_3000_5000_pct": 0.411534036084007,
}


# =========================================================
# SESSION
# =========================================================

defaults = {
    "page": "select",
    "target": None,
    "audio_bytes": None,
    "recorder_id": 0,
    "result": None,
    "history": [],
}


for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# RECORDER
# =========================================================

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
    display:block;
    width:100%;
    height:100%;
    overflow:hidden;
}

.recorder {

    width:100%;
    height:100%;

    box-sizing:border-box;

    border:1px solid #e1e4e8;
    border-radius:17px;

    display:flex;
    flex-direction:column;

    align-items:center;
    justify-content:center;

    padding:12px 10px;

    overflow:hidden;

    background:var(--st-background-color);

    font-family:var(--st-font);
}


.record-button {

    width:78px;
    height:78px;

    flex:0 0 78px;

    border:none;
    border-radius:50%;

    background:#fff;

    display:flex;
    align-items:center;
    justify-content:center;

    cursor:pointer;

    box-shadow:
        0 4px 14px rgba(0,0,0,.13),
        inset 0 0 0 1px rgba(0,0,0,.06);
}


.record-button:active {
    transform:scale(.95);
}


.mic-icon {
    font-size:34px;
    line-height:1;
}


.record-button.recording {

    background:#e53935;

    animation:pulse 1.05s infinite;
}


.record-button.recording .mic-icon {
    font-size:0;
}


.record-button.recording .mic-icon::after {

    content:"■";

    font-size:25px;

    color:white;
}


.record-button.done {
    background:#e8f6ed;
}


.record-button.done .mic-icon {
    font-size:0;
}


.record-button.done .mic-icon::after {

    content:"✓";

    font-size:36px;

    font-weight:800;

    color:#269451;
}


.status {

    margin-top:8px;

    font-size:14px;

    font-weight:750;

    color:var(--st-text-color);
}


.status.recording {
    color:#d32f2f;
}


.status.done {
    color:#25834a;
}


.timer {

    margin-top:3px;

    font-size:25px;

    font-weight:850;

    font-variant-numeric:tabular-nums;

    color:var(--st-text-color);
}


.progress-wrap {

    width:min(300px,80%);

    height:5px;

    margin-top:8px;

    background:#eceff2;

    border-radius:999px;

    overflow:hidden;
}


.progress {

    width:0%;

    height:100%;

    background:#e53935;

    border-radius:999px;
}


.help {

    margin-top:6px;

    font-size:10.5px;

    color:#7a7f87;

    text-align:center;
}


@keyframes pulse {

    0% {
        box-shadow:0 0 0 0 rgba(229,57,53,.28);
    }

    70% {
        box-shadow:0 0 0 12px rgba(229,57,53,0);
    }

    100% {
        box-shadow:0 0 0 0 rgba(229,57,53,0);
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
            Math.min(
                ms,
                RECORD_MS
            )
            /
            1000;

        return "00:"
            +
            sec
            .toFixed(1)
            .padStart(
                4,
                "0"
            );
    }


    function mergeBuffers(parts) {

        let total = 0;

        for (const part of parts) {

            total +=
                part.length;
        }


        const merged =
            new Float32Array(
                total
            );


        let offset = 0;


        for (const part of parts) {

            merged.set(
                part,
                offset
            );

            offset +=
                part.length;
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
                44
                +
                samples.length * 2
            );


        const view =
            new DataView(
                buffer
            );


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
                type:"audio/wav"
            }
        );
    }


    function blobToBase64(blob) {

        return new Promise(
            (
                resolve,
                reject
            ) => {

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


                reader.readAsDataURL(
                    blob
                );
            }
        );
    }


    function updateTimer() {

        if (!recording) {
            return;
        }


        const elapsed =
            performance.now()
            -
            startTime;


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
                capped
                /
                RECORD_MS
                *
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


        recording =
            false;


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
                sampleRate
                *
                5
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

                        echoCancellation:
                            false,

                        noiseSuppression:
                            false,

                        autoGainControl:
                            false,

                        channelCount:
                            1
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


            silentGain.gain.value =
                0;


            buffers = [];


            processor.onaudioprocess =
                (event) => {

                    if (!recording) {
                        return;
                    }


                    const input =
                        event
                        .inputBuffer
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


            recording =
                true;


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
    "student_resonance_recorder_v1",
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


# =========================================================
# AUDIO ANALYSIS
# =========================================================

def safe_float(value):

    try:

        value = float(
            value
        )


        if np.isfinite(
            value
        ):

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


    return float(
        np.polyfit(
            x,
            y,
            1
        )[0]
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


    values = []


    for t in np.linspace(
        start,
        end,
        20
    ):

        try:

            value = safe_float(
                call(
                    formant,
                    "Get value at time",
                    number,
                    float(t),
                    "Hertz",
                    "Linear",
                )
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


        y_trim, _ = librosa.effects.trim(
            y,
            top_db=35
        )


        voice_duration = (
            len(y_trim)
            /
            sr
        )


        if voice_duration < 2:

            raise ValueError(
                "유효한 발성 시간이 너무 짧습니다. "
                "/아/를 충분히 유지해주세요."
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


        f0 = safe_float(
            call(
                pitch,
                "Get mean",
                0,
                0,
                "Hertz"
            )
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


        duration = float(
            call(
                sound,
                "Get total duration"
            )
        )


        f1 = mean_formant(
            formant,
            1,
            duration
        )


        f2 = mean_formant(
            formant,
            2,
            duration
        )


        f3 = mean_formant(
            formant,
            3,
            duration
        )


        harmonicity = call(
            sound,
            "To Harmonicity (cc)",
            0.01,
            75,
            0.1,
            1.0
        )


        hnr = safe_float(
            call(
                harmonicity,
                "Get mean",
                0,
                0
            )
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


        return {

            "voice_duration_sec":
                voice_duration,

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
                spectral_tilt(
                    mean_power,
                    frequencies
                ),

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


# =========================================================
# CLASSIFIER
# =========================================================

def classify(
    features
):

    distances = {}


    for resonance, centroid in CENTROIDS.items():

        z_values = []


        for key in FEATURES:

            value = features[
                key
            ]


            if not np.isfinite(
                value
            ):

                raise ValueError(
                    f"{key} 값을 안정적으로 측정하지 못했습니다."
                )


            sd = max(
                POOLED_STD[
                    key
                ],
                1e-12
            )


            z_values.append(
                (
                    value
                    -
                    centroid[
                        key
                    ]
                )
                /
                sd
            )


        z_values = np.array(
            z_values,
            dtype=float
        )


        distances[
            resonance
        ] = float(
            np.sqrt(
                np.mean(
                    z_values
                    **
                    2
                )
            )
        )


    ordered = sorted(
        distances.items(),
        key=lambda x: x[1]
    )


    raw = {
        name:
            np.exp(
                -distance
            )
        for name, distance in ordered
    }


    total = (
        sum(
            raw.values()
        )
        +
        1e-12
    )


    similarity = {
        name:
            raw[
                name
            ]
            /
            total
            *
            100
        for name in raw
    }


    return (
        ordered,
        similarity
    )


def build_result(
    target,
    ordered,
    similarity
):

    prediction = (
        ordered[
            0
        ][
            0
        ]
    )


    d1 = (
        ordered[
            0
        ][
            1
        ]
    )


    d2 = (
        ordered[
            1
        ][
            1
        ]
    )


    target_rank = (
        [
            name
            for name, _
            in ordered
        ]
        .index(
            target
        )
        +
        1
    )


    target_distance = dict(
        ordered
    )[
        target
    ]


    margin = (
        (
            d2
            -
            d1
        )
        /
        max(
            d2,
            1e-9
        )
    )


    out_of_reference = (
        d1 > 3.0
    )


    if out_of_reference:

        status = (
            "판정 보류"
        )


        css = (
            "result-hold"
        )


        message = (
            "현재 기준 데이터와 차이가 커서 "
            "공명을 자신 있게 판정하기 어렵습니다."
        )


    elif prediction == target:

        if margin >= 0.25:

            status = (
                "잘 되고 있습니다"
            )


            css = (
                "result-good"
            )


            message = (
                f"현재 발성은 {target} 공명에 "
                "가장 뚜렷하게 가깝습니다."
            )


        else:

            status = (
                "방향은 맞습니다"
            )


            css = (
                "result-mid"
            )


            message = (
                f"{target} 공명이 가장 가깝지만 "
                "다른 공명과의 경계가 아직 크지 않습니다."
            )


    else:

        if (
            target_rank == 2
            and
            target_distance
            <=
            d1 * 1.30
        ):

            status = (
                "방향은 맞습니다"
            )


            css = (
                "result-mid"
            )


            message = (
                f"{target} 공명도 가깝지만 "
                f"현재는 {prediction} 공명이 "
                "조금 더 강하게 나타납니다."
            )


        else:

            status = (
                "다른 공명이 더 가깝습니다"
            )


            css = (
                "result-bad"
            )


            message = (
                f"현재 발성은 목표인 {target}보다 "
                f"{prediction} 공명에 더 가깝게 나타납니다."
            )


    if out_of_reference:

        confidence = (
            "낮음"
        )


    elif margin >= 0.35:

        confidence = (
            "높음"
        )


    elif margin >= 0.15:

        confidence = (
            "보통"
        )


    else:

        confidence = (
            "낮음"
        )


    return {

        "target":
            target,

        "prediction":
            prediction,

        "status":
            status,

        "css":
            css,

        "message":
            message,

        "confidence":
            confidence,

        "target_rank":
            target_rank,

        "nearest_distance":
            d1,

        "margin":
            margin,

        "out_of_reference":
            out_of_reference,

        "similarity":
            similarity,

        "ordered":
            ordered,
    }


def reset_measurement(
    keep_target=True
):

    st.session_state.audio_bytes = (
        None
    )


    st.session_state.result = (
        None
    )


    st.session_state.recorder_id += (
        1
    )


    if (
        keep_target
        and
        st.session_state.target
    ):

        st.session_state.page = (
            "record"
        )


    else:

        st.session_state.target = (
            None
        )


        st.session_state.page = (
            "select"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="beta">
남성용 · 연구용 베타
</div>

<div class="main-title">
🎙️ 공명 훈련 피드백
</div>

<div class="sub-title">
가슴 · 입천장 · 이빨·전방 · 비강 중 하나를 선택하고
/아/를 5초간 발성하세요.
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SELECT
# =========================================================

if st.session_state.page == "select":

    st.markdown(
        """
<div class="card">

<b>연습할 공명을 선택하세요.</b><br>

<span class="small">
두개골 공명은 이번 버전부터 분석 대상에서 제외했습니다.
</span>

</div>
""",
        unsafe_allow_html=True,
    )


    col1, col2 = st.columns(
        2
    )


    with col1:

        if st.button(
            "🫁 가슴",
            use_container_width=True
        ):

            st.session_state.target = (
                "가슴"
            )


            st.session_state.page = (
                "record"
            )


            st.rerun()


        if st.button(
            "🦷 이빨·전방",
            use_container_width=True
        ):

            st.session_state.target = (
                "이빨·전방"
            )


            st.session_state.page = (
                "record"
            )


            st.rerun()


    with col2:

        if st.button(
            "👄 입천장",
            use_container_width=True
        ):

            st.session_state.target = (
                "입천장"
            )


            st.session_state.page = (
                "record"
            )


            st.rerun()


        if st.button(
            "👃 비강",
            use_container_width=True
        ):

            st.session_state.target = (
                "비강"
            )


            st.session_state.page = (
                "record"
            )


            st.rerun()


# =========================================================
# RECORD
# =========================================================

elif st.session_state.page == "record":

    target = (
        st.session_state.target
    )


    info = INFO[
        target
    ]


    st.markdown(
        f"""
<div class="card">

<div class="small">
현재 목표
</div>

<div class="target-title">
{info['icon']} {target}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
<div class="guide">

<b>/아/를 약 5초간 유지하세요.</b><br>

{info['guide']}

</div>
""",
        unsafe_allow_html=True,
    )


    audio = five_second_recorder(
        key=(
            "student_"
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


    if st.button(
        "← 다른 공명 선택",
        use_container_width=True
    ):

        reset_measurement(
            keep_target=False
        )


        st.rerun()


# =========================================================
# REVIEW
# =========================================================

elif st.session_state.page == "review":

    target = (
        st.session_state.target
    )


    st.markdown(
        f"""
<div class="card">

<b>
{INFO[target]['icon']} {target}
</b>

공명 녹음 확인

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

            reset_measurement(
                keep_target=True
            )


            st.rerun()


    with col2:

        if st.button(
            "분석하기",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "공명 특성을 분석하고 있습니다..."
                ):

                    features = analyze_audio(
                        st.session_state.audio_bytes
                    )


                    ordered, similarity = classify(
                        features
                    )


                    result = build_result(
                        target,
                        ordered,
                        similarity
                    )


                    history_row = {

                        "목표":
                            target,

                        "판정":
                            result[
                                "prediction"
                            ],

                        "상태":
                            result[
                                "status"
                            ],

                        "신뢰도":
                            result[
                                "confidence"
                            ],

                        "F0":
                            features[
                                "f0_hz"
                            ],

                        "F1":
                            features[
                                "f1_hz"
                            ],

                        "F2":
                            features[
                                "f2_hz"
                            ],

                        "F3":
                            features[
                                "f3_hz"
                            ],

                        "HNR":
                            features[
                                "hnr_db"
                            ],

                        "Centroid":
                            features[
                                "spectral_centroid_hz"
                            ],

                        "Tilt":
                            features[
                                "spectral_tilt"
                            ],

                        "80_500":
                            features[
                                "band_80_500_pct"
                            ],

                        "500_1500":
                            features[
                                "band_500_1500_pct"
                            ],

                        "1500_3000":
                            features[
                                "band_1500_3000_pct"
                            ],

                        "3000_5000":
                            features[
                                "band_3000_5000_pct"
                            ],
                    }


                    st.session_state.history.append(
                        history_row
                    )


                    st.session_state.result = (
                        result
                    )


                    st.session_state.page = (
                        "result"
                    )


                st.rerun()


            except Exception as error:

                st.error(
                    f"분석 오류: {error}"
                )


# =========================================================
# RESULT
# =========================================================

elif st.session_state.page == "result":

    result = (
        st.session_state.result
    )


    target = (
        result[
            "target"
        ]
    )


    st.markdown(
        f"""
<div class="{result['css']}">

<div style="font-size:.82rem;font-weight:750;">
분석 결과
</div>

<div class="result-big">
{result['status']}
</div>

<div>
{result['message']}
</div>

<div class="small" style="margin-top:7px;">
판정 신뢰도 · <b>{result['confidence']}</b>
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        "#### 4개 공명 상대 유사도"
    )


    ordered_similarity = sorted(
        result[
            "similarity"
        ].items(),
        key=lambda x: x[1],
        reverse=True
    )


    for (
        name,
        value
    ) in ordered_similarity:

        icon = INFO[
            name
        ][
            "icon"
        ]


        st.markdown(
            f"""
<div class="bar-row">

<div class="bar-label">

<span>
{icon} {name}
</span>

<span>
{value:.0f}
</span>

</div>

<div class="bar-bg">

<div
class="bar-fill"
style="width:{min(max(value, 0), 100):.1f}%">
</div>

</div>

</div>
""",
            unsafe_allow_html=True,
        )


    if (
        result[
            "prediction"
        ]
        !=
        target
    ):

        st.markdown(
            f"""
<div class="tip">

<b>
다음 발성
</b>

<br>

{INFO[target]['tip']}

</div>
""",
            unsafe_allow_html=True,
        )


    st.caption(
        "위 숫자는 공명이 존재할 확률이 아니라 "
        "현재 4개 기준 패턴 사이의 상대적 유사도를 "
        "보기 쉽게 환산한 값입니다."
    )


    if result[
        "out_of_reference"
    ]:

        st.warning(
            "현재 모델은 남성 연구용 1차 기준 데이터로 만든 베타입니다. "
            "기준 범위를 크게 벗어난 음성은 억지로 정답을 내지 않고 "
            "판정을 보류합니다."
        )


    col1, col2 = st.columns(
        2
    )


    with col1:

        if st.button(
            "🔄 같은 공명 다시",
            use_container_width=True
        ):

            reset_measurement(
                keep_target=True
            )


            st.rerun()


    with col2:

        if st.button(
            "다른 공명 선택",
            type="primary",
            use_container_width=True
        ):

            reset_measurement(
                keep_target=False
            )


            st.rerun()


# =========================================================
# RESEARCH LOG
# =========================================================

if st.session_state.history:

    with st.expander(
        "연구용 측정 기록"
    ):

        history_df = pd.DataFrame(
            st.session_state.history
        )


        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


        csv_bytes = (
            history_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        )


        st.download_button(
            "측정 기록 CSV 다운로드",
            data=csv_bytes,
            file_name="male_student_resonance_results.csv",
            mime="text/csv",
            use_container_width=True
        )
