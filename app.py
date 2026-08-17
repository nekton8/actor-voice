import base64
import json
import tempfile
from datetime import datetime

import librosa
import numpy as np
import pandas as pd
import parselmouth
import streamlit as st
from parselmouth.praat import call


st.set_page_config(
    page_title="공명 훈련 피드백",
    page_icon="🎙️",
    layout="centered",
)


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
<style>
.block-container {
    max-width: 720px;
    padding-top: 1.35rem;
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
    margin-bottom: 0.75rem;
}

.beta {
    display: inline-block;
    font-size: 0.71rem;
    font-weight: 800;
    padding: 3px 7px;
    border-radius: 999px;
    background: #eef4ff;
    color: #365f9d;
    margin-bottom: 8px;
}

.card {
    border: 1px solid #e2e5e8;
    border-radius: 14px;
    padding: 13px 14px;
    margin-bottom: 8px;
    background: #fafbfc;
}

.icon-small {
    height: 38px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 4px 0 3px 0;
}

.icon-small img {
    width: 38px;
    height: 38px;
    display: block;
}

.target-title {
    font-size: 1.38rem;
    font-weight: 850;
}

.syllable {
    font-size: 1.12rem;
    font-weight: 850;
    color: #40536f;
    margin-top: 2px;
}

.guide {
    background: #f5f6f8;
    border-radius: 13px;
    padding: 10px 12px;
    font-size: 0.86rem;
    line-height: 1.5;
    margin-bottom: 8px;
}

.result-good {
    background: #eaf7ef;
    border: 1px solid #b8e1c6;
    border-radius: 15px;
    padding: 16px;
    margin: 8px 0 12px;
}

.result-mid {
    background: #fff8e8;
    border: 1px solid #ead7a7;
    border-radius: 15px;
    padding: 16px;
    margin: 8px 0 12px;
}

.result-bad {
    background: #fff0f0;
    border: 1px solid #efc0c0;
    border-radius: 15px;
    padding: 16px;
    margin: 8px 0 12px;
}

.result-big {
    font-size: 1.62rem;
    font-weight: 900;
    margin: 4px 0;
}

.small {
    color: #6d7278;
    font-size: 0.82rem;
    line-height: 1.45;
}

.score-box {
    text-align: center;
    border: 1px solid #e1e4e8;
    border-radius: 14px;
    padding: 12px 8px;
    background: #fff;
    margin-bottom: 9px;
}

.score-label {
    color: #777;
    font-size: 0.77rem;
    font-weight: 700;
}

.score-value {
    font-size: 2rem;
    font-weight: 900;
    line-height: 1.1;
    margin-top: 3px;
}

.score-delta-up {
    color: #237a43;
    font-weight: 750;
}

.score-delta-down {
    color: #b34040;
    font-weight: 750;
}

.score-delta-flat {
    color: #69717a;
    font-weight: 750;
}

.bar-row {
    margin: 8px 0 11px;
}

.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.87rem;
    font-weight: 700;
    margin-bottom: 3px;
}

.bar-bg {
    height: 10px;
    background: #eceff2;
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: #657693;
    border-radius: 999px;
}

.tip {
    background: #f7f8fa;
    border-radius: 12px;
    padding: 11px 12px;
    font-size: 0.87rem;
    line-height: 1.48;
    margin-top: 8px;
}

.diary-card {
    border: 1px solid #e3e6ea;
    border-radius: 13px;
    padding: 11px 12px;
    margin: 7px 0;
    background: #fbfcfd;
}

.diary-title {
    font-weight: 850;
    font-size: 0.94rem;
}

.diary-meta {
    color: #6f747b;
    font-size: 0.78rem;
    margin-top: 3px;
}

.trend-box {
    background: #f3f6fb;
    border-radius: 13px;
    padding: 11px 12px;
    margin: 8px 0 10px;
    font-size: 0.86rem;
    line-height: 1.48;
}

@media(max-width:600px) {
    .block-container {
        padding-top: 1.15rem;
        padding-left: 0.78rem;
        padding-right: 0.78rem;
        padding-bottom: 1rem;
    }

    .main-title {
        font-size: 1.45rem;
    }

    .sub-title {
        font-size: 0.82rem;
    }

    .card {
        padding: 10px 11px;
    }

    .guide {
        padding: 9px 10px;
        font-size: 0.81rem;
    }

    .result-good,
    .result-mid,
    .result-bad {
        padding: 13px;
    }

    button {
        min-height: 42px !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# CONFIG
# =========================================================

RESONANCES = [
    "가슴",
    "입천장",
    "이빨",
    "비강",
]


INFO = {
    "가슴": {
        "syllable": "하—",
        "guide": "가슴 쪽 울림을 분명하게 느끼며 ‘하—’를 길게 발성하세요.",
        "tip": "목에 힘을 주기보다 가슴 앞쪽의 울림을 느끼면서 ‘하—’를 다시 발성해보세요.",
    },

    "입천장": {
        "syllable": "허—",
        "guide": "입천장과 구강 위쪽의 울림을 느끼며 ‘허—’를 길게 발성하세요.",
        "tip": "입천장 위쪽 공간을 느끼며 ‘허—’를 다시 발성해보세요.",
    },

    "이빨": {
        "syllable": "히—",
        "guide": "윗니와 입 앞쪽에 소리가 모이는 감각을 느끼며 ‘히—’를 길게 발성하세요.",
        "tip": "소리가 뒤로 빠지지 않도록 윗니와 입 앞쪽에 초점을 두고 ‘히—’를 다시 발성해보세요.",
    },

    "비강": {
        "syllable": "미—",
        "guide": "코 주변과 얼굴 중앙의 울림을 느끼며 ‘미—’를 길게 발성하세요.",
        "tip": "코 주변과 얼굴 중앙의 진동을 느끼며 ‘미—’를 다시 발성해보세요.",
    },
}


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
# MODEL
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

    "이빨": {
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
# FIRST-SCREEN ICONS + SELECTOR
# =========================================================

# 아이콘을 app.py 내부에 직접 포함해 경로/배포 문제 없이 표시합니다.
ICON_BASE64 = {
    '가슴': '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',
    '입천장': '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',
    '이빨': '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',
    '비강': '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',
}


SELECTOR_HTML = """
<div id="resonanceGrid" class="resonance-grid"></div>
"""


SELECTOR_CSS = """
:host {
    display: block;
    width: 100%;
    height: 100%;
}

.resonance-grid {
    width: 100%;
    height: 100%;
    box-sizing: border-box;

    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.resonance-card-button {
    width: 100%;
    min-height: 78px;

    border: 1px solid #dfe3e8;
    border-radius: 13px;
    background: #ffffff;

    display: flex;
    align-items: center;

    padding: 10px 14px;

    cursor: pointer;
    text-align: left;

    font-family: var(--st-font);
    color: var(--st-text-color);

    transition:
        border-color .15s ease,
        box-shadow .15s ease,
        transform .08s ease;
}

.resonance-card-button:hover {
    border-color: #bcc7d6;
    box-shadow: 0 2px 8px rgba(29, 45, 68, .06);
}

.resonance-card-button:active {
    transform: scale(.99);
}

.resonance-icon-wrap {
    width: 54px;
    height: 54px;
    flex: 0 0 54px;

    display: flex;
    align-items: center;
    justify-content: center;
}

.resonance-icon-wrap img {
    display: block;
    width: 54px;
    height: 54px;
    object-fit: contain;
}

.resonance-divider {
    width: 1px;
    height: 42px;
    background: #e3e6ea;
    margin: 0 14px;
    flex: 0 0 1px;
}

.resonance-label {
    font-size: 15px;
    line-height: 1.25;
    font-weight: 780;
    letter-spacing: -0.02em;
    white-space: nowrap;
}

@media (max-width: 600px) {

    .resonance-grid {
        gap: 8px;
    }

    .resonance-card-button {
        min-height: 68px;
        padding: 8px 9px;
        border-radius: 12px;
    }

    .resonance-icon-wrap {
        width: 44px;
        height: 44px;
        flex-basis: 44px;
    }

    .resonance-icon-wrap img {
        width: 44px;
        height: 44px;
    }

    .resonance-divider {
        height: 34px;
        margin: 0 9px;
    }

    .resonance-label {
        font-size: 13px;
    }
}
"""


SELECTOR_JS = r"""
export default function(component) {

    const {
        data,
        parentElement,
        setTriggerValue
    } = component;

    const grid =
        parentElement.querySelector("#resonanceGrid");

    const items = [
        ["가슴", "가슴 공명"],
        ["입천장", "입천장 공명"],
        ["이빨", "이빨 공명"],
        ["비강", "비강 공명"],
    ];

    grid.innerHTML = "";

    for (const [key, label] of items) {

        const button =
            document.createElement("button");

        button.type =
            "button";

        button.className =
            "resonance-card-button";

        const icon =
            data?.icons?.[key] ?? "";

        button.innerHTML = `
            <span class="resonance-icon-wrap">
                ${
                    icon
                    ?
                    `<img src="data:image/png;base64,${icon}" alt="">`
                    :
                    ""
                }
            </span>

            <span class="resonance-divider"></span>

            <span class="resonance-label">
                ${label}
            </span>
        `;

        button.onclick =
            () => {

                setTriggerValue(
                    "selected",
                    key
                );
            };

        grid.appendChild(
            button
        );
    }
}
"""


selector_component = st.components.v2.component(
    "resonance_selector_cards_v1",
    html=SELECTOR_HTML,
    css=SELECTOR_CSS,
    js=SELECTOR_JS,
)


def resonance_selector():

    result = selector_component(
        data={
            "icons": ICON_BASE64
        },
        on_selected_change=lambda: None,
        key="resonance_selector",
        width="stretch",
        height=170,
    )

    return getattr(
        result,
        "selected",
        None
    )


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


if st.session_state.target not in RESONANCES:

    st.session_state.target = None

    if st.session_state.page != "select":
        st.session_state.page = "select"


if st.session_state.page not in {
    "select",
    "record",
    "review",
    "result",
}:

    st.session_state.page = "select"



# =========================================================
# DEVICE-LOCAL PRACTICE HISTORY
# =========================================================

STORAGE_JS = r"""
export default function(component) {

    const {
        data,
        setStateValue
    } = component;

    const STORAGE_KEY =
        "resonance_practice_history_v1";

    if (data?.mode === "load") {

        const stored =
            window.localStorage.getItem(
                STORAGE_KEY
            );

        setStateValue(
            "stored_json",
            stored ?? "[]"
        );

        return;
    }

    if (data?.mode === "save") {

        const value =
            data?.history_json ?? "[]";

        if (
            window.localStorage.getItem(
                STORAGE_KEY
            )
            !==
            value
        ) {

            window.localStorage.setItem(
                STORAGE_KEY,
                value
            );
        }
    }
}
"""


storage_component = st.components.v2.component(
    "resonance_local_history_v1",
    js=STORAGE_JS,
)


if "storage_loaded" not in st.session_state:

    st.session_state.storage_loaded = False


storage_result = storage_component(
    data={
        "mode":
            (
                "save"
                if st.session_state.storage_loaded
                else
                "load"
            ),

        "history_json":
            (
                json.dumps(
                    st.session_state.history,
                    ensure_ascii=False
                )
                if st.session_state.storage_loaded
                else
                "[]"
            ),
    },
    default={
        "stored_json": None
    },
    on_stored_json_change=lambda: None,
    key="practice_history_storage",
    width=1,
    height=1,
)


if not st.session_state.storage_loaded:

    stored_json = getattr(
        storage_result,
        "stored_json",
        None
    )

    if stored_json is not None:

        try:

            loaded_history = json.loads(
                stored_json
            )

            if isinstance(
                loaded_history,
                list
            ):

                st.session_state.history = (
                    loaded_history
                )

        except Exception:

            st.session_state.history = []

        st.session_state.storage_loaded = True

        st.rerun()


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

.record-button.recording {
    background:#e53935;
}

.mic-icon {
    font-size:34px;
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
    color:#269451;
}

.status {
    margin-top:8px;
    font-size:14px;
    font-weight:750;
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
}

.help {
    margin-top:6px;
    font-size:10.5px;
    color:#7a7f87;
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


    function writeString(view, offset, text) {

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


    function encodeWav(samples, sr) {

        const buffer =
            new ArrayBuffer(
                44 + samples.length * 2
            );

        const view =
            new DataView(buffer);

        writeString(view, 0, "RIFF");

        view.setUint32(
            4,
            36 + samples.length * 2,
            true
        );

        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");

        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);

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

        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);

        writeString(view, 36, "data");

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
            "00:"
            +
            (
                capped / 1000
            )
            .toFixed(1)
            .padStart(4, "0");

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
    }


    async function stopRecording() {

        if (!recording) {
            return;
        }

        recording = false;

        clearTimeout(stopTimer);

        cancelAnimationFrame(
            timerFrame
        );

        timerText.textContent =
            "00:05.0";

        progressBar.style.width =
            "100%";

        statusText.textContent =
            "저장 중...";

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
                        echoCancellation:false,
                        noiseSuppression:false,
                        autoGainControl:false,
                        channelCount:1
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

            source.connect(processor);

            processor.connect(
                silentGain
            );

            silentGain.connect(
                audioContext.destination
            );

            recording = true;

            button.classList.add(
                "recording"
            );

            statusText.classList.add(
                "recording"
            );

            statusText.textContent =
                "● 녹음 중";

            helpText.textContent =
                "5초 후 자동 종료됩니다.";

            startTime =
                performance.now();

            updateTimer();

            stopTimer =
                setTimeout(
                    stopRecording,
                    RECORD_MS
                );

        } catch (error) {

            statusText.textContent =
                "마이크 사용 불가";

            helpText.textContent =
                "마이크 권한을 허용해주세요.";
        }
    }


    button.onclick =
        () => {

            if (!recording) {
                startRecording();
            }
        };


    return () => {

        clearTimeout(stopTimer);

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
    };
}
"""


recorder_component = st.components.v2.component(
    "student_resonance_recorder_v4",
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

    return base64.b64decode(
        audio_b64
    )


# =========================================================
# AUDIO ANALYSIS
# =========================================================

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
            power[total_mask]
        )
        +
        1e-12
    )

    return float(
        np.sum(
            power[band_mask]
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

    freq = frequencies[mask]
    pwr = power[mask]

    valid = (
        (freq > 0)
        &
        (pwr > 0)
    )

    freq = freq[valid]
    pwr = pwr[valid]

    if len(freq) < 10:
        return np.nan

    return float(
        np.polyfit(
            np.log10(freq),
            10
            *
            np.log10(
                pwr + 1e-12
            ),
            1,
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

            if np.isfinite(value):
                values.append(value)

        except Exception:
            pass

    if not values:
        return np.nan

    return float(
        np.median(values)
    )


def analyze_audio(
    audio_bytes
):

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=True
    ) as tmp:

        tmp.write(audio_bytes)

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
                "발성 시간이 너무 짧습니다. 다시 녹음해주세요."
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

        power = spectrum ** 2

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


def classify(features):

    distances = {}

    for resonance, centroid in CENTROIDS.items():

        z_values = []

        for key in FEATURES:

            value = features[key]

            if not np.isfinite(value):

                raise ValueError(
                    "음향 분석이 불안정합니다. 다시 녹음해주세요."
                )

            sd = max(
                POOLED_STD[key],
                1e-12
            )

            z_values.append(
                (
                    value
                    -
                    centroid[key]
                )
                /
                sd
            )

        distances[resonance] = float(
            np.sqrt(
                np.mean(
                    np.array(z_values) ** 2
                )
            )
        )

    ordered = sorted(
        distances.items(),
        key=lambda x: x[1]
    )

    raw = {
        name: np.exp(-distance)
        for name, distance
        in ordered
    }

    total = (
        sum(raw.values())
        +
        1e-12
    )

    scores = {
        name:
            raw[name]
            /
            total
            *
            100
        for name in raw
    }

    return ordered, scores


def previous_target_score(target):

    rows = [
        row
        for row
        in st.session_state.history
        if row["target"] == target
    ]

    if not rows:
        return None

    return float(
        rows[-1]["target_score"]
    )


def build_result(
    target,
    ordered,
    scores
):

    prediction = ordered[0][0]

    target_score = float(
        scores[target]
    )

    top_score = float(
        scores[prediction]
    )

    second_name = ordered[1][0]

    second_score = float(
        scores[second_name]
    )

    previous = previous_target_score(
        target
    )

    delta = (
        None
        if previous is None
        else
        target_score
        -
        previous
    )

    if prediction == target:

        if target_score >= 50:

            status = (
                "잘 되고 있습니다"
            )

            css = (
                "result-good"
            )

            message = (
                f"현재 발성은 목표인 {target} 공명과 가장 잘 일치합니다."
            )

        else:

            status = (
                "방향은 맞습니다"
            )

            css = (
                "result-mid"
            )

            message = (
                f"{target} 공명이 가장 강하지만 다른 공명도 함께 나타납니다."
            )

    else:

        status = (
            f"{prediction} 공명이 더 강합니다"
        )

        css = (
            "result-bad"
        )

        message = (
            f"목표는 {target} 공명이지만 현재는 "
            f"{prediction} 공명의 특징이 더 강합니다."
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

        "target_score":
            target_score,

        "top_score":
            top_score,

        "second_name":
            second_name,

        "second_score":
            second_score,

        "scores":
            scores,

        "delta":
            delta,
    }


def reset_measurement(
    keep_target=True
):

    st.session_state.audio_bytes = None

    st.session_state.result = None

    st.session_state.recorder_id += 1

    if keep_target:

        st.session_state.page = (
            "record"
        )

    else:

        st.session_state.target = None

        st.session_state.page = (
            "select"
        )



def practice_trend_message(target):

    rows = [
        row
        for row
        in st.session_state.history
        if row.get("target") == target
    ]

    if not rows:

        return (
            "아직 연습 기록이 없습니다."
        )

    scores = [
        float(
            row.get(
                "target_score",
                0
            )
        )
        for row
        in rows
    ]

    if len(scores) == 1:

        return (
            "첫 기록입니다. 같은 공명을 반복하면 "
            "변화 추세를 확인할 수 있습니다."
        )

    recent = scores[-3:]

    if len(recent) >= 3:

        change = (
            recent[-1]
            -
            recent[0]
        )

        if change >= 8:

            return (
                f"최근 {len(recent)}회에서 "
                f"{target} 공명 일치도가 "
                "점차 좋아지고 있습니다."
            )

        if change <= -8:

            return (
                f"최근 {len(recent)}회에서 "
                f"{target} 공명 일치도가 낮아졌습니다. "
                "공명 위치를 다시 점검해보세요."
            )

    delta = (
        scores[-1]
        -
        scores[-2]
    )

    if delta >= 5:

        return (
            f"직전 {target} 연습보다 "
            "일치도가 좋아졌습니다."
        )

    if delta <= -5:

        return (
            f"직전 {target} 연습보다 "
            "일치도가 낮아졌습니다."
        )

    return (
        f"직전 {target} 연습과 "
        "비슷한 수준을 유지하고 있습니다."
    )


def build_diary_summary():

    summary_rows = []

    for resonance in RESONANCES:

        rows = [
            row
            for row
            in st.session_state.history
            if row.get("target") == resonance
        ]

        if not rows:
            continue

        scores = [
            float(
                row.get(
                    "target_score",
                    0
                )
            )
            for row
            in rows
        ]

        if len(scores) == 1:

            trend = "첫 기록"

        else:

            recent = scores[-3:]

            change = (
                recent[-1]
                -
                recent[0]
            )

            if change >= 8:
                trend = "좋아지는 중"

            elif change <= -8:
                trend = "점검 필요"

            else:
                trend = "비슷하게 유지"

        summary_rows.append(
            {
                "공명":
                    resonance,

                "연습횟수":
                    len(rows),

                "최근 일치도":
                    round(
                        scores[-1]
                    ),

                "최고 일치도":
                    round(
                        max(scores)
                    ),

                "최근 추세":
                    trend,
            }
        )

    return summary_rows


# =========================================================
# DIARY
# =========================================================

def render_diary():

    if not st.session_state.history:
        return

    with st.expander(
        "📘 연습일지"
    ):

        summary_rows = (
            build_diary_summary()
        )

        if summary_rows:

            st.markdown(
                "#### 연습 요약"
            )

            st.dataframe(
                pd.DataFrame(
                    summary_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            "#### 회차별 기록"
        )

        for row in reversed(
            st.session_state.history
        ):

            delta_text = ""

            if row.get(
                "delta"
            ) is not None:

                sign = (
                    "+"
                    if row["delta"] >= 0
                    else
                    ""
                )

                delta_text = (
                    f" · 이전 대비 "
                    f"{sign}{row['delta']:.0f}"
                )

            st.markdown(
                f"""
<div class="diary-card">

<div class="diary-title">
{row.get('session_no', '-')}회차 · {row.get('target', '-')}
</div>

<div class="diary-meta">
{row.get('status', '-')}
</div>

<div style="margin-top:5px;font-size:.87rem;">

공명 일치도
<b>{float(row.get('target_score', 0)):.0f}</b>
{delta_text}

<br>

가장 강한 공명:
<b>{row.get('prediction', '-')} {float(row.get('top_score', 0)):.0f}</b>

· 2순위:
{row.get('second_name', '-')} {float(row.get('second_score', 0)):.0f}

</div>

</div>
""",
                unsafe_allow_html=True,
            )

        detail_df = pd.DataFrame(
            st.session_state.history
        )

        with st.expander(
            "연구자용 상세 데이터"
        ):

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True,
            )

        csv_bytes = (
            detail_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        )

        st.download_button(
            "연구자용 상세 데이터 CSV 다운로드",
            data=csv_bytes,
            file_name="resonance_research_data.csv",
            mime="text/csv",
            use_container_width=True,
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
연습할 공명을 선택하고 안내된 소리를 5초간 발성하세요.
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
<b>연습할 공명을 선택하세요.</b>
</div>
""",
        unsafe_allow_html=True,
    )

    selected = resonance_selector()

    if selected in RESONANCES:

        st.session_state.target = (
            selected
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

    info = INFO[target]

    st.markdown(
        f"""
<div class="card">

<div class="small">
현재 연습
</div>

<div class="target-title">
{target} 공명
</div>

<div class="syllable">
{info['syllable']}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
<div class="guide">

<b>
{info['syllable']}를 약 5초간 유지하세요.
</b>

<br>

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

<b>{target} 공명</b><br>

<span class="small">
{INFO[target]['syllable']} 녹음 확인
</span>

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

                    ordered, scores = classify(
                        features
                    )

                    result = build_result(
                        target,
                        ordered,
                        scores
                    )


                    history_row = {

                        "session_no":
                            len(
                                st.session_state.history
                            )
                            +
                            1,

                        "timestamp":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "target":
                            target,

                        "syllable":
                            INFO[target][
                                "syllable"
                            ],

                        "prediction":
                            result[
                                "prediction"
                            ],

                        "status":
                            result[
                                "status"
                            ],

                        "target_score":
                            result[
                                "target_score"
                            ],

                        "top_score":
                            result[
                                "top_score"
                            ],

                        "second_name":
                            result[
                                "second_name"
                            ],

                        "second_score":
                            result[
                                "second_score"
                            ],

                        "delta":
                            result[
                                "delta"
                            ],

                        **features,
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
                    str(error)
                )


# =========================================================
# RESULT
# =========================================================

elif st.session_state.page == "result":

    result = (
        st.session_state.result
    )

    target = (
        result["target"]
    )


    st.markdown(
        f"""
<div class="{result['css']}">

<div class="small">
분석 결과
</div>

<div class="result-big">
{result['status']}
</div>

<div>
{result['message']}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    delta = (
        result[
            "delta"
        ]
    )


    if delta is None:

        delta_text = (
            "첫 기록"
        )

    elif delta >= 3:

        delta_text = (
            f"▲ {delta:.0f}"
        )

    elif delta <= -3:

        delta_text = (
            f"▼ {abs(delta):.0f}"
        )

    else:

        delta_text = (
            "비슷함"
        )


    st.markdown(
        f"""
<div class="score-box">

<div class="score-label">
{target} 공명 일치도
</div>

<div class="score-value">
{result['target_score']:.0f}
</div>

<div class="small">
이전 같은 공명 연습 대비 {delta_text}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        "#### 현재 공명 분석"
    )


    ordered_scores = sorted(
        result[
            "scores"
        ].items(),
        key=lambda x: x[1],
        reverse=True
    )


    for name, value in ordered_scores:

        st.markdown(
            f"""
<div class="bar-row">

<div class="bar-label">

<span>{name}</span>
<span>{value:.0f}</span>

</div>

<div class="bar-bg">

<div class="bar-fill"
style="width:{value:.1f}%">
</div>

</div>

</div>
""",
            unsafe_allow_html=True,
        )



    st.markdown(
        f"""
<div class="trend-box">

<b>연습 변화</b><br>

{practice_trend_message(target)}

</div>
""",
        unsafe_allow_html=True,
    )


    if (
        result["prediction"]
        !=
        target
    ):

        st.markdown(
            f"""
<div class="tip">

<b>다음 발성</b><br>

{INFO[target]['tip']}

</div>
""",
            unsafe_allow_html=True,
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
# DIARY
# =========================================================

render_diary()
