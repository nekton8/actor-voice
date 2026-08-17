import base64
import json
import tempfile
from datetime import datetime
from pathlib import Path

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


st.markdown(
    """
<style>

.block-container {
    max-width: 720px;
    padding-top: 0.55rem;
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

.result-good,
.result-mid,
.result-bad {
    border-radius: 15px;
    padding: 16px;
    margin: 8px 0 12px;
}

.result-good {
    background: #eaf7ef;
    border: 1px solid #b8e1c6;
}

.result-mid {
    background: #fff8e8;
    border: 1px solid #ead7a7;
}

.result-bad {
    background: #fff0f0;
    border: 1px solid #efc0c0;
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

.trend-box {
    background: #f3f6fb;
    border-radius: 13px;
    padding: 11px 12px;
    margin: 8px 0 10px;
    font-size: 0.86rem;
    line-height: 1.48;
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

@media(max-width:600px) {

    .block-container {
        padding-top: 0.4rem;
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


RESONANCES = [
    "가슴",
    "입천장",
    "이빨·전방",
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

    "이빨·전방": {
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
# FIRST SCREEN ICONS
# =========================================================

ASSET_DIR = (
    Path(__file__).resolve().parent
    /
    "assets"
)


def load_png_base64(filename):

    path = (
        ASSET_DIR
        /
        filename
    )

    if not path.exists():
        return ""

    return base64.b64encode(
        path.read_bytes()
    ).decode(
        "ascii"
    )


ICON_BASE64 = {

    "가슴":
        load_png_base64(
            "chest_icon.png"
        ),

    "입천장":
        load_png_base64(
            "palate_icon.png"
        ),

    "이빨·전방":
        load_png_base64(
            "front_icon.png"
        ),

    "비강":
        load_png_base64(
            "nasal_icon.png"
        ),
}


SELECTOR_HTML = """
<div
id="resonanceGrid"
class="resonance-grid">
</div>
"""


SELECTOR_CSS = """
:host {
    display:block;
    width:100%;
    height:100%;
}

.resonance-grid {
    width:100%;
    height:100%;

    box-sizing:border-box;

    display:grid;

    grid-template-columns:
        repeat(
            2,
            minmax(
                0,
                1fr
            )
        );

    gap:10px;
}

.resonance-card-button {

    width:100%;

    min-height:78px;

    border:
        1px
        solid
        #dfe3e8;

    border-radius:13px;

    background:#fff;

    display:flex;

    align-items:center;

    padding:
        10px
        14px;

    cursor:pointer;

    text-align:left;

    font-family:
        var(--st-font);

    color:
        var(--st-text-color);

    transition:
        border-color .15s ease,
        box-shadow .15s ease,
        transform .08s ease;
}

.resonance-card-button:hover {

    border-color:
        #bcc7d6;

    box-shadow:
        0
        2px
        8px
        rgba(
            29,
            45,
            68,
            .06
        );
}

.resonance-card-button:active {
    transform:
        scale(.99);
}

.resonance-icon-wrap {

    width:54px;

    height:54px;

    flex:
        0
        0
        54px;

    display:flex;

    align-items:center;

    justify-content:center;
}

.resonance-icon-wrap img {

    display:block;

    width:54px;

    height:54px;

    object-fit:contain;
}

.resonance-divider {

    width:1px;

    height:42px;

    background:
        #e3e6ea;

    margin:
        0
        14px;

    flex:
        0
        0
        1px;
}

.resonance-label {

    font-size:15px;

    line-height:1.25;

    font-weight:780;

    letter-spacing:
        -0.02em;

    white-space:nowrap;
}


@media(max-width:600px) {

    .resonance-grid {
        gap:8px;
    }

    .resonance-card-button {

        min-height:68px;

        padding:
            8px
            9px;

        border-radius:12px;
    }

    .resonance-icon-wrap {

        width:44px;

        height:44px;

        flex-basis:44px;
    }

    .resonance-icon-wrap img {

        width:44px;

        height:44px;
    }

    .resonance-divider {

        height:34px;

        margin:
            0
            9px;
    }

    .resonance-label {
        font-size:13px;
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
        parentElement
        .querySelector(
            "#resonanceGrid"
        );


    const items = [

        [
            "가슴",
            "가슴 공명"
        ],

        [
            "입천장",
            "입천장 공명"
        ],

        [
            "이빨·전방",
            "이빨·전방 공명"
        ],

        [
            "비강",
            "비강 공명"
        ]
    ];


    grid.innerHTML =
        "";


    for (
        const [
            key,
            label
        ]
        of
        items
    ) {

        const button =
            document
            .createElement(
                "button"
            );


        button.type =
            "button";


        button.className =
            "resonance-card-button";


        const icon =
            data?.icons?.[key]
            ??
            "";


        button.innerHTML = `

            <span
            class="resonance-icon-wrap">

                ${
                    icon
                    ?
                    `
                    <img
                    src="data:image/png;base64,${icon}"
                    alt="">
                    `
                    :
                    ""
                }

            </span>

            <span
            class="resonance-divider">
            </span>

            <span
            class="resonance-label">

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


selector_component = (
    st.components.v2.component(
        "resonance_selector_cards_v2",
        html=SELECTOR_HTML,
        css=SELECTOR_CSS,
        js=SELECTOR_JS,
    )
)


def resonance_selector():

    result = selector_component(

        data={
            "icons":
                ICON_BASE64
        },

        on_selected_change=
            lambda: None,

        key=
            "resonance_selector",

        width=
            "stretch",

        height=
            170,
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
    "page":
        "select",

    "target":
        None,

    "audio_bytes":
        None,

    "recorder_id":
        0,

    "result":
        None,

    "history":
        [],
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


if (
    st.session_state.target
    not in
    RESONANCES
):

    st.session_state.target = (
        None
    )

    if (
        st.session_state.page
        !=
        "select"
    ):

        st.session_state.page = (
            "select"
        )


if (
    st.session_state.page
    not in
    {
        "select",
        "record",
        "review",
        "result",
    }
):

    st.session_state.page = (
        "select"
    )


# =========================================================
# LOCAL STORAGE
# =========================================================

STORAGE_JS = r"""
export default function(component) {

    const {
        data,
        setStateValue
    } = component;


    const KEY =
        "resonance_practice_history_v1";


    if (
        data?.mode
        ===
        "load"
    ) {

        setStateValue(
            "stored_json",
            window
            .localStorage
            .getItem(
                KEY
            )
            ??
            "[]"
        );

        return;
    }


    if (
        data?.mode
        ===
        "save"
    ) {

        const value =
            data?.history_json
            ??
            "[]";


        if (
            window
            .localStorage
            .getItem(
                KEY
            )
            !==
            value
        ) {

            window
            .localStorage
            .setItem(
                KEY,
                value
            );
        }
    }
}
"""


storage_component = (
    st.components.v2.component(
        "resonance_local_history_v2",
        js=STORAGE_JS,
    )
)


if (
    "storage_loaded"
    not in
    st.session_state
):

    st.session_state.storage_loaded = (
        False
    )


storage_result = storage_component(

    data={

        "mode":
            (
                "save"
                if
                st.session_state.storage_loaded
                else
                "load"
            ),

        "history_json":
            (
                json.dumps(
                    st.session_state.history,
                    ensure_ascii=False
                )
                if
                st.session_state.storage_loaded
                else
                "[]"
            ),
    },

    default={
        "stored_json":
            None
    },

    on_stored_json_change=
        lambda: None,

    key=
        "practice_history_storage",

    width=
        0,

    height=
        0,
)


if not st.session_state.storage_loaded:

    stored_json = getattr(
        storage_result,
        "stored_json",
        None
    )


    if stored_json is not None:

        try:

            loaded = json.loads(
                stored_json
            )


            if isinstance(
                loaded,
                list
            ):

                st.session_state.history = (
                    loaded
                )


        except Exception:

            st.session_state.history = (
                []
            )


        st.session_state.storage_loaded = (
            True
        )


        st.rerun()


# =========================================================
# RECORDER
# =========================================================

RECORDER_HTML = """
<div class="recorder">

    <button
    id="recordButton"
    class="record-button">

        <div class="mic-icon">
            🎙
        </div>

    </button>

    <div
    id="statusText"
    class="status">
        녹음 시작
    </div>

    <div
    id="timerText"
    class="timer">
        00:00.0
    </div>

    <div class="progress-wrap">

        <div
        id="progressBar"
        class="progress">
        </div>

    </div>

    <div
    id="helpText"
    class="help">

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

    border:
        1px
        solid
        #e1e4e8;

    border-radius:17px;

    display:flex;

    flex-direction:column;

    align-items:center;

    justify-content:center;

    padding:
        12px
        10px;

    overflow:hidden;

    background:
        var(--st-background-color);

    font-family:
        var(--st-font);
}

.record-button {

    width:78px;

    height:78px;

    flex:
        0
        0
        78px;

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

    animation:
        pulse
        1.05s
        infinite;
}

.record-button.recording
.mic-icon {

    font-size:0;
}

.record-button.recording
.mic-icon::after {

    content:"■";

    font-size:25px;

    color:white;
}

.record-button.done {
    background:#e8f6ed;
}

.record-button.done
.mic-icon {

    font-size:0;
}

.record-button.done
.mic-icon::after {

    content:"✓";

    font-size:36px;

    font-weight:800;

    color:#269451;
}

.status {

    margin-top:8px;

    font-size:14px;

    font-weight:750;

    color:
        var(--st-text-color);
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

    font-variant-numeric:
        tabular-nums;

    color:
        var(--st-text-color);
}

.progress-wrap {

    width:min(
        300px,
        80%
    );

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
        box-shadow:
            0
            0
            0
            0
            rgba(
                229,
                57,
                53,
                .28
            );
    }

    70% {
        box-shadow:
            0
            0
            0
            12px
            rgba(
                229,
                57,
                53,
                0
            );
    }

    100% {
        box-shadow:
            0
            0
            0
            0
            rgba(
                229,
                57,
                53,
                0
            );
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
        parentElement
        .querySelector(
            "#recordButton"
        );


    const statusText =
        parentElement
        .querySelector(
            "#statusText"
        );


    const timerText =
        parentElement
        .querySelector(
            "#timerText"
        );


    const progressBar =
        parentElement
        .querySelector(
            "#progressBar"
        );


    const helpText =
        parentElement
        .querySelector(
            "#helpText"
        );


    const RECORD_MS =
        5000;


    let recording =
        false;


    let stream =
        null;


    let audioContext =
        null;


    let source =
        null;


    let processor =
        null;


    let silentGain =
        null;


    let buffers =
        [];


    let sampleRate =
        44100;


    let startTime =
        0;


    let timerFrame =
        null;


    let stopTimer =
        null;


    function mergeBuffers(
        parts
    ) {

        let total =
            0;


        for (
            const part
            of
            parts
        ) {

            total +=
                part.length;
        }


        const merged =
            new Float32Array(
                total
            );


        let offset =
            0;


        for (
            const part
            of
            parts
        ) {

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
            36
            +
            samples.length * 2,
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


        let offset =
            44;


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


            offset +=
                2;
        }


        return new Blob(
            [view],
            {
                type:
                    "audio/wav"
            }
        );
    }


    function blobToBase64(
        blob
    ) {

        return new Promise(
            (
                resolve,
                reject
            ) => {

                const reader =
                    new FileReader();


                reader.onloadend =
                    () => {

                        resolve(
                            reader
                            .result
                            .substring(
                                reader
                                .result
                                .indexOf(
                                    ","
                                )
                                +
                                1
                            )
                        );
                    };


                reader.onerror =
                    reject;


                reader
                .readAsDataURL(
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
            "00:"
            +
            (
                capped
                /
                1000
            )
            .toFixed(1)
            .padStart(
                4,
                "0"
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

            processor
            ?.disconnect();

        } catch {}


        try {

            source
            ?.disconnect();

        } catch {}


        try {

            silentGain
            ?.disconnect();

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

                await audioContext
                .close();

            } catch {}
        }


        processor =
            null;

        source =
            null;

        silentGain =
            null;

        stream =
            null;

        audioContext =
            null;
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
                await navigator
                .mediaDevices
                .getUserMedia(
                    {
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
                    }
                );


            audioContext =
                new (
                    window.AudioContext
                    ||
                    window.webkitAudioContext
                )();


            sampleRate =
                audioContext
                .sampleRate;


            source =
                audioContext
                .createMediaStreamSource(
                    stream
                );


            processor =
                audioContext
                .createScriptProcessor(
                    4096,
                    1,
                    1
                );


            silentGain =
                audioContext
                .createGain();


            silentGain.gain.value =
                0;


            buffers =
                [];


            processor
            .onaudioprocess =
                event => {

                    if (!recording) {
                        return;
                    }


                    buffers.push(
                        new Float32Array(
                            event
                            .inputBuffer
                            .getChannelData(
                                0
                            )
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

                audioContext
                .close();

            } catch {}
        }
    };
}
"""


recorder_component = (
    st.components.v2.component(
        "student_resonance_recorder_v5",
        html=RECORDER_HTML,
        css=RECORDER_CSS,
        js=RECORDER_JS,
    )
)


def five_second_recorder(key):

    result = recorder_component(

        default={
            "audio_b64":
                None
        },

        on_audio_b64_change=
            lambda: None,

        key=
            key,

        width=
            "stretch",

        height=
            205,
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


    freq = (
        frequencies[
            mask
        ]
    )


    pwr = (
        power[
            mask
        ]
    )


    valid = (
        (freq > 0)
        &
        (pwr > 0)
    )


    freq = (
        freq[
            valid
        ]
    )


    pwr = (
        pwr[
            valid
        ]
    )


    if len(freq) < 10:

        return np.nan


    return float(
        np.polyfit(
            np.log10(
                freq
            ),
            10
            *
            np.log10(
                pwr
                +
                1e-12
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


        y_trim, _ = (
            librosa.effects.trim(
                y,
                top_db=35
            )
        )


        voice_duration = (
            len(
                y_trim
            )
            /
            sr
        )


        if voice_duration < 2:

            raise ValueError(
                "발성 시간이 너무 짧습니다. "
                "소리를 조금 더 길게 유지해서 다시 녹음해주세요."
            )


        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        y_trim
                    )
                )
                +
                1e-12
            )
        )


        rms_db = (
            20
            *
            np.log10(
                rms
            )
        )


        peak = float(
            np.max(
                np.abs(
                    y_trim
                )
            )
        )


        if rms_db < -48:

            raise ValueError(
                "녹음된 소리가 너무 작습니다. "
                "휴대폰을 조금 가까이 두고 다시 녹음해주세요."
            )


        if peak >= 0.999:

            raise ValueError(
                "소리가 너무 크게 녹음되었습니다. "
                "조금 작게 발성해서 다시 녹음해주세요."
            )


        y_norm = (
            librosa.util.normalize(
                y_trim
            )
        )


        sound = (
            parselmouth.Sound(
                tmp.name
            )
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


        mean_power = (
            np.mean(
                power,
                axis=1
            )
        )


        frequencies = (
            librosa
            .fft_frequencies(
                sr=sr,
                n_fft=2048
            )
        )


        centroid = float(
            np.mean(
                librosa
                .feature
                .spectral_centroid(
                    y=y_norm,
                    sr=sr
                )
            )
        )


        return {

            "voice_duration_sec":
                voice_duration,

            "rms_db":
                rms_db,

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


    for (
        resonance,
        centroid
    ) in CENTROIDS.items():


        z_values = []


        for key in FEATURES:

            value = (
                features[
                    key
                ]
            )


            if not np.isfinite(
                value
            ):

                raise ValueError(
                    "음향 특징을 안정적으로 분석하지 못했습니다. "
                    "다시 녹음해주세요."
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
                max(
                    POOLED_STD[
                        key
                    ],
                    1e-12
                )
            )


        distances[
            resonance
        ] = float(
            np.sqrt(
                np.mean(
                    np.array(
                        z_values,
                        dtype=float
                    )
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
        for (
            name,
            distance
        )
        in
        ordered
    }


    total = (
        sum(
            raw.values()
        )
        +
        1e-12
    )


    scores = {
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
        scores
    )


# =========================================================
# RESULT + TREND
# =========================================================

def previous_target_score(
    target
):

    rows = [
        row
        for row
        in st.session_state.history
        if row.get(
            "target"
        )
        ==
        target
    ]


    if not rows:

        return None


    return float(
        rows[
            -1
        ][
            "target_score"
        ]
    )


def practice_trend_message(
    target,
    current_score=None
):

    scores = [
        float(
            row.get(
                "target_score",
                0
            )
        )
        for row
        in st.session_state.history
        if row.get(
            "target"
        )
        ==
        target
    ]


    if (
        current_score
        is not None
        and
        (
            not scores
            or
            abs(
                scores[-1]
                -
                current_score
            )
            >
            1e-9
        )
    ):

        scores.append(
            float(
                current_score
            )
        )


    if not scores:

        return (
            "아직 연습 기록이 없습니다."
        )


    if len(
        scores
    ) == 1:

        return (
            "첫 기록입니다. "
            "같은 공명을 반복하면 변화 추세를 확인할 수 있습니다."
        )


    recent = (
        scores[
            -3:
        ]
    )


    if len(
        recent
    ) >= 3:

        change = (
            recent[
                -1
            ]
            -
            recent[
                0
            ]
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
        scores[
            -1
        ]
        -
        scores[
            -2
        ]
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


def build_result(
    target,
    ordered,
    scores
):

    prediction = (
        ordered[
            0
        ][
            0
        ]
    )


    top_score = float(
        scores[
            prediction
        ]
    )


    target_score = float(
        scores[
            target
        ]
    )


    second_name = (
        ordered[
            1
        ][
            0
        ]
    )


    second_score = float(
        scores[
            second_name
        ]
    )


    previous = (
        previous_target_score(
            target
        )
    )


    delta = (
        None
        if
        previous is None
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
                f"현재 발성은 목표인 {target} 공명과 "
                "가장 잘 일치합니다."
            )


        else:

            status = (
                "방향은 맞습니다"
            )


            css = (
                "result-mid"
            )


            message = (
                f"{target} 공명이 가장 강하지만 "
                "다른 공명도 함께 나타나고 있습니다."
            )


    else:

        target_rank = (
            [
                name
                for (
                    name,
                    _
                )
                in
                ordered
            ]
            .index(
                target
            )
            +
            1
        )


        if (
            target_rank == 2
            and
            target_score
            >=
            top_score - 10
        ):

            status = (
                "목표 공명이 충분히 분명하지 않습니다"
            )


            css = (
                "result-mid"
            )


            message = (
                f"{target} 공명도 나타나지만 "
                f"현재는 {prediction} 공명이 "
                "조금 더 강합니다."
            )


        else:

            status = (
                f"{prediction} 공명이 더 강합니다"
            )


            css = (
                "result-bad"
            )


            message = (
                f"목표는 {target} 공명이지만 "
                f"현재 발성에서는 {prediction} 공명의 특징이 "
                "더 강하게 나타납니다."
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
        in
        RESONANCES
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
# DIARY
# =========================================================

def build_diary_summary():

    output = []


    for resonance in RESONANCES:

        rows = [
            row
            for row
            in st.session_state.history
            if row.get(
                "target"
            )
            ==
            resonance
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


        recent = (
            scores[
                -3:
            ]
        )


        if len(
            scores
        ) == 1:

            trend = (
                "첫 기록"
            )


        else:

            change = (
                recent[
                    -1
                ]
                -
                recent[
                    0
                ]
            )


            if change >= 8:

                trend = (
                    "좋아지는 중"
                )


            elif change <= -8:

                trend = (
                    "점검 필요"
                )


            else:

                trend = (
                    "비슷하게 유지"
                )


        output.append(
            {
                "공명":
                    resonance,

                "연습횟수":
                    len(
                        rows
                    ),

                "최근 일치도":
                    round(
                        scores[
                            -1
                        ]
                    ),

                "최고 일치도":
                    round(
                        max(
                            scores
                        )
                    ),

                "최근 추세":
                    trend,
            }
        )


    return output


def render_diary():

    if not st.session_state.history:

        return


    with st.expander(
        "📘 연습일지"
    ):

        summary = (
            build_diary_summary()
        )


        if summary:

            st.markdown(
                "#### 연습 요약"
            )


            st.dataframe(
                pd.DataFrame(
                    summary
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

            delta = row.get(
                "delta"
            )


            delta_text = (
                ""
            )


            if delta is not None:

                sign = (
                    "+"
                    if
                    delta >= 0
                    else
                    ""
                )


                delta_text = (
                    f" · 이전 대비 "
                    f"{sign}{delta:.0f}"
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
<b>
{float(row.get('target_score', 0)):.0f}
</b>
{delta_text}

<br>

가장 강한 공명:
<b>
{row.get('prediction', '-')}
{float(row.get('top_score', 0)):.0f}
</b>

· 2순위:
{row.get('second_name', '-')}
{float(row.get('second_score', 0)):.0f}

</div>

</div>
""",
                unsafe_allow_html=True,
            )


        detail_df = (
            pd.DataFrame(
                st.session_state.history
            )
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
<b>
연습할 공명을 선택하세요.
</b>
</div>
""",
        unsafe_allow_html=True,
    )


    selected = (
        resonance_selector()
    )


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


    info = (
        INFO[
            target
        ]
    )


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
        key=
            "student_"
            +
            str(
                st.session_state.recorder_id
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
            False
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
{target} 공명
</b>

<br>

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


    col1, col2 = (
        st.columns(
            2
        )
    )


    with col1:

        if st.button(
            "🔄 다시 녹음",
            use_container_width=True
        ):

            reset_measurement(
                True
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


                    row = {

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
                            INFO[
                                target
                            ][
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
                        row
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
                    str(
                        error
                    )
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
이전 같은 공명 연습 대비
{delta_text}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        "#### 현재 공명 분석"
    )


    for (
        name,
        value
    ) in sorted(
        result[
            "scores"
        ].items(),
        key=lambda x: x[1],
        reverse=True
    ):

        st.markdown(
            f"""
<div class="bar-row">

<div class="bar-label">

<span>
{name}
</span>

<span>
{value:.0f}
</span>

</div>

<div class="bar-bg">

<div
class="bar-fill"
style="
width:
{min(max(value, 0), 100):.1f}%;
">
</div>

</div>

</div>
""",
            unsafe_allow_html=True,
        )


    st.markdown(
        f"""
<div class="trend-box">

<b>
연습 변화
</b>

<br>

{practice_trend_message(target)}

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
        "공명 일치도는 실제 공명의 양이나 확률이 아니라, "
        "현재 음성이 4개 훈련 기준 패턴 중 어디에 가까운지를 "
        "비교한 연습용 점수입니다."
    )


    col1, col2 = (
        st.columns(
            2
        )
    )


    with col1:

        if st.button(
            "🔄 같은 공명 다시",
            use_container_width=True
        ):

            reset_measurement(
                True
            )


            st.rerun()


    with col2:

        if st.button(
            "다른 공명 선택",
            type="primary",
            use_container_width=True
        ):

            reset_measurement(
                False
            )


            st.rerun()


render_diary()
