import base64
import random
import tempfile

import librosa
import numpy as np
import pandas as pd
import parselmouth
import streamlit as st
from parselmouth.praat import call


st.set_page_config(
    page_title="남성 공명 블라인드 테스트 2",
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
    font-size: 1.7rem;
    font-weight: 850;
    letter-spacing: -0.04em;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #666;
    font-size: 0.88rem;
    line-height: 1.45;
    margin-bottom: 0.7rem;
}

.card {
    border: 1px solid #e1e5e9;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 8px;
    background: #fafbfc;
}

.target {
    font-size: 1.45rem;
    font-weight: 850;
    margin-top: 3px;
}

.guide {
    background: #f5f6f8;
    border-radius: 13px;
    padding: 10px 12px;
    font-size: 0.86rem;
    line-height: 1.45;
    margin-bottom: 8px;
}

.result-ok {
    background: #eaf7ef;
    border: 1px solid #b8e1c6;
    border-radius: 15px;
    padding: 16px;
    margin: 8px 0 12px;
}

.result-no {
    background: #fff0f0;
    border: 1px solid #efc0c0;
    border-radius: 15px;
    padding: 16px;
    margin: 8px 0 12px;
}

.result-title {
    font-size: 1.2rem;
    font-weight: 850;
}

.result-big {
    font-size: 1.7rem;
    font-weight: 900;
    margin: 5px 0;
}

.mini {
    border: 1px solid #e2e5e8;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 7px;
}

.done {
    background: #eef4ff;
    border: 1px solid #cad8ef;
    border-radius: 15px;
    padding: 17px;
    margin: 8px 0 14px;
}

@media(max-width:600px) {

    .block-container {
        padding-top: 0.45rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-bottom: 1rem;
    }

    .main-title {
        font-size: 1.45rem;
    }

    .sub-title {
        font-size: 0.82rem;
        margin-bottom: 0.55rem;
    }

    .card {
        padding: 9px 11px;
        margin-bottom: 7px;
    }

    .target {
        font-size: 1.28rem;
    }

    .guide {
        padding: 9px 10px;
        font-size: 0.81rem;
    }

    .result-ok,
    .result-no {
        padding: 13px;
    }

    button {
        min-height: 40px !important;
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


INFO = {

    "가슴": (
        "🫁",
        "다른 공명은 최대한 배제하고 가슴 공명만 분명하게 만들어 /아/를 발성하세요.",
    ),

    "입천장": (
        "👄",
        "다른 공명은 최대한 배제하고 입천장·구강 공간의 울림만 분명하게 만들어 /아/를 발성하세요.",
    ),

    "이빨·전방": (
        "🦷",
        "다른 공명은 최대한 배제하고 윗니와 입 앞쪽에 소리가 모이도록 /아/를 발성하세요.",
    ),

    "비강": (
        "👃",
        "다른 공명은 최대한 배제하고 코 주변과 얼굴 중앙의 울림만 분명하게 만들어 /아/를 발성하세요.",
    ),

    "두개골": (
        "💀",
        "다른 공명은 최대한 배제하고 머리 위쪽에서 느껴지는 울림만 분명하게 만들어 /아/를 발성하세요.",
    ),
}


# F0는 의도적으로 분류에서 제외
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


GLOBAL_MEAN = {

    "f1_hz":
        414.53556596260626,

    "f2_hz":
        1537.403933280021,

    "f3_hz":
        2686.63862367215,

    "hnr_db":
        30.701164255199377,

    "spectral_centroid_hz":
        1777.635734511814,

    "spectral_tilt":
        -24.32365232119321,

    "band_80_500_pct":
        91.29624309539795,

    "band_500_1500_pct":
        5.888846766576171,

    "band_1500_3000_pct":
        1.8783772562630474,

    "band_3000_5000_pct":
        0.9365309207438258,
}


GLOBAL_STD = {

    "f1_hz":
        86.41343613459281,

    "f2_hz":
        618.6489094733525,

    "f3_hz":
        327.8617395358285,

    "hnr_db":
        6.198973596714277,

    "spectral_centroid_hz":
        776.5672839668334,

    "spectral_tilt":
        11.710428586714382,

    "band_80_500_pct":
        5.825569561659128,

    "band_500_1500_pct":
        5.2336113647202005,

    "band_1500_3000_pct":
        3.1001735341103616,

    "band_3000_5000_pct":
        1.3656767213953476,
}


CENTROIDS = {

    "가슴": {

        "f1_hz":
            429.3799370666848,

        "f2_hz":
            1017.1615176036128,

        "f3_hz":
            2580.3789875088983,

        "hnr_db":
            24.80149227117852,

        "spectral_centroid_hz":
            1649.8053769813519,

        "spectral_tilt":
            -36.40100501880513,

        "band_80_500_pct":
            95.10030364990234,

        "band_500_1500_pct":
            4.751633865492685,

        "band_1500_3000_pct":
            0.13858066978199138,

        "band_3000_5000_pct":
            0.0094843928569129,
    },


    "입천장": {

        "f1_hz":
            449.63294208713563,

        "f2_hz":
            911.0323934524258,

        "f3_hz":
            2582.03539012955,

        "hnr_db":
            33.95648862528039,

        "spectral_centroid_hz":
            1271.594411523234,

        "spectral_tilt":
            -31.791285104950294,

        "band_80_500_pct":
            89.28483327229817,

        "band_500_1500_pct":
            10.335038105646769,

        "band_1500_3000_pct":
            0.36595392723878223,

        "band_3000_5000_pct":
            0.014173120289342424,
    },


    "이빨·전방": {

        "f1_hz":
            408.7733250987476,

        "f2_hz":
            2108.0104354632076,

        "f3_hz":
            2894.2009432917475,

        "hnr_db":
            24.854095241103376,

        "spectral_centroid_hz":
            2016.0606759589994,

        "spectral_tilt":
            -16.812605367527826,

        "band_80_500_pct":
            87.87335641043526,

        "band_500_1500_pct":
            1.347805184977395,

        "band_1500_3000_pct":
            7.757015739168439,

        "band_3000_5000_pct":
            3.0218154532568797,
    },


    "비강": {

        "f1_hz":
            255.68053047068733,

        "f2_hz":
            1664.5628130759887,

        "f3_hz":
            2240.4529698192177,

        "hnr_db":
            40.5488476802471,

        "spectral_centroid_hz":
            1103.7341741963933,

        "spectral_tilt":
            -27.16501664346996,

        "band_80_500_pct":
            99.5788083757673,

        "band_500_1500_pct":
            0.26623884907790585,

        "band_1500_3000_pct":
            0.1325980467455727,

        "band_3000_5000_pct":
            0.022349016674395056,
    },


    "두개골": {

        "f1_hz":
            504.14154071511206,

        "f2_hz":
            2433.660749538866,

        "f3_hz":
            3210.841422998909,

        "hnr_db":
            27.01966576527247,

        "spectral_centroid_hz":
            3208.4421217480776,

        "spectral_tilt":
            -4.114326054243489,

        "band_80_500_pct":
            86.08063507080078,

        "band_500_1500_pct":
            9.5676668712071,

        "band_1500_3000_pct":
            2.0780402762549266,

        "band_3000_5000_pct":
            2.2736596209662303,
    },
}


def make_sequence():

    sequence = RESONANCES * 2

    random.shuffle(
        sequence
    )

    return sequence


defaults = {

    "sequence":
        make_sequence(),

    "test_index":
        0,

    "audio_bytes":
        None,

    "recorder_id":
        0,

    "page":
        "record",

    "current_features":
        None,

    "current_prediction":
        None,

    "current_distances":
        None,

    "results":
        [],
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
export default function(component){

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


function formatTime(ms){

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


function mergeBuffers(parts){

    let total = 0;

    for(const part of parts){

        total +=
            part.length;
    }

    const merged =
        new Float32Array(
            total
        );

    let offset = 0;

    for(const part of parts){

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
){

    for(
        let i = 0;
        i < text.length;
        i++
    ){

        view.setUint8(
            offset + i,
            text.charCodeAt(i)
        );
    }
}


function encodeWav(
    samples,
    sr
){

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


    for(
        let i = 0;
        i < samples.length;
        i++
    ){

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


function blobToBase64(blob){

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
                            value.indexOf(",")
                            +
                            1
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


function updateTimer(){

    if(!recording){
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


async function cleanup(){

    try{

        if(processor){
            processor.disconnect();
        }

    }catch{}


    try{

        if(source){
            source.disconnect();
        }

    }catch{}


    try{

        if(silentGain){
            silentGain.disconnect();
        }

    }catch{}


    if(stream){

        for(
            const track
            of
            stream.getTracks()
        ){

            track.stop();
        }
    }


    if(audioContext){

        try{

            await audioContext.close();

        }catch{}
    }


    processor = null;

    source = null;

    silentGain = null;

    stream = null;

    audioContext = null;
}


async function stopRecording(){

    if(!recording){
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


async function startRecording(){

    if(recording){
        return;
    }


    try{

        stream =
            await navigator.mediaDevices.getUserMedia({

                audio:{

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

                if(!recording){
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


    }catch(error){

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

        if(!recording){

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


    if(stream){

        for(
            const track
            of
            stream.getTracks()
        ){

            track.stop();
        }
    }


    if(audioContext){

        try{

            audioContext.close();

        }catch{}
    }
};

}
"""


recorder_component = st.components.v2.component(
    "blind_resonance_recorder_v2",
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


        if (
            len(y_trim)
            /
            sr
            <
            2
        ):

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


        tilt = spectral_tilt(
            mean_power,
            frequencies
        )


        return {

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


# =========================================================
# CLASSIFIER
# =========================================================

def classify(
    features
):

    sample = []

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


        sample.append(
            (
                value
                -
                GLOBAL_MEAN[
                    key
                ]
            )
            /
            max(
                GLOBAL_STD[
                    key
                ],
                1e-12
            )
        )


    sample = np.array(
        sample,
        dtype=float
    )


    distances = {}


    for resonance, data in CENTROIDS.items():

        centroid = np.array(
            [
                (
                    data[key]
                    -
                    GLOBAL_MEAN[key]
                )
                /
                max(
                    GLOBAL_STD[key],
                    1e-12
                )
                for key in FEATURES
            ],
            dtype=float
        )


        distances[
            resonance
        ] = float(
            np.linalg.norm(
                sample
                -
                centroid
            )
        )


    ordered = sorted(
        distances.items(),
        key=lambda x: x[1]
    )


    return (
        ordered[0][0],
        ordered
    )


def reset_test():

    st.session_state.sequence = (
        make_sequence()
    )

    st.session_state.test_index = (
        0
    )

    st.session_state.audio_bytes = (
        None
    )

    st.session_state.recorder_id += (
        1
    )

    st.session_state.page = (
        "record"
    )

    st.session_state.current_features = (
        None
    )

    st.session_state.current_prediction = (
        None
    )

    st.session_state.current_distances = (
        None
    )

    st.session_state.results = []


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="main-title">
🎙️ 남성 공명 독립 검증 테스트 2
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="sub-title">
입천장 모델을 새 데이터로 보정한 뒤,
새 발성 10개로 다시 검증합니다.
F0는 분류에 사용하지 않습니다.
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# RECORD
# =========================================================

if st.session_state.page == "record":

    idx = (
        st.session_state.test_index
    )

    target = (
        st.session_state.sequence[
            idx
        ]
    )

    icon, guide = INFO[
        target
    ]


    st.progress(
        idx / 10
    )


    st.markdown(
        f"""
<div class="card">

<div style="font-size:.78rem;color:#777;font-weight:700">
TEST {idx + 1} / 10
</div>

<div class="target">
{icon} {target} 공명을 발성하세요
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
<div class="guide">

<b>/아/를 약 5초간 유지하세요.</b><br>

{guide}<br>

캘리브레이션 때와 같은 방식으로 발성하고,
일부러 결과를 맞추려고 음높이를 바꾸지는 마세요.

</div>
""",
        unsafe_allow_html=True,
    )


    audio = five_second_recorder(
        f"blind2_{st.session_state.recorder_id}"
    )


    if audio:

        st.session_state.audio_bytes = (
            audio
        )

        st.session_state.page = (
            "review"
        )

        st.rerun()


# =========================================================
# REVIEW
# =========================================================

elif st.session_state.page == "review":

    idx = (
        st.session_state.test_index
    )

    target = (
        st.session_state.sequence[
            idx
        ]
    )


    st.markdown(
        f"""
<div class="card">

<b>TEST {idx + 1} / 10</b><br>

{target} 공명 녹음 확인

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
            "분석하기",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "새 모델이 공명을 판별하고 있습니다..."
                ):

                    features = analyze_audio(
                        st.session_state.audio_bytes
                    )

                    prediction, distances = classify(
                        features
                    )


                st.session_state.current_features = (
                    features
                )

                st.session_state.current_prediction = (
                    prediction
                )

                st.session_state.current_distances = (
                    distances
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

    idx = (
        st.session_state.test_index
    )

    target = (
        st.session_state.sequence[
            idx
        ]
    )

    prediction = (
        st.session_state.current_prediction
    )

    distances = (
        st.session_state.current_distances
    )

    features = (
        st.session_state.current_features
    )


    correct = (
        prediction
        ==
        target
    )


    css = (
        "result-ok"
        if correct
        else
        "result-no"
    )


    status = (
        "✅ 일치"
        if correct
        else
        "❌ 불일치"
    )


    st.markdown(
        f"""
<div class="{css}">

<div class="result-title">
{status}
</div>

<div class="result-big">
모델 예측 · {prediction}
</div>

<div>
실제 발성 목표 · <b>{target}</b>
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        "#### 모델이 가까이 본 순서"
    )


    for rank, (
        name,
        distance
    ) in enumerate(
        distances[:3],
        start=1
    ):

        st.markdown(
            f"""
<div class="mini">

<b>{rank}위 · {name}</b><br>

<span style="color:#777;font-size:.85rem">
모델 거리 {distance:.2f}
</span>

</div>
""",
            unsafe_allow_html=True,
        )


    st.caption(
        "이번 분류에도 F0(음높이)는 사용하지 않습니다."
    )


    button_text = (
        "다음 테스트 ➡️"
        if idx < 9
        else
        "최종 결과 보기"
    )


    if st.button(
        button_text,
        type="primary",
        use_container_width=True
    ):

        row = {

            "test":
                idx + 1,

            "target":
                target,

            "prediction":
                prediction,

            "correct":
                correct,

            **features,

            "model_first":
                distances[0][0],

            "model_distance_1":
                distances[0][1],

            "model_second":
                distances[1][0],

            "model_distance_2":
                distances[1][1],
        }


        st.session_state.results.append(
            row
        )


        st.session_state.audio_bytes = (
            None
        )

        st.session_state.current_features = (
            None
        )

        st.session_state.current_prediction = (
            None
        )

        st.session_state.current_distances = (
            None
        )

        st.session_state.recorder_id += (
            1
        )


        if idx >= 9:

            st.session_state.page = (
                "complete"
            )

        else:

            st.session_state.test_index += (
                1
            )

            st.session_state.page = (
                "record"
            )


        st.rerun()


# =========================================================
# COMPLETE
# =========================================================

elif st.session_state.page == "complete":

    df = pd.DataFrame(
        st.session_state.results
    )


    correct_count = int(
        df[
            "correct"
        ].sum()
    )


    accuracy = (
        correct_count
        /
        len(df)
        *
        100
        if len(df)
        else
        0
    )


    st.markdown(
        f"""
<div class="done">

<div style="font-size:1.2rem;font-weight:850">
검증 완료
</div>

<div style="font-size:2.1rem;font-weight:900;margin:5px 0">
{correct_count} / 10
</div>

<div>
정확도 <b>{accuracy:.0f}%</b>
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    summary = (
        df.groupby(
            "target"
        )[
            "correct"
        ]
        .agg(
            [
                "sum",
                "count"
            ]
        )
        .reset_index()
    )


    summary[
        "accuracy_pct"
    ] = (
        summary[
            "sum"
        ]
        /
        summary[
            "count"
        ]
        *
        100
    )


    summary.columns = [
        "공명",
        "정답 수",
        "테스트 수",
        "정확도(%)",
    ]


    st.subheader(
        "공명별 결과"
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    display_df = df[
        [
            "test",
            "target",
            "prediction",
            "correct",
        ]
    ].copy()


    display_df.columns = [
        "회차",
        "목표",
        "모델 예측",
        "일치",
    ]


    st.subheader(
        "10회 상세 결과"
    )


    st.dataframe(
        display_df,
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
        "📥 검증 결과 CSV 다운로드",
        data=csv_bytes,
        file_name="male_resonance_blind_test_v2.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True
    )


    st.caption(
        "CSV를 이 대화에 올려주세요. "
        "입천장 보정 후 실제 독립 테스트 정확도를 다시 확인합니다."
    )


    if st.button(
        "10회 다시 테스트",
        use_container_width=True
    ):

        reset_test()

        st.rerun()
