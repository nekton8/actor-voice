import base64
import io
import tempfile

import librosa
import numpy as np
import parselmouth
import soundfile as sf
import streamlit as st
from parselmouth.praat import call


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="공명 훈련",
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
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.main-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-size: 0.98rem;
    color: #666;
    margin-bottom: 2rem;
    line-height: 1.6;
}

.step-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #777;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

.step-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 16px;
}

.guide-box {
    background: #f6f7f9;
    padding: 20px 22px;
    border-radius: 14px;
    line-height: 1.75;
    margin-bottom: 22px;
}

.complete-box {
    background: #eef9f2;
    border: 1px solid #cae8d4;
    padding: 18px 20px;
    border-radius: 14px;
    margin: 14px 0 20px 0;
}

.complete-title {
    font-size: 1.08rem;
    font-weight: 800;
}

.complete-time {
    color: #2d7b4b;
    margin-top: 5px;
}

/* 결과 */
.result-good {
    background: #eaf7ef;
    border: 1px solid #b8e1c6;
    border-radius: 18px;
    padding: 28px;
    margin: 20px 0;
}

.result-mid {
    background: #fff7e6;
    border: 1px solid #efd28e;
    border-radius: 18px;
    padding: 28px;
    margin: 20px 0;
}

.result-bad {
    background: #fff0f0;
    border: 1px solid #efc0c0;
    border-radius: 18px;
    padding: 28px;
    margin: 20px 0;
}

.result-info {
    background: #eef4ff;
    border: 1px solid #cad9f4;
    border-radius: 18px;
    padding: 28px;
    margin: 20px 0;
}

.result-title {
    font-size: 1.6rem;
    font-weight: 800;
}

.result-value {
    font-size: 3rem;
    font-weight: 900;
    margin: 7px 0;
}

.result-description {
    font-size: 1rem;
    line-height: 1.7;
}

/* 비교 카드 */
.compare-card {
    border: 1px solid #e1e4e8;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    min-height: 125px;
}

.compare-label {
    font-size: 0.9rem;
    color: #6f747b;
    margin-bottom: 7px;
}

.compare-value {
    font-size: 2rem;
    font-weight: 850;
}

.compare-change-up {
    font-size: 0.92rem;
    font-weight: 700;
    color: #25834a;
    margin-top: 4px;
}

.compare-change-down {
    font-size: 0.92rem;
    font-weight: 700;
    color: #c94545;
    margin-top: 4px;
}

.reason-box {
    border: 1px solid #e1e4e8;
    border-radius: 16px;
    padding: 20px 22px;
    margin: 14px 0 22px 0;
}

.tip-box {
    background: #f6f7f9;
    border-radius: 16px;
    padding: 20px 22px;
    line-height: 1.7;
    margin-bottom: 20px;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

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


# =========================================================
# RESONANCE INFO
# =========================================================

RESONANCES = {

    "가슴": {
        "icon": "🫁",
        "description": "가슴 쪽의 깊고 풍부한 울림",
        "guide": (
            "음을 억지로 낮추지 말고 "
            "가슴 쪽 울림을 풍부하게 만들어 보세요."
        ),
    },

    "입천장": {
        "icon": "👄",
        "description": "입천장과 구강 공간을 활용한 울림",
        "guide": (
            "입 안의 공간을 확보하고 "
            "입천장 쪽으로 울림이 퍼지는 느낌을 찾아보세요."
        ),
    },

    "이빨·전방": {
        "icon": "🦷",
        "description": "소리가 앞쪽으로 또렷하게 모이는 울림",
        "guide": (
            "소리를 억지로 밀어내지 말고 "
            "윗니와 입 앞쪽으로 울림이 모이는 느낌을 찾아보세요."
        ),
    },

    "비강": {
        "icon": "👃",
        "description": "코 주변과 얼굴 중앙에서 느껴지는 울림",
        "guide": (
            "코로 소리를 억지로 보내기보다 "
            "얼굴 중앙에 진동이 생기는 느낌을 찾아보세요."
        ),
    },

    "두개골": {
        "icon": "💀",
        "description": "머리 위쪽으로 확장되는 가볍고 높은 울림",
        "guide": (
            "목에 힘을 주지 말고 "
            "소리가 머리 위쪽으로 가볍게 확장되는 느낌을 찾아보세요."
        ),
    },
}


# =========================================================
# CUSTOM 5 SECOND RECORDER
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
        버튼을 누르면 5초 동안 자동으로 녹음됩니다.
    </div>

</div>
"""


RECORDER_CSS = """

.recorder {

    width: 100%;
    height: 100%;

    min-height: 285px;

    border: 1px solid #e2e5e9;
    border-radius: 22px;

    display: flex;
    flex-direction: column;

    align-items: center;
    justify-content: center;

    padding: 28px 22px;

    box-sizing: border-box;

    background: var(--st-background-color);

    font-family: var(--st-font);
}


.record-button {

    width: 108px;
    height: 108px;

    border: none;
    border-radius: 50%;

    background: white;

    cursor: pointer;

    display: flex;
    align-items: center;
    justify-content: center;

    box-shadow:
        0 5px 18px rgba(0,0,0,0.13),
        inset 0 0 0 1px rgba(0,0,0,0.07);

    transition:
        transform .18s ease,
        box-shadow .18s ease,
        background .18s ease;
}


.record-button:hover {

    transform: scale(1.045);

    box-shadow:
        0 8px 24px rgba(0,0,0,0.17),
        inset 0 0 0 1px rgba(0,0,0,0.07);
}


.record-button:active {
    transform: scale(.97);
}


.mic-icon {
    font-size: 45px;
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

    font-size: 34px;

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

    color: #269451;

    font-size: 48px;

    font-weight: 800;
}


.status {

    margin-top: 21px;

    font-size: 17px;

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

    margin-top: 7px;

    font-size: 34px;

    font-weight: 850;

    font-variant-numeric: tabular-nums;

    color: var(--st-text-color);
}


.progress-wrap {

    width: min(360px, 90%);

    height: 7px;

    margin-top: 17px;

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

    margin-top: 15px;

    font-size: 13px;

    color: #7a7f87;

    text-align: center;
}


@keyframes pulse {

    0% {
        box-shadow: 0 0 0 0 rgba(229,57,53,.32);
    }

    70% {
        box-shadow: 0 0 0 18px rgba(229,57,53,0);
    }

    100% {
        box-shadow: 0 0 0 0 rgba(229,57,53,0);
    }
}
"""


RECORDER_JS = """

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
        string
    ) {

        for (
            let i = 0;
            i < string.length;
            i++
        ) {

            view.setUint8(
                offset + i,
                string.charCodeAt(i)
            );
        }
    }


    function encodeWav(
        samples,
        sr
    ) {

        const buffer =
            new ArrayBuffer(
                44 +
                samples.length * 2
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

            let s =
                Math.max(
                    -1,
                    Math.min(
                        1,
                        samples[i]
                    )
                );


            s =
                s < 0
                ? s * 0x8000
                : s * 0x7FFF;


            view.setInt16(
                offset,
                s,
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

                        const comma =
                            value.indexOf(",");

                        resolve(
                            value.substring(
                                comma + 1
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


    function drawTimer() {

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
                drawTimer
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
            "녹음 저장 중...";


        helpText.textContent =
            "잠시만 기다려주세요.";


        const samples =
            mergeBuffers(
                buffers
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
            "5초 녹음이 자동으로 저장되었습니다.";


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
                "5초가 지나면 자동으로 종료됩니다.";


            timerText.textContent =
                "00:00.0";


            progressBar.style.width =
                "0%";


            startTime =
                performance.now();


            drawTimer();


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
                "마이크를 사용할 수 없습니다";


            helpText.textContent =
                "브라우저에서 마이크 사용 권한을 허용해주세요.";
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


recorder_component = (
    st.components.v2.component(
        "resonance_five_second_recorder",
        html=RECORDER_HTML,
        css=RECORDER_CSS,
        js=RECORDER_JS,
    )
)


def five_second_recorder(key):

    result = recorder_component(
        default={
            "audio_b64": None
        },

        on_audio_b64_change=lambda: None,

        key=key,

        width="stretch",

        height=295,
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
# AUDIO FUNCTIONS
# =========================================================

def audio_duration(
    audio_bytes
):

    data, sr = sf.read(
        io.BytesIO(
            audio_bytes
        )
    )

    return float(
        len(data)
        /
        sr
    )


def safe_value(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def percent_change(
    old,
    new
):

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


# =========================================================
# ANALYZE AUDIO
# =========================================================

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


        y, _ = (
            librosa.effects.trim(
                y,
                top_db=35
            )
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


        y = (
            librosa.util.normalize(
                y
            )
        )


        # -----------------------------------------------
        # F0
        # -----------------------------------------------

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


        # -----------------------------------------------
        # SPECTRUM
        # -----------------------------------------------

        spectrum = np.abs(
            librosa.stft(
                y,
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
            librosa.fft_frequencies(
                sr=sr,
                n_fft=2048
            )
        )


        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr
                )
            )
        )


        return {

            "duration":
                duration,

            "f0":
                f0,

            "centroid":
                centroid,

            "band_80_500":
                band_ratio(
                    mean_power,
                    frequencies,
                    80,
                    500
                ),

            "band_500_1500":
                band_ratio(
                    mean_power,
                    frequencies,
                    500,
                    1500
                ),

            "band_1500_3000":
                band_ratio(
                    mean_power,
                    frequencies,
                    1500,
                    3000
                ),

            "band_3000_5000":
                band_ratio(
                    mean_power,
                    frequencies,
                    3000,
                    5000
                ),
        }


# =========================================================
# CHEST JUDGEMENT
# =========================================================

def judge_chest(
    baseline,
    target
):


    low_gain = percent_change(
        baseline["band_80_500"],
        target["band_80_500"]
    )


    centroid_change = (
        (
            baseline["centroid"]
            -
            target["centroid"]
        )
        /
        baseline["centroid"]
        *
        100
    )


    f0_change = abs(
        percent_change(
            baseline["f0"],
            target["f0"]
        )
    )


    low_gain = float(
        np.clip(
            low_gain,
            -50,
            50
        )
    )


    centroid_change = float(
        np.clip(
            centroid_change,
            -30,
            30
        )
    )


    # 1차 연구용 지수
    score = (
        low_gain
        *
        0.8
        +
        centroid_change
        *
        0.2
    )


    score = float(
        np.clip(
            score,
            -50,
            50
        )
    )


    if (
        np.isfinite(f0_change)
        and
        f0_change > 15
    ):

        status = "retry"

        css = "result-mid"

        title = "다시 측정해 주세요"

        message = (
            "두 발성의 음높이가 많이 달라 "
            "가슴 공명 변화만을 정확하게 비교하기 어렵습니다."
        )


    elif score >= 10:

        status = "good"

        css = "result-good"

        title = "잘 되고 있습니다"

        message = (
            "기준 발성보다 가슴 공명과 관련된 "
            "음향 특성이 뚜렷하게 강화되었습니다."
        )


    elif score >= 3:

        status = "mid"

        css = "result-mid"

        title = "방향은 맞습니다"

        message = (
            "가슴 공명 변화가 나타나고 있습니다. "
            "현재 느낌을 유지하면서 울림을 조금 더 만들어 보세요."
        )


    elif score > -3:

        status = "neutral"

        css = "result-mid"

        title = "뚜렷한 변화가 없습니다"

        message = (
            "기준 발성과 비교했을 때 "
            "가슴 공명이 충분히 강화되지는 않았습니다."
        )


    else:

        status = "bad"

        css = "result-bad"

        title = "가슴 공명이 감소했습니다"

        message = (
            "기준 발성보다 가슴 공명과 관련된 "
            "음향 특성이 감소했습니다."
        )


    return {

        "status":
            status,

        "css":
            css,

        "title":
            title,

        "message":
            message,

        "score":
            score,

        "low_gain":
            low_gain,

        "centroid_change":
            centroid_change,

        "f0_change":
            f0_change,
    }


# =========================================================
# HEADER
# =========================================================

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
        f"나의 평소 발성과 비교하여 "
        f"{focus} 공명의 변화를 확인합니다."
    )


st.markdown(
    f"""
<div class="main-title">
{title}
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    f"""
<div class="sub-title">
{subtitle}
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# STEP 1 — SELECT
# =========================================================

if st.session_state.page == "select":

    st.markdown(
        '<div class="step-label">STEP 1</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="step-title">훈련할 공명을 선택하세요</div>',
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

<b>
{info['icon']} {focus} 공명
</b>

<br><br>

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


# =========================================================
# STEP 2 — BASELINE RECORD
# =========================================================

elif st.session_state.page == "baseline_record":

    st.markdown(
        '<div class="step-label">STEP 2</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="step-title">기준 발성 녹음</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        """
<div class="guide-box">

<b>
평소처럼 편안하게 /아/를 발성하세요.
</b>

<br><br>

특정 공명을 만들려고 하지 말고
가장 자연스러운 목소리를 사용합니다.

<br><br>

버튼을 누르면
<b>5초 후 자동으로 녹음이 끝납니다.</b>

</div>
""",
        unsafe_allow_html=True,
    )


    audio = five_second_recorder(
        key=(
            "baseline_"
            +
            str(
                st.session_state.baseline_recorder_id
            )
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


# =========================================================
# BASELINE REVIEW
# =========================================================

elif st.session_state.page == "baseline_review":

    duration = audio_duration(
        st.session_state.baseline_audio
    )


    st.markdown(
        '<div class="step-label">STEP 2</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="step-title">기준 발성 확인</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
<div class="complete-box">

<div class="complete-title">
✅ 녹음 완료
</div>

<div class="complete-time">
녹음 시간 · {duration:.1f}초
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        "#### ▶ 녹음 들어보기"
    )


    st.audio(
        st.session_state.baseline_audio,
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


# =========================================================
# STEP 3 — TARGET RECORD
# =========================================================

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
        f'<div class="step-title">{focus} 공명 발성 녹음</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
<div class="guide-box">

<b>
이번에는 {focus} 공명을 의도해 /아/를 발성하세요.
</b>

<br><br>

기준 발성과 가능한 한
<b>같은 음높이와 비슷한 크기</b>를 유지합니다.

<br><br>

{info['guide']}

<br><br>

녹음은 <b>5초 후 자동으로 종료</b>됩니다.

</div>
""",
        unsafe_allow_html=True,
    )


    audio = five_second_recorder(
        key=(
            "target_"
            +
            str(
                st.session_state.target_recorder_id
            )
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


# =========================================================
# TARGET REVIEW
# =========================================================

elif st.session_state.page == "target_review":

    focus = (
        st.session_state.focus
    )


    duration = audio_duration(
        st.session_state.target_audio
    )


    st.markdown(
        '<div class="step-label">STEP 3</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        f'<div class="step-title">{focus} 공명 발성 확인</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
<div class="complete-box">

<div class="complete-title">
✅ 녹음 완료
</div>

<div class="complete-time">
녹음 시간 · {duration:.1f}초
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        "#### ▶ 녹음 들어보기"
    )


    st.audio(
        st.session_state.target_audio,
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


# =========================================================
# STEP 4 — RESULT
# =========================================================

elif st.session_state.page == "result":

    focus = (
        st.session_state.focus
    )


    st.markdown(
        '<div class="step-label">STEP 4</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        f'<div class="step-title">{focus} 공명 결과</div>',
        unsafe_allow_html=True,
    )


    try:

        with st.spinner(
            "기준 발성과 비교하고 있습니다..."
        ):

            baseline = analyze_audio(
                st.session_state.baseline_audio
            )


            target = analyze_audio(
                st.session_state.target_audio
            )


        # =================================================
        # 가슴 공명 판정
        # =================================================

        if focus == "가슴":

            result = judge_chest(
                baseline,
                target
            )


            score = (
                result["score"]
            )


            if score >= 0:

                score_text = (
                    f"+{score:.1f}%"
                )

            else:

                score_text = (
                    f"{score:.1f}%"
                )


            # ---------------------------------------------
            # 1. 판정
            # ---------------------------------------------

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


            # ---------------------------------------------
            # 2. 기준 / 현재 직관적 비교
            # ---------------------------------------------

            st.markdown(
                "### 가슴 공명 관련 울림"
            )


            baseline_low = (
                baseline[
                    "band_80_500"
                ]
            )


            target_low = (
                target[
                    "band_80_500"
                ]
            )


            difference_pp = (
                target_low
                -
                baseline_low
            )


            col1, col2 = (
                st.columns(2)
            )


            with col1:

                st.markdown(
                    f"""
<div class="compare-card">

<div class="compare-label">
기준 발성
</div>

<div class="compare-value">
{baseline_low:.1f}%
</div>

</div>
""",
                    unsafe_allow_html=True,
                )


            with col2:

                change_class = (
                    "compare-change-up"
                    if difference_pp >= 0
                    else
                    "compare-change-down"
                )


                arrow = (
                    "↑"
                    if difference_pp >= 0
                    else
                    "↓"
                )


                st.markdown(
                    f"""
<div class="compare-card">

<div class="compare-label">
가슴 공명 발성
</div>

<div class="compare-value">
{target_low:.1f}%
</div>

<div class="{change_class}">
{arrow} {difference_pp:+.1f}%p
</div>

</div>
""",
                    unsafe_allow_html=True,
                )


            # ---------------------------------------------
            # 3. 판단 근거
            # ---------------------------------------------

            st.markdown(
                "### 왜 이렇게 판단했나요?"
            )


            st.markdown(
                '<div class="reason-box">',
                unsafe_allow_html=True,
            )


            if result["low_gain"] > 3:

                st.write(
                    "✅ 가슴 공명과 관련된 "
                    f"저역 에너지 비율이 "
                    f"**{result['low_gain']:.1f}% 증가**했습니다."
                )


            elif result["low_gain"] < -3:

                st.write(
                    "❌ 가슴 공명과 관련된 "
                    f"저역 에너지 비율이 "
                    f"**{abs(result['low_gain']):.1f}% 감소**했습니다."
                )


            else:

                st.write(
                    "➖ 가슴 관련 저역 에너지의 "
                    "변화가 크지 않습니다."
                )


            if result["centroid_change"] > 3:

                st.write(
                    "✅ 전체 소리의 에너지 중심도 "
                    "낮은 쪽으로 이동했습니다."
                )


            elif result["centroid_change"] < -3:

                st.write(
                    "➖ 전체 소리의 에너지 중심은 "
                    "오히려 높은 쪽으로 이동했습니다."
                )


            else:

                st.write(
                    "➖ 전체적인 음색 중심 변화는 "
                    "크지 않습니다."
                )


            if np.isfinite(
                result["f0_change"]
            ):

                if result["f0_change"] <= 8:

                    st.write(
                        "✅ 두 발성의 음높이 차이가 "
                        f"**{result['f0_change']:.1f}%**로 "
                        "비교적 안정적입니다."
                    )


                elif result["f0_change"] <= 15:

                    st.write(
                        "⚠️ 두 발성의 음높이가 "
                        f"**{result['f0_change']:.1f}%** "
                        "차이납니다."
                    )


                else:

                    st.write(
                        "⚠️ 두 발성의 음높이가 "
                        f"**{result['f0_change']:.1f}%** 달라 "
                        "공명 변화만으로 보기 어렵습니다."
                    )


            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


            # ---------------------------------------------
            # 4. 다음 발성
            # ---------------------------------------------

            st.markdown(
                "### 다음 발성에서 해볼 것"
            )


            if result["status"] == "good":

                tip = (
                    "지금 만든 울림의 느낌을 기억하세요. "
                    "음높이를 그대로 유지하면서 "
                    "같은 울림을 다시 재현해 보세요."
                )


            elif result["status"] == "mid":

                tip = (
                    "방향은 맞습니다. "
                    "음을 낮추지 말고 "
                    "가슴 쪽의 울림을 조금 더 풍부하게 만들어 보세요."
                )


            elif result["status"] == "retry":

                tip = (
                    "공명보다 음높이가 많이 달라졌습니다. "
                    "기준 발성과 같은 높이의 /아/로 다시 시도해 보세요."
                )


            elif result["status"] == "neutral":

                tip = (
                    "기준 발성과 음높이는 유지하면서 "
                    "가슴 쪽에서 느껴지는 진동을 "
                    "조금 더 확장해 보세요."
                )


            else:

                tip = (
                    "음을 낮추는 것으로 가슴 공명을 만들려고 하지 말고, "
                    "기준 발성의 높이를 유지한 상태에서 "
                    "가슴 쪽 울림을 다시 찾아보세요."
                )


            st.markdown(
                f"""
<div class="tip-box">
{tip}
</div>
""",
                unsafe_allow_html=True,
            )


            # ---------------------------------------------
            # 5. 연구자용 상세
            # ---------------------------------------------

            with st.expander(
                "연구자용 상세 데이터"
            ):

                st.write(
                    f"기준 F0 : "
                    f"{baseline['f0']:.1f} Hz"
                )


                st.write(
                    f"훈련 F0 : "
                    f"{target['f0']:.1f} Hz"
                )


                st.write(
                    f"음높이 차이 : "
                    f"{result['f0_change']:.2f}%"
                )


                st.write(
                    f"기준 80–500 Hz 비율 : "
                    f"{baseline['band_80_500']:.2f}%"
                )


                st.write(
                    f"훈련 80–500 Hz 비율 : "
                    f"{target['band_80_500']:.2f}%"
                )


                st.write(
                    f"저역 변화율 : "
                    f"{result['low_gain']:+.2f}%"
                )


                st.write(
                    f"기준 Spectral Centroid : "
                    f"{baseline['centroid']:.1f} Hz"
                )


                st.write(
                    f"훈련 Spectral Centroid : "
                    f"{target['centroid']:.1f} Hz"
                )


                st.write(
                    f"Centroid 변화 : "
                    f"{result['centroid_change']:+.2f}%"
                )


                st.write(
                    f"가슴 공명 변화 지수 : "
                    f"{result['score']:+.2f}"
                )


                st.markdown("---")


                st.caption(
                    "현재 가슴 공명 변화 지수는 "
                    "80–500 Hz 상대 에너지 변화와 "
                    "Spectral Centroid 변화를 결합한 "
                    "1차 연구용 판정 모델입니다."
                )


        # =================================================
        # 아직 판정모델이 없는 공명
        # =================================================

        else:

            st.markdown(
                f"""
<div class="result-info">

<div class="result-title">
{focus} 공명 분석 데이터가 기록되었습니다
</div>

<div class="result-description">

현재 {focus} 공명은
판정 기준을 구축하는 단계입니다.

<br><br>

충분한 근거가 확보되기 전까지는
잘되고 있다 / 안되고 있다를
임의로 판정하지 않습니다.

</div>

</div>
""",
                unsafe_allow_html=True,
            )


            st.markdown(
                "### 기준 발성 대비 변화"
            )


            bands = [

                (
                    "저역 80–500 Hz",
                    "band_80_500"
                ),

                (
                    "중저역 500–1500 Hz",
                    "band_500_1500"
                ),

                (
                    "중고역 1500–3000 Hz",
                    "band_1500_3000"
                ),

                (
                    "고역 3000–5000 Hz",
                    "band_3000_5000"
                ),
            ]


            for label, key in bands:

                change = (
                    target[key]
                    -
                    baseline[key]
                )


                st.write(
                    f"**{label}** : "
                    f"{change:+.1f}%p"
                )


            f0_change = abs(
                percent_change(
                    baseline["f0"],
                    target["f0"]
                )
            )


            if np.isfinite(
                f0_change
            ):

                st.write(
                    f"**음높이 차이** : "
                    f"{f0_change:.1f}%"
                )


            with st.expander(
                "연구자용 상세 데이터"
            ):

                st.write(
                    f"기준 F0 : "
                    f"{baseline['f0']:.1f} Hz"
                )


                st.write(
                    f"훈련 F0 : "
                    f"{target['f0']:.1f} Hz"
                )


                for label, key in bands:

                    st.write(
                        f"{label} / 기준 : "
                        f"{baseline[key]:.2f}% / "
                        f"훈련 : "
                        f"{target[key]:.2f}%"
                    )


        # =================================================
        # BUTTONS
        # =================================================

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
