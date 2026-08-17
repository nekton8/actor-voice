import base64
import io
import tempfile

import librosa
import numpy as np
import parselmouth
import soundfile as sf
import streamlit as st


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
    border: 1px solid #cbdafa;
    border-radius: 18px;
    padding: 28px;
    margin: 20px 0;
}

.result-title {
    font-size: 1.55rem;
    font-weight: 800;
}

.result-value {
    font-size: 2.8rem;
    font-weight: 900;
    margin: 5px 0;
}

.result-description {
    font-size: 1rem;
    line-height: 1.7;
}

.reason-box {
    border: 1px solid #e1e1e1;
    border-radius: 15px;
    padding: 20px;
    margin-top: 18px;
}

div[data-testid="stAudio"] {
    margin-top: 4px;
    margin-bottom: 14px;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION
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
# RESONANCE DATA
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
            "입천장 쪽으로 울림이 퍼지는 느낌을 만들어 보세요."
        ),
    },

    "이빨·전방": {
        "icon": "🦷",
        "description": "소리가 앞쪽으로 또렷하게 모이는 울림",
        "guide": (
            "소리를 억지로 밀어내지 말고 "
            "윗니와 입 앞쪽에 울림이 모이는 느낌을 찾아보세요."
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
            "소리가 머리 위쪽으로 가볍게 확장되는 느낌을 만들어 보세요."
        ),
    },
}


# =========================================================
# CUSTOM 5-SECOND RECORDER
# =========================================================

RECORDER_HTML = """
<div class="recorder">

    <button id="recordButton" class="record-button" aria-label="녹음 시작">
        <div class="mic-icon">🎙</div>
    </button>

    <div id="statusText" class="status">
        녹음 버튼을 눌러주세요
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
    min-height: 290px;
    border: 1px solid #e2e5e9;
    border-radius: 22px;
    background: var(--st-background-color);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    padding: 28px 22px;
    font-family: var(--st-font);
}

.record-button {
    width: 104px;
    height: 104px;
    border: 0;
    border-radius: 50%;
    background: #ffffff;
    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.12),
        inset 0 0 0 1px rgba(0,0,0,0.08);
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease,
        background 0.18s ease;
}

.record-button:hover {
    transform: scale(1.04);
    box-shadow:
        0 7px 22px rgba(0, 0, 0, 0.16),
        inset 0 0 0 1px rgba(0,0,0,0.08);
}

.record-button:active {
    transform: scale(0.97);
}

.record-button.recording {
    background: #e53935;
    animation: pulse 1.1s infinite;
}

.record-button.done {
    background: #e8f6ed;
}

.mic-icon {
    font-size: 43px;
    line-height: 1;
}

.record-button.recording .mic-icon {
    font-size: 0;
}

.record-button.recording .mic-icon::after {
    content: "■";
    font-size: 34px;
    color: white;
}

.record-button.done .mic-icon {
    font-size: 0;
}

.record-button.done .mic-icon::after {
    content: "✓";
    font-size: 46px;
    font-weight: 800;
    color: #269451;
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
    color: #238447;
}

.timer {
    font-size: 33px;
    line-height: 1.2;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    margin-top: 7px;
    color: var(--st-text-color);
}

.progress-wrap {
    width: min(360px, 90%);
    height: 7px;
    background: #eceff2;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 17px;
}

.progress {
    width: 0%;
    height: 100%;
    border-radius: 999px;
    background: #e53935;
    transition: width 0.08s linear;
}

.help {
    margin-top: 15px;
    font-size: 13px;
    color: #7a7f87;
    text-align: center;
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(229,57,53,0.32);
    }
    70% {
        box-shadow: 0 0 0 17px rgba(229,57,53,0);
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
        setStateValue,
    } = component;

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
    let zeroGain = null;

    let chunks = [];
    let sampleRate = 44100;

    let startTime = null;
    let timerAnimation = null;
    let stopTimeout = null;


    function formatTime(ms) {

        const seconds = Math.max(
            0,
            Math.min(RECORD_MS, ms)
        ) / 1000;

        return `00:${seconds.toFixed(1).padStart(4, "0")}`;
    }


    function mergeBuffers(buffers) {

        let totalLength = 0;

        for (const buffer of buffers) {
            totalLength += buffer.length;
        }

        const result = new Float32Array(totalLength);

        let offset = 0;

        for (const buffer of buffers) {
            result.set(buffer, offset);
            offset += buffer.length;
        }

        return result;
    }


    function writeString(view, offset, string) {

        for (let i = 0; i < string.length; i++) {
            view.setUint8(
                offset + i,
                string.charCodeAt(i)
            );
        }
    }


    function encodeWav(samples, sr) {

        const buffer = new ArrayBuffer(
            44 + samples.length * 2
        );

        const view = new DataView(buffer);

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

        for (let i = 0; i < samples.length; i++) {

            let sample = Math.max(
                -1,
                Math.min(1, samples[i])
            );

            sample = (
                sample < 0
                    ? sample * 0x8000
                    : sample * 0x7FFF
            );

            view.setInt16(
                offset,
                sample,
                true
            );

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

                const result = reader.result;

                const comma = result.indexOf(",");

                resolve(
                    result.substring(comma + 1)
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

        const elapsed = (
            performance.now()
            -
            startTime
        );

        const capped = Math.min(
            elapsed,
            RECORD_MS
        );

        timerText.textContent = formatTime(
            capped
        );

        const progress = (
            capped
            /
            RECORD_MS
            *
            100
        );

        progressBar.style.width = `${progress}%`;

        timerAnimation = requestAnimationFrame(
            updateTimer
        );
    }


    async function cleanupAudio() {

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
            if (zeroGain) {
                zeroGain.disconnect();
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
        zeroGain = null;
        stream = null;
        audioContext = null;
    }


    async function stopRecording() {

        if (!recording) {
            return;
        }

        recording = false;

        clearTimeout(
            stopTimeout
        );

        cancelAnimationFrame(
            timerAnimation
        );

        timerText.textContent = "00:05.0";
        progressBar.style.width = "100%";

        button.classList.remove(
            "recording"
        );

        statusText.classList.remove(
            "recording"
        );

        statusText.textContent =
            "녹음 처리 중...";

        helpText.textContent =
            "잠시만 기다려주세요.";

        const samples = mergeBuffers(
            chunks
        );

        const wavBlob = encodeWav(
            samples,
            sampleRate
        );

        await cleanupAudio();

        const audioBase64 = await blobToBase64(
            wavBlob
        );

        button.classList.add(
            "done"
        );

        statusText.classList.add(
            "done"
        );

        statusText.textContent =
            "5초 녹음 완료";

        helpText.textContent =
            "녹음이 자동으로 저장되었습니다.";

        /*
        Python으로 WAV 데이터 전달
        */
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

            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    channelCount: 1
                }
            });

            audioContext = new (
                window.AudioContext
                ||
                window.webkitAudioContext
            )();

            sampleRate = audioContext.sampleRate;

            source = audioContext.createMediaStreamSource(
                stream
            );

            /*
            4096 프레임 단위 PCM 수집
            */
            processor = audioContext.createScriptProcessor(
                4096,
                1,
                1
            );

            /*
            브라우저에서 마이크 소리가
            스피커로 나오지 않도록 0 gain 연결
            */
            zeroGain = audioContext.createGain();

            zeroGain.gain.value = 0;

            chunks = [];

            processor.onaudioprocess = (event) => {

                if (!recording) {
                    return;
                }

                const input = event.inputBuffer
                    .getChannelData(0);

                chunks.push(
                    new Float32Array(input)
                );
            };

            source.connect(
                processor
            );

            processor.connect(
                zeroGain
            );

            zeroGain.connect(
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
                "5초가 지나면 자동으로 종료됩니다.";

            timerText.textContent =
                "00:00.0";

            progressBar.style.width =
                "0%";

            startTime = performance.now();

            updateTimer();

            stopTimeout = setTimeout(
                stopRecording,
                RECORD_MS
            );

        } catch (error) {

            console.error(error);

            statusText.textContent =
                "마이크를 사용할 수 없습니다";

            helpText.textContent =
                "브라우저의 마이크 사용 권한을 허용해주세요.";
        }
    }


    button.onclick = () => {

        if (!recording) {
            startRecording();
        }
    };


    return () => {

        clearTimeout(
            stopTimeout
        );

        cancelAnimationFrame(
            timerAnimation
        );

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
    "five_second_voice_recorder",
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
        height=300,
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
# AUDIO UTILS
# =========================================================

def audio_duration(audio_bytes):

    data, sr = sf.read(
        io.BytesIO(audio_bytes)
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
        or not np.isfinite(new)
        or abs(old) < 1e-10
    ):

        return np.nan

    return (
        (new - old)
        / abs(old)
        * 100
    )


def band_ratio(
    mean_power,
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
            mean_power[total_mask]
        )
        +
        1e-12
    )

    return float(
        np.sum(
            mean_power[band_mask]
        )
        /
        total
        *
        100
    )


# =========================================================
# AUDIO ANALYSIS
# =========================================================

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
            len(y) / sr
        )

        if duration < 2:

            raise ValueError(
                "유효한 발성 시간이 너무 짧습니다."
            )

        y = librosa.util.normalize(
            y
        )

        # ---------------------------------------------
        # PITCH
        # ---------------------------------------------

        sound = parselmouth.Sound(
            tmp.name
        )

        pitch = sound.to_pitch(
            pitch_floor=60,
            pitch_ceiling=500
        )

        f0_values = pitch.selected_array[
            "frequency"
        ]

        f0_values = f0_values[
            f0_values > 0
        ]

        if len(f0_values):

            f0 = float(
                np.median(
                    f0_values
                )
            )

        else:

            f0 = np.nan

        # ---------------------------------------------
        # SPECTRUM
        # ---------------------------------------------

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

        freqs = librosa.fft_frequencies(
            sr=sr,
            n_fft=2048
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


# =========================================================
# CHEST MODEL
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

    score = (
        low_gain * 0.8
        +
        centroid_change * 0.2
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
            "기준 발성과 음높이가 너무 달라 "
            "공명만의 변화를 정확하게 비교하기 어렵습니다."
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
            "현재 울림을 조금 더 확장해 보세요."
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

    focus = st.session_state.focus

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

    info = RESONANCES[
        focus
    ]

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
# STEP 2 — BASELINE
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

녹음 버튼을 누르면
<b>정확히 5초 후 자동으로 종료</b>됩니다.

</div>
""",
        unsafe_allow_html=True,
    )

    audio = five_second_recorder(
        key=(
            "baseline_recorder_"
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

    st.markdown(
        '<div class="step-label">STEP 2</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="step-title">기준 발성 확인</div>',
        unsafe_allow_html=True,
    )

    duration = audio_duration(
        st.session_state.baseline_audio
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
# STEP 3 — TARGET
# =========================================================

elif st.session_state.page == "target_record":

    focus = (
        st.session_state.focus
    )

    info = (
        RESONANCES[focus]
    )

    st.markdown(
        '<div class="step-label">STEP 3</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="step-title">{focus} 공명 발성</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="guide-box">

<b>
{focus} 공명을 의도해 /아/를 발성하세요.
</b>

<br><br>

기준 발성과 가능한 한
<b>같은 음높이와 비슷한 크기</b>를 유지하세요.

<br><br>

{info['guide']}

<br><br>

버튼을 누르면
<b>5초 동안 자동 녹음</b>됩니다.

</div>
""",
        unsafe_allow_html=True,
    )

    audio = five_second_recorder(
        key=(
            "target_recorder_"
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
✅ 5초 녹음 완료
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
# RESULT
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
            "두 발성을 비교하고 있습니다..."
        ):

            baseline = analyze_audio(
                st.session_state.baseline_audio
            )

            target = analyze_audio(
                st.session_state.target_audio
            )


        # =============================================
        # CHEST
        # =============================================

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

            st.markdown(
                "### 왜 이렇게 판단했나요?"
            )

            st.markdown(
                '<div class="reason-box">',
                unsafe_allow_html=True,
            )

            if result["low_gain"] > 3:

                st.write(
                    "✅ 가슴 관련 저역 에너지가 "
                    f"**{result['low_gain']:.1f}% 증가**했습니다."
                )

            elif result["low_gain"] < -3:

                st.write(
                    "❌ 가슴 관련 저역 에너지가 "
                    f"**{abs(result['low_gain']):.1f}% 감소**했습니다."
                )

            else:

                st.write(
                    "➖ 가슴 관련 저역 에너지의 "
                    "변화가 크지 않습니다."
                )

            if np.isfinite(
                result["f0_change"]
            ):

                if (
                    result["f0_change"]
                    <=
                    8
                ):

                    st.write(
                        "✅ 기준 발성과 음높이가 비슷해 "
                        "비교 조건이 안정적입니다."
                    )

                elif (
                    result["f0_change"]
                    <=
                    15
                ):

                    st.write(
                        "⚠️ 기준 발성과 음높이가 "
                        "조금 다릅니다."
                    )

                else:

                    st.write(
                        "⚠️ 음높이 차이가 커서 "
                        "공명 변화만으로 보기 어렵습니다."
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


        # =============================================
        # OTHER RESONANCES
        # =============================================

        else:

            st.markdown(
                f"""
<div class="result-info">

<div class="result-title">
{focus} 공명 데이터 기록 완료
</div>

<div class="result-description">

현재 이 공명은 판정 기준을 구축하고 있는 단계입니다.

<br><br>

근거가 충분히 확보되기 전까지는
잘되고 있다 / 안되고 있다를
임의로 판정하지 않습니다.

</div>

</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                "### 기준 발성 대비 음향 변화"
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

                change = (
                    target[key]
                    -
                    baseline[key]
                )

                st.write(
                    f"**{label}** "
                    f"{change:+.1f}%p"
                )


        # =============================================
        # BUTTONS
        # =============================================

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
