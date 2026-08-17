import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import librosa
import parselmouth
from parselmouth.praat import call
from audio_recorder_streamlit import audio_recorder

st.set_page_config(
    page_title="공명 훈련 음향 비교",
    page_icon="🎙️",
    layout="centered",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 860px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
.main-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem;
    color: #666;
    margin-bottom: 1.6rem;
}
.small-note {
    font-size: .9rem;
    color: #777;
    line-height: 1.55;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🎙️ 공명 훈련 음향 비교 앱</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    '개인의 기준 발성과 훈련 발성을 비교하여 실제 음향 변화를 확인하는 연구용 프로토타입'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "baseline_audio": None,
    "target_audio": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# ANALYSIS FUNCTIONS
# =========================================================

def safe_median(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    return float(np.median(arr))


def spectral_slope(y, sr):

    spectrum = np.abs(
        librosa.stft(
            y,
            n_fft=2048,
            hop_length=512
        )
    ) + 1e-10

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=2048
    )

    mean_spectrum = np.mean(
        spectrum,
        axis=1
    )

    mask = (
        (freqs >= 200)
        &
        (freqs <= min(5000, sr / 2 - 1))
    )

    if np.sum(mask) < 5:
        return np.nan

    x = freqs[mask] / 1000.0

    y_db = 20 * np.log10(
        mean_spectrum[mask]
    )

    slope, _ = np.polyfit(
        x,
        y_db,
        1
    )

    return float(slope)


def band_ratio(
    mean_power,
    freqs,
    low_frequency,
    high_frequency
):

    mask = (
        (freqs >= low_frequency)
        &
        (freqs < high_frequency)
    )

    total_mask = (
        (freqs >= 80)
        &
        (freqs < min(5000, freqs[-1]))
    )

    total = (
        np.sum(mean_power[total_mask])
        + 1e-12
    )

    if not np.any(mask):
        return np.nan

    return float(
        np.sum(mean_power[mask])
        / total
        * 100.0
    )


def analyze_audio(audio_bytes):

    if not audio_bytes:
        raise ValueError(
            "녹음 데이터가 없습니다."
        )

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=True
    ) as tmp:

        tmp.write(audio_bytes)
        tmp.flush()

        # -------------------------------------------------
        # AUDIO LOAD
        # -------------------------------------------------

        y, sr = librosa.load(
            tmp.name,
            sr=None,
            mono=True
        )

        y, _ = librosa.effects.trim(
            y,
            top_db=35
        )

        if len(y) < int(sr * 0.5):

            raise ValueError(
                "유효한 음성이 너무 짧습니다. "
                "2~4초 정도 길게 발성해 주세요."
            )

        y = librosa.util.normalize(y)

        # -------------------------------------------------
        # PRAAT SOUND
        # -------------------------------------------------

        sound = parselmouth.Sound(
            tmp.name
        )

        # -------------------------------------------------
        # F0
        # -------------------------------------------------

        pitch = sound.to_pitch_ac(
            time_step=0.0,
            pitch_floor=60.0,
            pitch_ceiling=500.0,
        )

        f0_values = (
            pitch
            .selected_array["frequency"]
        )

        f0_values = (
            f0_values[
                f0_values > 0
            ]
        )

        f0 = safe_median(
            f0_values
        )

        # -------------------------------------------------
        # FORMANTS
        # -------------------------------------------------

        if (
            np.isfinite(f0)
            and f0 < 170
        ):
            max_formant = 5000.0
        else:
            max_formant = 5500.0

        formant = sound.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=max_formant,
            window_length=0.025,
            pre_emphasis_from=50.0,
        )

        times = np.arange(
            0.05,
            max(
                0.06,
                sound.duration - 0.05
            ),
            0.01
        )

        f1_values = []
        f2_values = []
        f3_values = []

        for time in times:

            f1 = formant.get_value_at_time(
                1,
                time
            )

            f2 = formant.get_value_at_time(
                2,
                time
            )

            f3 = formant.get_value_at_time(
                3,
                time
            )

            if (
                f1 is not None
                and np.isfinite(f1)
            ):
                f1_values.append(f1)

            if (
                f2 is not None
                and np.isfinite(f2)
            ):
                f2_values.append(f2)

            if (
                f3 is not None
                and np.isfinite(f3)
            ):
                f3_values.append(f3)

        # -------------------------------------------------
        # SPECTRUM
        # -------------------------------------------------

        spectrum = np.abs(
            librosa.stft(
                y,
                n_fft=2048,
                hop_length=512
            )
        )

        power = spectrum ** 2

        mean_power = np.mean(
            power,
            axis=1
        )

        freqs = (
            librosa
            .fft_frequencies(
                sr=sr,
                n_fft=2048
            )
        )

        # -------------------------------------------------
        # SPECTRAL FEATURES
        # -------------------------------------------------

        centroid = float(
            np.mean(
                librosa
                .feature
                .spectral_centroid(
                    y=y,
                    sr=sr
                )
            )
        )

        rolloff = float(
            np.mean(
                librosa
                .feature
                .spectral_rolloff(
                    y=y,
                    sr=sr,
                    roll_percent=0.85
                )
            )
        )

        flatness = float(
            np.mean(
                librosa
                .feature
                .spectral_flatness(
                    y=y
                )
            )
        )

        rms = float(
            np.mean(
                librosa
                .feature
                .rms(
                    y=y
                )
            )
        )

        slope = spectral_slope(
            y,
            sr
        )

        # -------------------------------------------------
        # HNR
        # -------------------------------------------------

        harmonicity = (
            sound
            .to_harmonicity_cc(
                time_step=0.01,
                minimum_pitch=60.0,
                silence_threshold=0.1,
                periods_per_window=4.5
            )
        )

        hnr = call(
            harmonicity,
            "Get mean",
            0,
            0
        )

        if np.isfinite(hnr):
            hnr = float(hnr)
        else:
            hnr = np.nan

        # -------------------------------------------------
        # RETURN
        # -------------------------------------------------

        return {

            "duration":
                float(
                    len(y) / sr
                ),

            "f0":
                f0,

            "f1":
                safe_median(
                    f1_values
                ),

            "f2":
                safe_median(
                    f2_values
                ),

            "f3":
                safe_median(
                    f3_values
                ),

            "centroid":
                centroid,

            "rolloff":
                rolloff,

            "flatness":
                flatness,

            "rms":
                rms,

            "slope":
                slope,

            "hnr":
                hnr,

            "low":
                band_ratio(
                    mean_power,
                    freqs,
                    80,
                    500
                ),

            "low_mid":
                band_ratio(
                    mean_power,
                    freqs,
                    500,
                    1500
                ),

            "mid_high":
                band_ratio(
                    mean_power,
                    freqs,
                    1500,
                    3000
                ),

            "high":
                band_ratio(
                    mean_power,
                    freqs,
                    3000,
                    5000
                ),
        }


def percent_change(
    baseline,
    target
):

    if (
        baseline is None
        or target is None
        or not np.isfinite(baseline)
        or not np.isfinite(target)
        or abs(baseline) < 1e-12
    ):
        return np.nan

    return (
        (target - baseline)
        / abs(baseline)
        * 100.0
    )


def absolute_change(
    baseline,
    target
):

    if (
        baseline is None
        or target is None
        or not np.isfinite(baseline)
        or not np.isfinite(target)
    ):
        return np.nan

    return (
        target
        - baseline
    )


def format_value(
    value,
    unit="",
    digits=1
):

    if (
        value is None
        or not np.isfinite(value)
    ):
        return "측정 불가"

    return (
        f"{value:.{digits}f}"
        f"{unit}"
    )


def delta_text(
    value,
    unit="",
    digits=1
):

    if (
        value is None
        or not np.isfinite(value)
    ):
        return "—"

    sign = (
        "+"
        if value >= 0
        else ""
    )

    return (
        f"{sign}"
        f"{value:.{digits}f}"
        f"{unit}"
    )


# =========================================================
# TRAINING TARGET
# =========================================================

focus = st.selectbox(

    "이번에 훈련할 공명 초점",

    [
        "가슴",
        "입천장",
        "이빨·전방",
        "비강",
        "두개골",
    ],

    help=(
        "현재 단계에서는 이 선택 자체가 "
        "점수 계산에 영향을 주지 않습니다. "
        "어떤 공명을 의도했는지 기록하기 위한 "
        "연구용 라벨입니다."
    ),
)


with st.expander(
    "녹음 방법"
):

    st.markdown(
        """
- 같은 모음 **/아/** 로 2~4초 정도 발성합니다.
- 두 번의 녹음에서 음높이를 일부러 바꾸지 않습니다.
- 음량도 최대한 비슷하게 유지합니다.
- 같은 기기와 같은 거리에서 녹음합니다.
- 첫 번째는 편안하게 발성합니다.
- 두 번째는 선택한 공명만 의도적으로 강조합니다.
"""
    )


# =========================================================
# STEP 1
# =========================================================

st.markdown(
    "### STEP 1. 내 기준 발성"
)

st.markdown(

    """
<div class="small-note">

특정 공명을 만들려고 하지 말고  
가장 편안하고 자연스러운 <b>/아/</b>를  
2~4초 정도 유지하세요.

</div>
""",

    unsafe_allow_html=True,
)


baseline_recording = audio_recorder(

    text="",

    recording_color="#e74c3c",

    neutral_color="#444444",

    icon_name="microphone",

    icon_size="2x",

    key="baseline_recorder",
)


if baseline_recording:

    st.session_state.baseline_audio = (
        bytes(
            baseline_recording
        )
    )


if st.session_state.baseline_audio:

    st.audio(
        st.session_state.baseline_audio,
        format="audio/wav"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "기준 발성 다시 녹음",
            use_container_width=True
        ):

            st.session_state.baseline_audio = None

            st.rerun()

    with col2:

        st.success(
            "기준 발성 저장 완료"
        )


st.divider()


# =========================================================
# STEP 2
# =========================================================

st.markdown(
    f"### STEP 2. {focus} 공명 발성"
)

st.markdown(

    """
<div class="small-note">

STEP 1과 같은 모음과 비슷한 음높이·음량을 유지하세요.  
이번에는 선택한 공명 위치만 의도적으로 강조합니다.

</div>
""",

    unsafe_allow_html=True,
)


target_recording = audio_recorder(

    text="",

    recording_color="#e74c3c",

    neutral_color="#444444",

    icon_name="microphone",

    icon_size="2x",

    key="target_recorder",
)


if target_recording:

    st.session_state.target_audio = (
        bytes(
            target_recording
        )
    )


if st.session_state.target_audio:

    st.audio(
        st.session_state.target_audio,
        format="audio/wav"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "훈련 발성 다시 녹음",
            use_container_width=True
        ):

            st.session_state.target_audio = None

            st.rerun()

    with col2:

        st.success(
            "훈련 발성 저장 완료"
        )


st.divider()


# =========================================================
# ANALYSIS BUTTON
# =========================================================

ready = (

    st.session_state.baseline_audio
    is not None

    and

    st.session_state.target_audio
    is not None
)


analyze_clicked = st.button(

    "📊 두 발성 비교 분석",

    type="primary",

    use_container_width=True,

    disabled=not ready,
)


if not ready:

    st.info(
        "기준 발성과 훈련 발성을 모두 녹음하면 "
        "비교 분석을 시작할 수 있습니다."
    )


# =========================================================
# ANALYSIS RESULT
# =========================================================

if analyze_clicked:

    try:

        with st.spinner(
            "두 발성의 음향 특성을 비교하고 있습니다..."
        ):

            baseline = analyze_audio(
                st.session_state.baseline_audio
            )

            target = analyze_audio(
                st.session_state.target_audio
            )


        st.markdown(
            "## 분석 결과"
        )


        st.caption(

            """
현재 버전은 특정 공명을 몇 점이라고 단정하지 않습니다.

자신의 기준 발성과 비교했을 때  
어떤 음향적 변화가 나타났는지를 분석합니다.
"""
        )


        # =================================================
        # CORE METRICS
        # =================================================

        st.markdown(
            "### 1. 핵심 음향 변화"
        )


        metrics = [

            (
                "기본주파수 F0",
                "f0",
                " Hz"
            ),

            (
                "제1포먼트 F1",
                "f1",
                " Hz"
            ),

            (
                "제2포먼트 F2",
                "f2",
                " Hz"
            ),

            (
                "제3포먼트 F3",
                "f3",
                " Hz"
            ),

            (
                "스펙트럼 중심",
                "centroid",
                " Hz"
            ),

            (
                "HNR",
                "hnr",
                " dB"
            ),

            (
                "스펙트럼 기울기",
                "slope",
                " dB/kHz"
            ),
        ]


        for i in range(
            0,
            len(metrics),
            3
        ):

            columns = st.columns(3)

            for column, item in zip(
                columns,
                metrics[i:i + 3]
            ):

                label, key, unit = item

                difference = absolute_change(
                    baseline[key],
                    target[key]
                )

                with column:

                    st.metric(

                        label,

                        format_value(
                            target[key],
                            unit
                        ),

                        delta=delta_text(
                            difference,
                            unit
                        ),

                        help=(
                            "기준 발성: "
                            + format_value(
                                baseline[key],
                                unit
                            )
                        ),
                    )


        # =================================================
        # BAND ENERGY
        # =================================================

        st.markdown(
            "### 2. 주파수대 에너지 분포 변화"
        )


        band_labels = [

            "80–500",

            "500–1500",

            "1500–3000",

            "3000–5000",
        ]


        band_keys = [

            "low",

            "low_mid",

            "mid_high",

            "high",
        ]


        baseline_values = [

            baseline[key]

            for key
            in band_keys
        ]


        target_values = [

            target[key]

            for key
            in band_keys
        ]


        figure = go.Figure()


        figure.add_trace(

            go.Bar(

                name="기준 발성",

                x=band_labels,

                y=baseline_values,
            )
        )


        figure.add_trace(

            go.Bar(

                name=f"{focus} 훈련 발성",

                x=band_labels,

                y=target_values,
            )
        )


        figure.update_layout(

            barmode="group",

            xaxis_title=(
                "주파수 대역 (Hz)"
            ),

            yaxis_title=(
                "80–5000 Hz 내 상대 에너지 (%)"
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),

            height=380,

            legend=dict(
                orientation="h"
            ),
        )


        st.plotly_chart(
            figure,
            use_container_width=True
        )


        # =================================================
        # DELTA GRAPH
        # =================================================

        st.markdown(
            "### 3. 기준 발성 대비 변화"
        )


        delta_values = [

            absolute_change(
                baseline[key],
                target[key]
            )

            for key
            in band_keys
        ]


        delta_figure = go.Figure(

            go.Bar(

                x=band_labels,

                y=delta_values,
            )
        )


        delta_figure.add_hline(
            y=0
        )


        delta_figure.update_layout(

            xaxis_title=(
                "주파수 대역 (Hz)"
            ),

            yaxis_title=(
                "에너지 비율 변화 (%p)"
            ),

            margin=dict(
                l=20,
                r=20,
                t=25,
                b=20
            ),

            height=330,
        )


        st.plotly_chart(
            delta_figure,
            use_container_width=True
        )


        # =================================================
        # QUALITY CONTROL
        # =================================================

        st.markdown(
            "### 4. 비교 조건 확인"
        )


        f0_difference = abs(

            percent_change(

                baseline["f0"],

                target["f0"]
            )
        )


        duration_difference = abs(

            baseline["duration"]

            - target["duration"]
        )


        if (

            np.isfinite(
                f0_difference
            )

            and

            f0_difference <= 12
        ):

            st.success(

                "두 발성의 기본음 높이 차이: "
                f"약 {f0_difference:.1f}%"

                " — 비교에 무리가 크지 않습니다."
            )

        else:

            if np.isfinite(
                f0_difference
            ):

                st.warning(

                    "두 발성의 기본음 높이 차이가 "
                    f"약 {f0_difference:.1f}%입니다.\n\n"

                    "공명 변화와 음높이 변화가 함께 "
                    "반영되었을 가능성이 있습니다."
                )

            else:

                st.warning(

                    "기본주파수를 안정적으로 "
                    "측정하지 못했습니다."
                )


        if duration_difference > 1.5:

            st.warning(

                "두 녹음의 길이 차이가 큽니다. "
                "다음 측정에서는 비슷한 길이로 "
                "녹음해 주세요."
            )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.markdown(
            "### 연구용 해석"
        )


        st.info(

            f"""
이번 측정은 **{focus} 공명 훈련을 했을 때**
자신의 기준 발성에서 어떤 음향 변화가 나타났는지를 기록한 것입니다.

현재 단계에서는 이 변화를 곧바로 특정 공명의
'정답 점수'로 환산하지 않습니다.

반복 측정과 학생 데이터를 축적한 뒤,
각 공명 훈련에서 반복적으로 나타나는 음향 특징을 찾아
5대 공명 분석 기준을 보정하게 됩니다.
"""
        )


        # =================================================
        # RAW DATA
        # =================================================

        with st.expander(
            "측정 원자료 보기"
        ):

            metric_names = {

                "f0":
                    "F0 (Hz)",

                "f1":
                    "F1 (Hz)",

                "f2":
                    "F2 (Hz)",

                "f3":
                    "F3 (Hz)",

                "centroid":
                    "Spectral centroid (Hz)",

                "rolloff":
                    "Spectral rolloff (Hz)",

                "flatness":
                    "Spectral flatness",

                "rms":
                    "RMS",

                "slope":
                    "Spectral slope (dB/kHz)",

                "hnr":
                    "HNR (dB)",

                "low":
                    "80–500 (%)",

                "low_mid":
                    "500–1500 (%)",

                "mid_high":
                    "1500–3000 (%)",

                "high":
                    "3000–5000 (%)",
            }


            for key, label in metric_names.items():

                baseline_value = baseline[key]

                target_value = target[key]

                difference = absolute_change(
                    baseline_value,
                    target_value
                )

                st.write(

                    f"**{label}** — "

                    f"기준: {format_value(baseline_value)} / "

                    f"훈련: {format_value(target_value)} / "

                    f"변화: {delta_text(difference)}"
                )


        # =================================================
        # RESET
        # =================================================

        if st.button(

            "🔄 처음부터 다시 측정",

            use_container_width=True
        ):

            st.session_state.baseline_audio = None

            st.session_state.target_audio = None

            st.rerun()


    except Exception as error:

        st.error(
            f"분석 중 오류가 발생했습니다: {error}"
        )

        st.caption(
            "녹음이 너무 짧거나 "
            "마이크 입력이 약한 경우 "
            "다시 녹음해 보세요."
        )


st.markdown("---")

st.caption(
    "연구용 프로토타입 v1 · "
    "개인 기준 발성 대비 상대적 음향 변화 분석"
)
