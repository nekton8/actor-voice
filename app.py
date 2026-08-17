import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import librosa
import parselmouth
from parselmouth.praat import call
from audio_recorder_streamlit import audio_recorder


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="공명 훈련 음향 비교",
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
    max-width: 860px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.main-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-size: 1rem;
    color: #666;
    margin-bottom: 1.6rem;
}

.small-note {
    font-size: 0.95rem;
    color: #666;
    line-height: 1.65;
    margin-bottom: 1rem;
}

.step-box {
    padding: 18px 20px;
    border-radius: 12px;
    background: #f7f8fa;
    margin-bottom: 20px;
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
    "step": 1,
    "focus": "가슴",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# UTIL
# =========================================================

def safe_median(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def safe_value(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except:
        pass

    return np.nan


def format_value(
    value,
    unit="",
    digits=1
):

    if not np.isfinite(value):
        return "측정 불가"

    return (
        f"{value:.{digits}f}"
        f"{unit}"
    )


def absolute_change(
    baseline,
    target
):

    if (
        not np.isfinite(baseline)
        or
        not np.isfinite(target)
    ):
        return np.nan

    return (
        target
        -
        baseline
    )


def percent_change(
    baseline,
    target
):

    if (
        not np.isfinite(baseline)
        or
        not np.isfinite(target)
        or
        abs(baseline) < 1e-12
    ):
        return np.nan

    return (
        (target - baseline)
        /
        abs(baseline)
        *
        100
    )


def delta_text(
    value,
    unit="",
    digits=1
):

    if not np.isfinite(value):
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
# SPECTRAL FUNCTIONS
# =========================================================

def spectral_slope(
    y,
    sr
):

    spectrum = np.abs(
        librosa.stft(
            y,
            n_fft=2048,
            hop_length=512
        )
    )

    spectrum = (
        spectrum
        +
        1e-10
    )

    freqs = (
        librosa
        .fft_frequencies(
            sr=sr,
            n_fft=2048
        )
    )

    mean_spectrum = (
        np.mean(
            spectrum,
            axis=1
        )
    )

    upper = min(
        5000,
        sr / 2 - 1
    )

    mask = (
        (freqs >= 200)
        &
        (freqs <= upper)
    )

    if np.sum(mask) < 5:
        return np.nan

    x = (
        freqs[mask]
        /
        1000
    )

    y_db = (
        20
        *
        np.log10(
            mean_spectrum[mask]
        )
    )

    slope, _ = np.polyfit(
        x,
        y_db,
        1
    )

    return float(
        slope
    )


def band_ratio(
    mean_power,
    freqs,
    low_frequency,
    high_frequency
):

    band_mask = (
        (freqs >= low_frequency)
        &
        (freqs < high_frequency)
    )

    total_mask = (
        (freqs >= 80)
        &
        (freqs < 5000)
    )

    total = (
        np.sum(
            mean_power[
                total_mask
            ]
        )
        +
        1e-12
    )

    if not np.any(
        band_mask
    ):
        return np.nan

    return float(
        np.sum(
            mean_power[
                band_mask
            ]
        )
        /
        total
        *
        100
    )


# =========================================================
# MAIN AUDIO ANALYSIS
# =========================================================

def analyze_audio(
    audio_bytes
):

    if not audio_bytes:

        raise ValueError(
            "녹음 데이터가 없습니다."
        )

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=True
    ) as tmp:

        tmp.write(
            audio_bytes
        )

        tmp.flush()

        # -------------------------------------------------
        # LOAD AUDIO
        # -------------------------------------------------

        y, sr = librosa.load(
            tmp.name,
            sr=None,
            mono=True
        )

        y, _ = (
            librosa
            .effects
            .trim(
                y,
                top_db=35
            )
        )

        if len(y) < (
            sr * 0.5
        ):

            raise ValueError(
                "유효한 음성이 너무 짧습니다. "
                "2~4초 정도 발성해 주세요."
            )

        y = (
            librosa
            .util
            .normalize(
                y
            )
        )

        # -------------------------------------------------
        # PRAAT SOUND
        # -------------------------------------------------

        sound = (
            parselmouth
            .Sound(
                tmp.name
            )
        )

        # -------------------------------------------------
        # F0
        #
        # to_pitch_ac 대신 Praat call 사용
        # 버전 호환성 문제 방지
        # -------------------------------------------------

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

        # -------------------------------------------------
        # FORMANT
        # -------------------------------------------------

        if (
            np.isfinite(f0)
            and
            f0 < 170
        ):

            max_formant = 5000

        else:

            max_formant = 5500

        formant = call(
            sound,
            "To Formant (burg)",
            0.01,
            5,
            max_formant,
            0.025,
            50
        )

        duration = (
            sound.duration
        )

        times = np.arange(
            0.05,
            max(
                0.06,
                duration - 0.05
            ),
            0.01
        )

        f1_values = []
        f2_values = []
        f3_values = []

        for t in times:

            try:

                f1 = call(
                    formant,
                    "Get value at time",
                    1,
                    float(t),
                    "Hertz",
                    "Linear"
                )

                f2 = call(
                    formant,
                    "Get value at time",
                    2,
                    float(t),
                    "Hertz",
                    "Linear"
                )

                f3 = call(
                    formant,
                    "Get value at time",
                    3,
                    float(t),
                    "Hertz",
                    "Linear"
                )

                if np.isfinite(f1):
                    f1_values.append(f1)

                if np.isfinite(f2):
                    f2_values.append(f2)

                if np.isfinite(f3):
                    f3_values.append(f3)

            except:
                continue

        f1 = safe_median(
            f1_values
        )

        f2 = safe_median(
            f2_values
        )

        f3 = safe_median(
            f3_values
        )

        # -------------------------------------------------
        # HNR
        # -------------------------------------------------

        try:

            harmonicity = call(
                sound,
                "To Harmonicity (cc)",
                0.01,
                60,
                0.1,
                4.5
            )

            hnr = call(
                harmonicity,
                "Get mean",
                0,
                0
            )

            hnr = safe_value(
                hnr
            )

        except:

            hnr = np.nan

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
        # BAND RATIOS
        # -------------------------------------------------

        low = band_ratio(
            mean_power,
            freqs,
            80,
            500
        )

        low_mid = band_ratio(
            mean_power,
            freqs,
            500,
            1500
        )

        mid_high = band_ratio(
            mean_power,
            freqs,
            1500,
            3000
        )

        high = band_ratio(
            mean_power,
            freqs,
            3000,
            5000
        )

        # -------------------------------------------------
        # RETURN
        # -------------------------------------------------

        return {

            "duration":
                float(
                    len(y)
                    /
                    sr
                ),

            "f0":
                f0,

            "f1":
                f1,

            "f2":
                f2,

            "f3":
                f3,

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
                low,

            "low_mid":
                low_mid,

            "mid_high":
                mid_high,

            "high":
                high,
        }


# =========================================================
# STEP 1
# =========================================================

if st.session_state.step == 1:

    st.markdown(
        "## STEP 1. 내 기준 발성"
    )

    st.markdown(
        """
<div class="step-box">

<b>기준 발성 녹음</b><br><br>

특정 공명을 만들려고 하지 말고<br>
가장 편안하고 자연스러운 <b>/아/</b>를<br>
약 2~4초 동안 유지하세요.<br><br>

이 소리가 이후 모든 비교의 개인 기준점이 됩니다.

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

        st.markdown(
            "#### 녹음 확인"
        )

        st.audio(
            st.session_state.baseline_audio,
            format="audio/wav"
        )

        st.success(
            "기준 발성 녹음 완료"
        )

        col1, col2 = (
            st.columns(2)
        )

        with col1:

            if st.button(
                "🔄 다시 녹음",
                use_container_width=True
            ):

                st.session_state.baseline_audio = None

                st.rerun()

        with col2:

            if st.button(
                "다음 단계 ➡️",
                type="primary",
                use_container_width=True
            ):

                st.session_state.step = 2

                st.rerun()


# =========================================================
# STEP 2
# =========================================================

elif st.session_state.step == 2:

    st.markdown(
        "## STEP 2. 훈련 공명 선택"
    )

    options = [
        "가슴",
        "입천장",
        "이빨·전방",
        "비강",
        "두개골",
    ]

    focus = st.selectbox(
        "이번에 훈련할 공명",
        options,
        index=options.index(
            st.session_state.focus
        ),
    )

    st.session_state.focus = (
        focus
    )

    st.markdown(
        f"""
<div class="step-box">

<b>{focus} 공명 발성</b><br><br>

STEP 1에서 녹음한 것과 같은 <b>/아/</b>를 사용하세요.<br><br>

음높이와 음량을 가능한 비슷하게 유지하면서<br>
이번에는 <b>{focus} 공명</b>을 의도적으로 강조합니다.<br><br>

2~4초 정도 유지하세요.

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

        st.markdown(
            "#### 녹음 확인"
        )

        st.audio(
            st.session_state.target_audio,
            format="audio/wav"
        )

        st.success(
            f"{focus} 공명 발성 녹음 완료"
        )

        col1, col2 = (
            st.columns(2)
        )

        with col1:

            if st.button(
                "🔄 다시 녹음",
                use_container_width=True
            ):

                st.session_state.target_audio = None

                st.rerun()

        with col2:

            if st.button(
                "📊 분석 결과 보기",
                type="primary",
                use_container_width=True
            ):

                st.session_state.step = 3

                st.rerun()

    st.write("")

    if st.button(
        "⬅️ 기준 발성 다시 녹음"
    ):

        st.session_state.baseline_audio = None

        st.session_state.target_audio = None

        st.session_state.step = 1

        st.rerun()


# =========================================================
# STEP 3
# =========================================================

elif st.session_state.step == 3:

    focus = (
        st.session_state.focus
    )

    st.markdown(
        "## STEP 3. 분석 결과"
    )

    try:

        with st.spinner(
            "두 발성의 음향 특성을 분석하고 있습니다..."
        ):

            baseline = analyze_audio(
                st.session_state.baseline_audio
            )

            target = analyze_audio(
                st.session_state.target_audio
            )

        st.success(
            "분석 완료"
        )

        st.caption(
            f"분석 대상: {focus} 공명"
        )

        st.info(
            """
현재 단계에서는 특정 공명을 임의의 점수로 환산하지 않습니다.

본인의 기준 발성에서 훈련 발성으로 변화했을 때
실제 음향 지표가 어떻게 달라졌는지를 비교합니다.
"""
        )


        # =================================================
        # CORE
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

            cols = st.columns(3)

            for col, item in zip(
                cols,
                metrics[i:i + 3]
            ):

                label, key, unit = (
                    item
                )

                difference = (
                    absolute_change(
                        baseline[key],
                        target[key]
                    )
                )

                with col:

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
                            +
                            format_value(
                                baseline[key],
                                unit
                            )
                        ),
                    )


        # =================================================
        # ENERGY
        # =================================================

        st.markdown(
            "### 2. 주파수대 에너지 변화"
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

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="기준 발성",
                x=band_labels,
                y=baseline_values,
            )
        )

        fig.add_trace(
            go.Bar(
                name=f"{focus} 공명",
                x=band_labels,
                y=target_values,
            )
        )

        fig.update_layout(
            barmode="group",
            xaxis_title="주파수 대역 (Hz)",
            yaxis_title="상대 에너지 (%)",
            height=380,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =================================================
        # DELTA
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

        delta_fig = (
            go.Figure(
                go.Bar(
                    x=band_labels,
                    y=delta_values,
                )
            )
        )

        delta_fig.add_hline(
            y=0
        )

        delta_fig.update_layout(
            xaxis_title="주파수 대역 (Hz)",
            yaxis_title="변화량 (%p)",
            height=330,
            margin=dict(
                l=20,
                r=20,
                t=25,
                b=20
            ),
        )

        st.plotly_chart(
            delta_fig,
            use_container_width=True
        )


        # =================================================
        # QUALITY
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

        if (
            np.isfinite(
                f0_difference
            )
            and
            f0_difference <= 12
        ):

            st.success(
                f"두 발성의 음높이 차이는 약 "
                f"{f0_difference:.1f}%입니다. "
                f"비교 조건이 비교적 안정적입니다."
            )

        elif np.isfinite(
            f0_difference
        ):

            st.warning(
                f"두 발성의 음높이 차이가 약 "
                f"{f0_difference:.1f}%입니다.\n\n"
                "공명 변화와 음높이 변화가 동시에 "
                "반영되었을 가능성이 있습니다."
            )

        else:

            st.warning(
                "기본주파수를 안정적으로 "
                "측정하지 못했습니다."
            )


        # =================================================
        # RESEARCH
        # =================================================

        st.markdown(
            "### 5. 현재 결과의 의미"
        )

        st.info(
            f"""
이번 결과는 **{focus} 공명을 의도했을 때**
본인의 자연스러운 기준 발성에서 어떤 음향적 변화가
발생했는지를 보여줍니다.

현재 단계에서는 이를 곧바로
'{focus} 공명 점수'로 환산하지 않습니다.

향후 반복 측정과 학생 데이터를 통해
각 공명 발성에서 반복적으로 나타나는 특징을 찾아
공명별 분석 기준을 만들어갈 수 있습니다.
"""
        )


        # =================================================
        # RAW
        # =================================================

        with st.expander(
            "측정 원자료 보기"
        ):

            raw_metrics = {

                "F0 (Hz)":
                    "f0",

                "F1 (Hz)":
                    "f1",

                "F2 (Hz)":
                    "f2",

                "F3 (Hz)":
                    "f3",

                "Spectral centroid":
                    "centroid",

                "Spectral rolloff":
                    "rolloff",

                "Spectral flatness":
                    "flatness",

                "RMS":
                    "rms",

                "Spectral slope":
                    "slope",

                "HNR":
                    "hnr",

                "80–500 Hz (%)":
                    "low",

                "500–1500 Hz (%)":
                    "low_mid",

                "1500–3000 Hz (%)":
                    "mid_high",

                "3000–5000 Hz (%)":
                    "high",
            }

            for label, key in (
                raw_metrics.items()
            ):

                st.write(
                    f"**{label}**  "
                    f"기준: "
                    f"{format_value(baseline[key])} / "
                    f"훈련: "
                    f"{format_value(target[key])} / "
                    f"변화: "
                    f"{delta_text(absolute_change(baseline[key], target[key]))}"
                )


        st.divider()


        col1, col2 = (
            st.columns(2)
        )

        with col1:

            if st.button(
                "⬅️ 공명 발성 다시 녹음",
                use_container_width=True
            ):

                st.session_state.target_audio = None

                st.session_state.step = 2

                st.rerun()


        with col2:

            if st.button(
                "🔄 처음부터 다시 측정",
                type="primary",
                use_container_width=True
            ):

                st.session_state.baseline_audio = None

                st.session_state.target_audio = None

                st.session_state.step = 1

                st.rerun()


    except Exception as error:

        st.error(
            f"분석 중 오류가 발생했습니다: {error}"
        )

        if st.button(
            "⬅️ 공명 발성 다시 녹음"
        ):

            st.session_state.target_audio = None

            st.session_state.step = 2

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "연구용 프로토타입 v1.2 · "
    "개인 기준 발성 대비 상대적 음향 변화 분석"
)
