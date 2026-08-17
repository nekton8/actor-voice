import tempfile
import numpy as np
import streamlit as st
import librosa
import parselmouth
from parselmouth.praat import call
from audio_recorder_streamlit import audio_recorder


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="가슴 공명 훈련",
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
    max-width: 820px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-size: 1rem;
    color: #666;
    margin-bottom: 2rem;
}

.guide-box {
    background: #f5f6f8;
    border-radius: 14px;
    padding: 20px 22px;
    line-height: 1.7;
    margin-bottom: 20px;
}

.result-good {
    background: #eaf7ef;
    border: 1px solid #b8e1c6;
    border-radius: 16px;
    padding: 28px;
    margin: 20px 0;
}

.result-mid {
    background: #fff7e6;
    border: 1px solid #f1d28c;
    border-radius: 16px;
    padding: 28px;
    margin: 20px 0;
}

.result-bad {
    background: #fff0f0;
    border: 1px solid #efc0c0;
    border-radius: 16px;
    padding: 28px;
    margin: 20px 0;
}

.result-title {
    font-size: 1.65rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-value {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 8px 0;
}

.result-description {
    font-size: 1rem;
    line-height: 1.7;
}

.reason-box {
    border: 1px solid #ddd;
    border-radius: 14px;
    padding: 20px;
    margin-top: 18px;
}

.reason-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "step": 1,
    "baseline_audio": None,
    "target_audio": None,
    "baseline_recorder_id": 0,
    "target_recorder_id": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# UTILITIES
# =========================================================

def safe_value(value):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except:
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

        # -------------------------
        # LOAD
        # -------------------------

        y, sr = librosa.load(
            tmp.name,
            sr=None,
            mono=True
        )

        y, _ = librosa.effects.trim(
            y,
            top_db=35
        )

        duration = len(y) / sr

        if duration < 1.0:

            raise ValueError(
                "발성이 너무 짧습니다. "
                "약 2~4초 동안 발성해 주세요."
            )

        # 음량 자체의 영향을 줄이기 위해 정규화
        y = librosa.util.normalize(y)

        # -------------------------
        # PRAAT F0
        # -------------------------

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

        f0 = safe_value(f0)

        # -------------------------
        # SPECTRUM
        # -------------------------

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

        freqs = librosa.fft_frequencies(
            sr=sr,
            n_fft=2048
        )

        # 가슴 공명 관련 저역 비율
        low_ratio = band_ratio(
            mean_power,
            freqs,
            80,
            500
        )

        # 전체 스펙트럼 무게 중심
        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr
                )
            )
        )

        return {
            "duration": duration,
            "f0": f0,
            "low_ratio": low_ratio,
            "centroid": centroid,
        }


# =========================================================
# CHEST RESONANCE JUDGEMENT
# =========================================================

def judge_chest_resonance(
    baseline,
    target
):

    # -------------------------------------------------
    # 1. 가슴 관련 저역 에너지 변화
    # -------------------------------------------------

    low_gain = percent_change(
        baseline["low_ratio"],
        target["low_ratio"]
    )

    # -------------------------------------------------
    # 2. 스펙트럼 중심 변화
    #
    # 중심이 낮아질수록 양수로 계산
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 3. 음높이 변화
    # -------------------------------------------------

    f0_change = abs(
        percent_change(
            baseline["f0"],
            target["f0"]
        )
    )

    # -------------------------------------------------
    # 이상치 방지
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 가슴 공명 변화 지수
    #
    # 핵심:
    # 저역 에너지 변화 80%
    # 스펙트럼 중심 변화 20%
    # -------------------------------------------------

    chest_index = (
        low_gain * 0.8
        +
        centroid_change * 0.2
    )

    chest_index = float(
        np.clip(
            chest_index,
            -50,
            50
        )
    )

    # -------------------------------------------------
    # 음높이가 너무 달라지면 신뢰도 낮음
    # -------------------------------------------------

    pitch_unstable = (
        np.isfinite(f0_change)
        and
        f0_change > 15
    )

    # -------------------------------------------------
    # 판정
    # -------------------------------------------------

    if pitch_unstable:

        status = "retry"
        title = "다시 측정하는 것이 좋습니다"
        css = "result-mid"

        message = (
            "두 발성의 음높이 차이가 커서 "
            "가슴 공명 변화만을 정확하게 비교하기 어렵습니다."
        )

    elif chest_index >= 10:

        status = "good"
        title = "잘 되고 있습니다"
        css = "result-good"

        message = (
            "기준 발성보다 가슴 공명과 관련된 "
            "음향 특성이 뚜렷하게 강화되었습니다."
        )

    elif chest_index >= 3:

        status = "mid"
        title = "방향은 맞습니다"
        css = "result-mid"

        message = (
            "가슴 공명 변화가 나타나고 있습니다. "
            "현재 방향을 유지하면서 울림을 조금 더 확장해 보세요."
        )

    elif chest_index > -3:

        status = "neutral"
        title = "뚜렷한 변화가 없습니다"
        css = "result-mid"

        message = (
            "기준 발성과 비교했을 때 "
            "가슴 공명이 뚜렷하게 강화되지는 않았습니다."
        )

    else:

        status = "bad"
        title = "가슴 공명이 오히려 감소했습니다"
        css = "result-bad"

        message = (
            "기준 발성보다 가슴 공명과 관련된 "
            "저역 특성이 감소했습니다."
        )

    return {
        "index": chest_index,
        "low_gain": low_gain,
        "centroid_change": centroid_change,
        "f0_change": f0_change,
        "status": status,
        "title": title,
        "css": css,
        "message": message,
    }


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🎙️ 가슴 공명 훈련'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    '나의 평소 발성과 비교하여 가슴 공명이 얼마나 강화되었는지 확인합니다.'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# STEP 1
# =========================================================

if st.session_state.step == 1:

    st.markdown(
        "## STEP 1. 기준 발성"
    )

    st.markdown(
        """
<div class="guide-box">

<b>평소처럼 편안하게 /아/를 발성하세요.</b><br><br>

특정 부위에 울림을 만들려고 하지 말고<br>
가장 자연스러운 목소리로 약 <b>2~4초</b> 동안 발성합니다.

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
        key=(
            "baseline_"
            +
            str(
                st.session_state.baseline_recorder_id
            )
        ),
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

        st.success(
            "기준 발성 녹음 완료"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔄 다시 녹음",
                use_container_width=True
            ):

                st.session_state.baseline_audio = None

                st.session_state.baseline_recorder_id += 1

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
        "## STEP 2. 가슴 공명 발성"
    )

    st.markdown(
        """
<div class="guide-box">

<b>이번에는 가슴 공명을 의도해 /아/를 발성하세요.</b><br><br>

STEP 1과 가능한 한 <b>같은 음높이와 비슷한 크기</b>로 발성합니다.<br><br>

음높이를 일부러 낮추지 말고,<br>
가슴 쪽 울림만 더 풍부하게 만든다고 생각해 보세요.<br><br>

약 <b>2~4초</b> 동안 유지합니다.

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
        key=(
            "target_"
            +
            str(
                st.session_state.target_recorder_id
            )
        ),
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

        st.success(
            "가슴 공명 발성 녹음 완료"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔄 다시 녹음",
                use_container_width=True
            ):

                st.session_state.target_audio = None

                st.session_state.target_recorder_id += 1

                st.rerun()

        with col2:

            if st.button(
                "결과 확인 ➡️",
                type="primary",
                use_container_width=True
            ):

                st.session_state.step = 3

                st.rerun()

    st.write("")

    if st.button(
        "⬅️ 기준 발성부터 다시"
    ):

        st.session_state.baseline_audio = None
        st.session_state.target_audio = None

        st.session_state.baseline_recorder_id += 1
        st.session_state.target_recorder_id += 1

        st.session_state.step = 1

        st.rerun()


# =========================================================
# STEP 3
# =========================================================

elif st.session_state.step == 3:

    st.markdown(
        "## STEP 3. 가슴 공명 결과"
    )

    try:

        with st.spinner(
            "가슴 공명 변화를 분석하고 있습니다..."
        ):

            baseline = analyze_audio(
                st.session_state.baseline_audio
            )

            target = analyze_audio(
                st.session_state.target_audio
            )

            result = judge_chest_resonance(
                baseline,
                target
            )

        # -------------------------------------------------
        # MAIN RESULT
        # -------------------------------------------------

        index = result["index"]

        if index >= 0:
            index_text = f"+{index:.1f}%"
        else:
            index_text = f"{index:.1f}%"

        st.markdown(
            f"""
<div class="{result['css']}">

<div class="result-title">
{result['title']}
</div>

<div class="result-value">
{index_text}
</div>

<div class="result-description">
{result['message']}
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # SIMPLE VISUAL
        # -------------------------------------------------

        st.markdown(
            "### 가슴 관련 저역 에너지"
        )

        baseline_low = (
            baseline["low_ratio"]
        )

        target_low = (
            target["low_ratio"]
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "기준 발성",
                f"{baseline_low:.1f}%"
            )

        with col2:

            change = (
                target_low
                -
                baseline_low
            )

            st.metric(
                "가슴 공명 발성",
                f"{target_low:.1f}%",
                delta=f"{change:+.1f}%p"
            )

        st.write("")

        # -------------------------------------------------
        # REASONS
        # -------------------------------------------------

        st.markdown(
            """
<div class="reason-box">
<div class="reason-title">
왜 이렇게 판단했나요?
</div>
""",
            unsafe_allow_html=True,
        )

        low_gain = (
            result["low_gain"]
        )

        centroid_change = (
            result["centroid_change"]
        )

        f0_change = (
            result["f0_change"]
        )

        if low_gain > 3:

            st.write(
                f"✅ 가슴 공명과 관련된 저역 에너지 비율이 "
                f"기준 발성보다 **{low_gain:.1f}% 증가**했습니다."
            )

        elif low_gain < -3:

            st.write(
                f"❌ 가슴 공명과 관련된 저역 에너지 비율이 "
                f"기준 발성보다 **{abs(low_gain):.1f}% 감소**했습니다."
            )

        else:

            st.write(
                "➖ 가슴 관련 저역 에너지의 변화가 크지 않습니다."
            )

        if centroid_change > 3:

            st.write(
                "✅ 소리의 에너지 중심이 낮은 쪽으로 이동했습니다."
            )

        elif centroid_change < -3:

            st.write(
                "➖ 소리의 에너지 중심은 오히려 높은 쪽으로 이동했습니다."
            )

        else:

            st.write(
                "➖ 전체적인 음색 중심 변화는 크지 않습니다."
            )

        if np.isfinite(f0_change):

            if f0_change <= 8:

                st.write(
                    f"✅ 두 발성의 음높이 차이는 "
                    f"**{f0_change:.1f}%**로 비교적 안정적입니다."
                )

            elif f0_change <= 15:

                st.write(
                    f"⚠️ 두 발성의 음높이가 "
                    f"**{f0_change:.1f}%** 차이납니다."
                )

            else:

                st.write(
                    f"⚠️ 음높이가 **{f0_change:.1f}%** 달라 "
                    f"공명만의 변화로 보기 어렵습니다."
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # TRAINING TIP
        # -------------------------------------------------

        st.markdown(
            "### 다음 발성에서 해볼 것"
        )

        if result["status"] == "good":

            st.success(
                "지금 만든 울림의 느낌을 기억하세요. "
                "음높이를 유지하면서 같은 울림을 다시 재현해 보세요."
            )

        elif result["status"] == "mid":

            st.info(
                "방향은 맞습니다. "
                "목소리를 억지로 낮추지 말고 "
                "현재 울림을 조금 더 풍부하게 만들어 보세요."
            )

        elif result["status"] == "retry":

            st.warning(
                "공명보다 음높이가 많이 달라졌습니다. "
                "기준 발성과 같은 높이의 /아/를 다시 만들어 보세요."
            )

        else:

            st.info(
                "음을 낮추려고 하기보다 "
                "기준 발성의 음높이를 유지하면서 "
                "가슴 쪽에서 느껴지는 울림을 확장해 보세요."
            )

        # -------------------------------------------------
        # RESEARCH DATA
        # -------------------------------------------------

        with st.expander(
            "연구자용 상세 데이터"
        ):

            st.write(
                f"기준 F0: {baseline['f0']:.1f} Hz"
            )

            st.write(
                f"훈련 F0: {target['f0']:.1f} Hz"
            )

            st.write(
                f"기준 저역 비율: {baseline['low_ratio']:.2f}%"
            )

            st.write(
                f"훈련 저역 비율: {target['low_ratio']:.2f}%"
            )

            st.write(
                f"저역 변화율: {result['low_gain']:+.2f}%"
            )

            st.write(
                f"기준 Spectral Centroid: "
                f"{baseline['centroid']:.1f} Hz"
            )

            st.write(
                f"훈련 Spectral Centroid: "
                f"{target['centroid']:.1f} Hz"
            )

            st.write(
                f"Centroid 변화: "
                f"{result['centroid_change']:+.2f}%"
            )

            st.write(
                f"가슴 공명 변화 지수: "
                f"{result['index']:+.2f}"
            )

            st.markdown("---")

            st.caption(
                "현재 가슴 공명 변화 지수는 "
                "80–500 Hz 상대 에너지 변화 80%와 "
                "Spectral Centroid 변화 20%를 결합한 "
                "1차 연구용 휴리스틱입니다. "
                "향후 반복 측정 및 학생 데이터를 통해 "
                "가중치와 판정 기준을 보정합니다."
            )

        # -------------------------------------------------
        # BUTTONS
        # -------------------------------------------------

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🎙️ 가슴 공명 다시 해보기",
                use_container_width=True
            ):

                st.session_state.target_audio = None

                st.session_state.target_recorder_id += 1

                st.session_state.step = 2

                st.rerun()

        with col2:

            if st.button(
                "🔄 처음부터 다시",
                type="primary",
                use_container_width=True
            ):

                st.session_state.baseline_audio = None
                st.session_state.target_audio = None

                st.session_state.baseline_recorder_id += 1
                st.session_state.target_recorder_id += 1

                st.session_state.step = 1

                st.rerun()

    except Exception as error:

        st.error(
            f"분석 중 오류가 발생했습니다: {error}"
        )

        if st.button(
            "가슴 공명 다시 녹음"
        ):

            st.session_state.target_audio = None

            st.session_state.target_recorder_id += 1

            st.session_state.step = 2

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "공명 훈련 연구용 프로토타입 · 가슴 공명 1차 판정 모델"
)
