import tempfile

import librosa
import numpy as np
import parselmouth
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from parselmouth.praat import call


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
    padding: 18px 20px;
    border-radius: 14px;
    line-height: 1.75;
    margin-bottom: 24px;
}

.recorder-card {
    border: 1px solid #e3e5e8;
    border-radius: 18px;
    padding: 26px 20px 22px 20px;
    text-align: center;
    background: white;
    margin-bottom: 18px;
}

.recorder-ready {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.recorder-help {
    font-size: 0.9rem;
    color: #777;
    margin-bottom: 8px;
}

.complete-box {
    background: #eef9f2;
    border: 1px solid #cae8d4;
    padding: 16px 18px;
    border-radius: 13px;
    margin: 14px 0;
}

.complete-title {
    font-size: 1.05rem;
    font-weight: 800;
}

.complete-time {
    color: #2d7b4b;
    font-size: 0.95rem;
    margin-top: 4px;
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

.big-number {
    font-size: 2rem;
    font-weight: 800;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION
# =========================================================

defaults = {
    "page": "baseline_record",
    "baseline_audio": None,
    "target_audio": None,
    "baseline_recorder_id": 0,
    "target_recorder_id": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# AUDIO UTILS
# =========================================================

def audio_duration(audio_bytes):

    if not audio_bytes:
        return 0.0

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

        return float(
            len(y) / sr
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

        if duration < 1.0:
            raise ValueError(
                "발성이 너무 짧습니다. "
                "2~4초 정도 발성해 주세요."
            )

        # 녹음 음량 차이의 영향을 줄임
        y = librosa.util.normalize(y)

        # -------------------------------------------------
        # F0
        # -------------------------------------------------

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

        # 가슴 공명 관련 저역 비율
        low_ratio = band_ratio(
            mean_power,
            freqs,
            80,
            500
        )

        # 스펙트럼 중심
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
# CHEST RESONANCE
# =========================================================

def judge_chest_resonance(
    baseline,
    target
):

    low_gain = percent_change(
        baseline["low_ratio"],
        target["low_ratio"]
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

    # 1차 연구용 가슴 공명 지수
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

    pitch_unstable = (
        np.isfinite(f0_change)
        and f0_change > 15
    )

    if pitch_unstable:

        status = "retry"
        title = "다시 측정해 주세요"
        css = "result-mid"

        message = (
            "두 발성의 음높이가 너무 달라 "
            "가슴 공명만의 변화를 정확히 판단하기 어렵습니다."
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
            "현재 방향을 유지하면서 울림을 조금 더 만들어 보세요."
        )

    elif chest_index > -3:

        status = "neutral"
        title = "뚜렷한 변화가 없습니다"
        css = "result-mid"

        message = (
            "기준 발성과 비교했을 때 "
            "가슴 공명이 충분히 강화되지는 않았습니다."
        )

    else:

        status = "bad"
        title = "가슴 공명이 감소했습니다"
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
    """
<div class="main-title">
🎙️ 가슴 공명 훈련
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="sub-title">
나의 평소 발성과 비교하여 가슴 공명이
얼마나 강화되었는지 확인합니다.
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# PAGE 1 : BASELINE RECORD
# =========================================================

if st.session_state.page == "baseline_record":

    st.markdown(
        '<div class="step-label">STEP 1</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="step-title">기준 발성 녹음</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="guide-box">

<b>평소처럼 편안한 /아/를 녹음하세요.</b><br><br>

특정 공명을 만들려고 하지 말고
가장 자연스러운 목소리를 사용합니다.<br>

약 <b>2~4초</b> 동안 일정하게 유지하세요.

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="recorder-card">

<div class="recorder-ready">
🎙️ 녹음 준비
</div>

<div class="recorder-help">
아래 마이크 버튼을 누르면 녹음이 시작됩니다.<br>
다시 누르면 녹음이 종료됩니다.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    recording = audio_recorder(
        text="",
        recording_color="#E53935",
        neutral_color="#374151",
        icon_name="microphone",
        icon_size="3x",
        key=(
            "baseline_recorder_"
            +
            str(
                st.session_state.baseline_recorder_id
            )
        ),
    )

    if recording:

        st.session_state.baseline_audio = (
            bytes(recording)
        )

        st.session_state.page = (
            "baseline_review"
        )

        st.rerun()


# =========================================================
# PAGE 2 : BASELINE REVIEW
# =========================================================

elif st.session_state.page == "baseline_review":

    duration = audio_duration(
        st.session_state.baseline_audio
    )

    st.markdown(
        '<div class="step-label">STEP 1</div>',
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
녹음 길이 · {duration:.1f}초
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

    if duration < 1.5:

        st.warning(
            "녹음이 조금 짧습니다. "
            "2~4초 정도로 다시 녹음하는 것을 권장합니다."
        )

    elif duration > 6:

        st.warning(
            "녹음이 조금 깁니다. "
            "2~4초 정도로 녹음하면 비교가 더 안정적입니다."
        )

    else:

        st.success(
            "녹음 길이가 적절합니다."
        )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔄 다시 녹음",
            use_container_width=True
        ):

            st.session_state.baseline_audio = None

            st.session_state.baseline_recorder_id += 1

            st.session_state.page = (
                "baseline_record"
            )

            st.rerun()

    with col2:

        if st.button(
            "이 녹음 사용하기 ➡️",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = (
                "target_record"
            )

            st.rerun()


# =========================================================
# PAGE 3 : TARGET RECORD
# =========================================================

elif st.session_state.page == "target_record":

    st.markdown(
        '<div class="step-label">STEP 2</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="step-title">가슴 공명 발성 녹음</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="guide-box">

<b>이번에는 가슴 공명을 의도해 /아/를 발성하세요.</b><br><br>

기준 발성과 가능한 한
<b>같은 음높이와 비슷한 크기</b>를 유지합니다.<br><br>

음을 일부러 낮추지 말고,
가슴 쪽의 울림만 더 풍부하게 만든다고 생각하세요.<br>

약 <b>2~4초</b> 동안 유지합니다.

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="recorder-card">

<div class="recorder-ready">
🎙️ 가슴 공명 녹음 준비
</div>

<div class="recorder-help">
마이크 버튼을 누르면 녹음이 시작됩니다.<br>
녹음 중에는 버튼이 빨간색으로 표시됩니다.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    recording = audio_recorder(
        text="",
        recording_color="#E53935",
        neutral_color="#374151",
        icon_name="microphone",
        icon_size="3x",
        key=(
            "target_recorder_"
            +
            str(
                st.session_state.target_recorder_id
            )
        ),
    )

    if recording:

        st.session_state.target_audio = (
            bytes(recording)
        )

        st.session_state.page = (
            "target_review"
        )

        st.rerun()

    st.write("")

    if st.button(
        "⬅️ 기준 발성 다시 녹음"
    ):

        st.session_state.baseline_audio = None
        st.session_state.target_audio = None

        st.session_state.baseline_recorder_id += 1
        st.session_state.target_recorder_id += 1

        st.session_state.page = (
            "baseline_record"
        )

        st.rerun()


# =========================================================
# PAGE 4 : TARGET REVIEW
# =========================================================

elif st.session_state.page == "target_review":

    duration = audio_duration(
        st.session_state.target_audio
    )

    st.markdown(
        '<div class="step-label">STEP 2</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="step-title">가슴 공명 발성 확인</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="complete-box">

<div class="complete-title">
✅ 녹음 완료
</div>

<div class="complete-time">
녹음 길이 · {duration:.1f}초
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

    if duration < 1.5:

        st.warning(
            "녹음이 조금 짧습니다. "
            "2~4초 정도로 다시 녹음하는 것을 권장합니다."
        )

    elif duration > 6:

        st.warning(
            "녹음이 조금 깁니다. "
            "2~4초 정도로 녹음하면 비교가 더 안정적입니다."
        )

    else:

        st.success(
            "녹음 길이가 적절합니다."
        )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔄 다시 녹음",
            use_container_width=True
        ):

            st.session_state.target_audio = None

            st.session_state.target_recorder_id += 1

            st.session_state.page = (
                "target_record"
            )

            st.rerun()

    with col2:

        if st.button(
            "이 녹음 분석하기 ➡️",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = (
                "result"
            )

            st.rerun()


# =========================================================
# PAGE 5 : RESULT
# =========================================================

elif st.session_state.page == "result":

    st.markdown(
        '<div class="step-label">STEP 3</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="step-title">가슴 공명 결과</div>',
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

            result = judge_chest_resonance(
                baseline,
                target
            )

        index = result["index"]

        if index >= 0:
            index_text = (
                f"+{index:.1f}%"
            )
        else:
            index_text = (
                f"{index:.1f}%"
            )

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
        # SIMPLE COMPARISON
        # -------------------------------------------------

        st.markdown(
            "### 기준 발성과 비교"
        )

        baseline_low = (
            baseline["low_ratio"]
        )

        target_low = (
            target["low_ratio"]
        )

        change_pp = (
            target_low
            -
            baseline_low
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "기준 발성",
                f"{baseline_low:.1f}%"
            )

        with col2:

            st.metric(
                "가슴 공명 발성",
                f"{target_low:.1f}%",
                delta=f"{change_pp:+.1f}%p"
            )

        # -------------------------------------------------
        # REASON
        # -------------------------------------------------

        st.markdown(
            "### 왜 이렇게 판단했나요?"
        )

        st.markdown(
            '<div class="reason-box">',
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
                "✅ 가슴 공명과 관련된 "
                f"저역 에너지 비율이 **{low_gain:.1f}% 증가**했습니다."
            )

        elif low_gain < -3:

            st.write(
                "❌ 가슴 공명과 관련된 "
                f"저역 에너지 비율이 **{abs(low_gain):.1f}% 감소**했습니다."
            )

        else:

            st.write(
                "➖ 가슴 관련 저역 에너지 변화가 크지 않습니다."
            )

        if centroid_change > 3:

            st.write(
                "✅ 전체 소리의 에너지 중심도 낮은 쪽으로 이동했습니다."
            )

        elif centroid_change < -3:

            st.write(
                "➖ 전체 소리의 에너지 중심은 높은 쪽으로 이동했습니다."
            )

        else:

            st.write(
                "➖ 전체적인 음색 중심 변화는 크지 않습니다."
            )

        if np.isfinite(f0_change):

            if f0_change <= 8:

                st.write(
                    "✅ 두 발성의 음높이 차이가 "
                    f"**{f0_change:.1f}%**로 안정적입니다."
                )

            elif f0_change <= 15:

                st.write(
                    "⚠️ 두 발성의 음높이가 "
                    f"**{f0_change:.1f}%** 차이납니다."
                )

            else:

                st.write(
                    "⚠️ 두 발성의 음높이가 "
                    f"**{f0_change:.1f}%** 달라 "
                    "공명만의 변화로 보기 어렵습니다."
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # TIP
        # -------------------------------------------------

        st.markdown(
            "### 다음 발성"
        )

        if result["status"] == "good":

            st.success(
                "지금 만든 울림의 느낌을 기억하세요. "
                "같은 음높이를 유지하면서 다시 재현해 보세요."
            )

        elif result["status"] == "mid":

            st.info(
                "방향은 맞습니다. "
                "음을 낮추지 않고 가슴 쪽 울림만 "
                "조금 더 풍부하게 만들어 보세요."
            )

        elif result["status"] == "retry":

            st.warning(
                "기준 발성과 같은 음높이로 다시 발성해 보세요."
            )

        else:

            st.info(
                "음을 낮추는 것보다 "
                "기준 발성의 음높이를 유지하면서 "
                "가슴에서 느껴지는 울림을 확장해 보세요."
            )

        # -------------------------------------------------
        # RESEARCH DATA
        # -------------------------------------------------

        with st.expander(
            "연구자용 상세 데이터"
        ):

            st.write(
                f"기준 F0 : {baseline['f0']:.1f} Hz"
            )

            st.write(
                f"훈련 F0 : {target['f0']:.1f} Hz"
            )

            st.write(
                f"음높이 차이 : {result['f0_change']:.2f}%"
            )

            st.write(
                f"기준 저역 비율 : {baseline['low_ratio']:.2f}%"
            )

            st.write(
                f"훈련 저역 비율 : {target['low_ratio']:.2f}%"
            )

            st.write(
                f"저역 변화율 : {result['low_gain']:+.2f}%"
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
                f"가슴 공명 변화 지수 : "
                f"{result['index']:+.2f}"
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🎙️ 가슴 공명 다시 해보기",
                use_container_width=True
            ):

                st.session_state.target_audio = None

                st.session_state.target_recorder_id += 1

                st.session_state.page = (
                    "target_record"
                )

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

                st.session_state.page = (
                    "baseline_record"
                )

                st.rerun()

    except Exception as error:

        st.error(
            f"분석 중 오류가 발생했습니다: {error}"
        )

        if st.button(
            "🎙️ 다시 녹음"
        ):

            st.session_state.target_audio = None

            st.session_state.target_recorder_id += 1

            st.session_state.page = (
                "target_record"
            )

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "공명 훈련 연구용 프로토타입 · 가슴 공명 1차 판정 모델"
)
