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
    padding: 18px 20px;
    border-radius: 14px;
    line-height: 1.75;
    margin-bottom: 24px;
}

.select-card {
    border: 1px solid #e3e5e8;
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 12px;
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
        "guide": "음을 억지로 낮추지 말고 가슴 쪽 울림을 풍부하게 만들어 보세요.",
    },

    "입천장": {
        "icon": "👄",
        "description": "입천장과 구강 공간을 활용한 울림",
        "guide": "입 안의 공간을 확보하고 입천장 쪽으로 울림이 퍼지는 느낌을 만들어 보세요.",
    },

    "이빨·전방": {
        "icon": "🦷",
        "description": "소리가 앞쪽으로 또렷하게 모이는 울림",
        "guide": "소리를 밀어내지 말고 윗니와 입 앞쪽으로 울림이 모이는 느낌을 만들어 보세요.",
    },

    "비강": {
        "icon": "👃",
        "description": "코 주변과 얼굴 중앙에서 느껴지는 울림",
        "guide": "코로 소리를 억지로 보내기보다 얼굴 중앙에 진동이 생기는 느낌을 찾아보세요.",
    },

    "두개골": {
        "icon": "💀",
        "description": "머리 위쪽으로 확장되는 가볍고 높은 울림",
        "guide": "목에 힘을 주지 말고 소리가 머리 위쪽으로 가볍게 확장되는 느낌을 만들어 보세요.",
    },
}


# =========================================================
# UTIL
# =========================================================

def audio_duration(audio_bytes):

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

        return float(len(y) / sr)


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
        np.sum(mean_power[total_mask])
        + 1e-12
    )

    return float(
        np.sum(mean_power[band_mask])
        / total
        * 100
    )


# =========================================================
# AUDIO ANALYSIS
# =========================================================

def analyze_audio(audio_bytes):

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

        duration = len(y) / sr

        if duration < 1.0:
            raise ValueError(
                "발성이 너무 짧습니다. 2~4초 정도 발성해 주세요."
            )

        y = librosa.util.normalize(y)

        # ---------------------------------
        # F0
        # ---------------------------------

        sound = parselmouth.Sound(
            tmp.name
        )

        pitch = sound.to_pitch(
            time_step=None,
            pitch_floor=60.0,
            pitch_ceiling=500.0,
        )

        f0_values = pitch.selected_array[
            "frequency"
        ]

        f0_values = f0_values[
            f0_values > 0
        ]

        if len(f0_values):
            f0 = float(np.median(f0_values))
        else:
            f0 = np.nan

        # ---------------------------------
        # SPECTRUM
        # ---------------------------------

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

            "centroid": centroid,

            "band_80_500": band_ratio(
                mean_power,
                freqs,
                80,
                500
            ),

            "band_500_1500": band_ratio(
                mean_power,
                freqs,
                500,
                1500
            ),

            "band_1500_3000": band_ratio(
                mean_power,
                freqs,
                1500,
                3000
            ),

            "band_3000_5000": band_ratio(
                mean_power,
                freqs,
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
        and f0_change > 15
    ):

        return {
            "status": "retry",
            "css": "result-mid",
            "title": "다시 측정해 주세요",
            "score": score,
            "low_gain": low_gain,
            "f0_change": f0_change,
            "centroid_change": centroid_change,

            "message":
                "기준 발성과 음높이가 너무 달라 "
                "가슴 공명만의 변화를 정확하게 비교하기 어렵습니다.",
        }

    if score >= 10:

        return {
            "status": "good",
            "css": "result-good",
            "title": "잘 되고 있습니다",
            "score": score,
            "low_gain": low_gain,
            "f0_change": f0_change,
            "centroid_change": centroid_change,

            "message":
                "기준 발성보다 가슴 공명과 관련된 "
                "음향 특성이 뚜렷하게 강화되었습니다.",
        }

    if score >= 3:

        return {
            "status": "mid",
            "css": "result-mid",
            "title": "방향은 맞습니다",
            "score": score,
            "low_gain": low_gain,
            "f0_change": f0_change,
            "centroid_change": centroid_change,

            "message":
                "가슴 공명 변화가 나타나고 있습니다. "
                "현재 느낌을 유지하면서 울림을 조금 더 만들어 보세요.",
        }

    if score > -3:

        return {
            "status": "neutral",
            "css": "result-mid",
            "title": "뚜렷한 변화가 없습니다",
            "score": score,
            "low_gain": low_gain,
            "f0_change": f0_change,
            "centroid_change": centroid_change,

            "message":
                "기준 발성과 비교했을 때 "
                "가슴 공명이 충분히 강화되지는 않았습니다.",
        }

    return {
        "status": "bad",
        "css": "result-bad",
        "title": "가슴 공명이 감소했습니다",
        "score": score,
        "low_gain": low_gain,
        "f0_change": f0_change,
        "centroid_change": centroid_change,

        "message":
            "기준 발성보다 가슴 공명과 관련된 "
            "저역 특성이 감소했습니다.",
    }


# =========================================================
# HEADER
# =========================================================

if st.session_state.page == "select":

    title = "공명 훈련"

    subtitle = (
        "훈련할 공명을 선택한 뒤 "
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
# PAGE 1 — SELECT
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

    info = RESONANCES[focus]

    st.markdown(
        f"""
<div class="guide-box">

<b>{info['icon']} {focus} 공명</b><br><br>

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

        st.session_state.focus = focus

        st.session_state.page = (
            "baseline_record"
        )

        st.rerun()


# =========================================================
# PAGE 2 — BASELINE RECORD
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

<b>평소처럼 편안한 /아/를 녹음하세요.</b><br><br>

특정 공명을 만들려고 하지 말고
가장 자연스러운 목소리를 사용합니다.<br><br>

약 <b>2~4초</b> 동안 일정하게 유지하세요.

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
            "baseline_"
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

    if 1.5 <= duration <= 6:

        st.success(
            "녹음 길이가 적절합니다."
        )

    else:

        st.warning(
            "2~4초 정도로 다시 녹음하는 것을 권장합니다."
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
# TARGET RECORD
# =========================================================

elif st.session_state.page == "target_record":

    focus = st.session_state.focus

    info = RESONANCES[focus]

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

<b>{focus} 공명을 의도해 /아/를 발성하세요.</b><br><br>

기준 발성과 가능한 한
<b>같은 음높이와 비슷한 크기</b>를 유지합니다.<br><br>

{info['guide']}<br><br>

약 <b>2~4초</b> 동안 유지하세요.

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
            "target_"
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


# =========================================================
# TARGET REVIEW
# =========================================================

elif st.session_state.page == "target_review":

    focus = st.session_state.focus

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
# RESULT
# =========================================================

elif st.session_state.page == "result":

    focus = st.session_state.focus

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


        # =================================================
        # 가슴 공명은 현재 판정 모델 적용
        # =================================================

        if focus == "가슴":

            result = judge_chest(
                baseline,
                target
            )

            score = result["score"]

            if score >= 0:
                score_text = f"+{score:.1f}%"
            else:
                score_text = f"{score:.1f}%"

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
                    "➖ 가슴 관련 저역 에너지 변화가 크지 않습니다."
                )

            if np.isfinite(
                result["f0_change"]
            ):

                if result["f0_change"] <= 8:

                    st.write(
                        "✅ 기준 발성과 음높이가 비슷해 "
                        "비교 조건이 안정적입니다."
                    )

                elif result["f0_change"] <= 15:

                    st.write(
                        "⚠️ 기준 발성과 음높이가 조금 다릅니다."
                    )

                else:

                    st.write(
                        "⚠️ 음높이 차이가 커서 "
                        "공명만의 변화로 보기 어렵습니다."
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


        # =================================================
        # 나머지 4개는 데이터 수집 단계
        # =================================================

        else:

            f0_change = abs(
                percent_change(
                    baseline["f0"],
                    target["f0"]
                )
            )

            st.markdown(
                f"""
<div class="result-info">

<div class="result-title">
{focus} 공명 분석 데이터가 기록되었습니다
</div>

<div class="result-description">
현재 이 공명은 판정 기준을 구축하는 단계입니다.<br><br>

근거가 충분하지 않은 상태에서
'잘되고 있다 / 안되고 있다'를 임의로 판정하지 않습니다.

</div>

</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                "### 현재 확인된 변화"
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

            if np.isfinite(f0_change):

                st.write(
                    f"**음높이 차이** : "
                    f"{f0_change:.1f}%"
                )


        # =================================================
        # 연구자 상세
        # =================================================

        with st.expander(
            "연구자용 상세 데이터"
        ):

            st.write(
                f"선택 공명 : {focus}"
            )

            st.write(
                f"기준 F0 : {baseline['f0']:.1f} Hz"
            )

            st.write(
                f"훈련 F0 : {target['f0']:.1f} Hz"
            )

            st.write(
                f"기준 80–500 Hz : "
                f"{baseline['band_80_500']:.2f}%"
            )

            st.write(
                f"훈련 80–500 Hz : "
                f"{target['band_80_500']:.2f}%"
            )

            st.write(
                f"기준 500–1500 Hz : "
                f"{baseline['band_500_1500']:.2f}%"
            )

            st.write(
                f"훈련 500–1500 Hz : "
                f"{target['band_500_1500']:.2f}%"
            )

            st.write(
                f"기준 1500–3000 Hz : "
                f"{baseline['band_1500_3000']:.2f}%"
            )

            st.write(
                f"훈련 1500–3000 Hz : "
                f"{target['band_1500_3000']:.2f}%"
            )

            st.write(
                f"기준 3000–5000 Hz : "
                f"{baseline['band_3000_5000']:.2f}%"
            )

            st.write(
                f"훈련 3000–5000 Hz : "
                f"{target['band_3000_5000']:.2f}%"
            )

            st.write(
                f"기준 Spectral Centroid : "
                f"{baseline['centroid']:.1f} Hz"
            )

            st.write(
                f"훈련 Spectral Centroid : "
                f"{target['centroid']:.1f} Hz"
            )


        # =================================================
        # BUTTONS
        # =================================================

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                f"🎙️ {focus} 다시 해보기",
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
                "다른 공명 선택",
                type="primary",
                use_container_width=True
            ):

                st.session_state.baseline_audio = None
                st.session_state.target_audio = None

                st.session_state.baseline_recorder_id += 1
                st.session_state.target_recorder_id += 1

                st.session_state.page = (
                    "select"
                )

                st.rerun()


    except Exception as error:

        st.error(
            f"분석 중 오류가 발생했습니다: {error}"
        )
