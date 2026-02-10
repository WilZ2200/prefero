"""分析侠 Prefero — Home page (main entry point for the multi-page app)."""

from __future__ import annotations

import streamlit as st

from utils import init_session_state, language_banner, sidebar_branding, slowbro_next_step

st.set_page_config(
    page_title="分析侠 Prefero - DCE Analysis Tool",
    page_icon="🧮",
    layout="wide",
)

init_session_state()
sidebar_branding()

# ── Multilingual banner ───────────────────────────────────────────
language_banner()

st.markdown(
    '<div style="text-align:right; margin-top:-6px; margin-bottom:4px;">',
    unsafe_allow_html=True,
)
st.markdown(
    "",
    help=(
        "We respect and celebrate linguistic and cultural diversity. "
        "The transliterations above represent our best effort to render "
        "\"Prefero\" faithfully in each writing system. We acknowledge that "
        "automated transliteration may not perfectly capture every nuance. "
        "If you feel a translation could be improved, we warmly welcome "
        "your feedback — please reach out via GitHub Issues. Thank you!"
    ),
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Welcome ────────────────────────────────────────────────────────
st.title("分析侠  Prefero")
st.caption("Discrete Choice Experiment Analysis")
st.markdown(
    """
    Welcome to **分析侠 Prefero**, a **completely code-free** interactive tool
    for specifying, estimating, and comparing discrete-choice models.

    No programming required — just upload your data, point and click, and let
    Slowbro handle the heavy lifting. Whether you're a seasoned econometrician
    or running your first choice experiment, everything works through this
    guided interface.
    """
)

# ── Workflow guide ─────────────────────────────────────────────────
st.subheader("Workflow")

steps = [
    ("1", "Data", "Load or simulate a choice dataset.", "📊"),
    ("2", "Model", "Configure CL, MXL, or Latent Class models.", "⚙️"),
    ("3", "Results", "Inspect estimates, WTP, and predictions.", "📈"),
    ("4", "Compare", "Compare models side-by-side.", "🔄"),
    ("5", "Bootstrap", "Bootstrap inference for robust SEs.", "🎯"),
    ("6", "Community", "Discuss methods and share insights.", "💬"),
]

cols = st.columns(6)
for col, (num, title, desc, icon) in zip(cols, steps):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**Step {num}: {title}**")
            st.caption(desc)

st.divider()
st.info("Use the **sidebar** to navigate between pages.")

# ── The Tale of Slowbro ──────────────────────────────────────────
st.divider()

# Pokémon official artwork from PokeAPI sprites (public)
_slowpoke_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/79.png"
_shellder_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/90.png"
_slowbro_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/80.png"

st.markdown("#### The Tale of Slowbro — Our Mascot")

_story_col1, _story_col2, _story_col3 = st.columns(3)

with _story_col1:
    st.image(_slowpoke_url, width=180, caption="Slowpoke")
    st.markdown(
        """
        Once upon a time, there was a **Slowpoke** who spent its days
        by the river, staring blankly at the water, occasionally
        dipping its tail in to fish... very, very slowly.
        """
    )

with _story_col2:
    st.image(_shellder_url, width=140, caption="Shellder")
    st.markdown(
        """
        One day, a **Shellder** swam by and clamped down hard on
        Slowpoke's tail. About five minutes later, Slowpoke finally
        noticed: *"Huh? Did something just bite me...?"*
        """
    )

with _story_col3:
    st.image(_slowbro_url, width=180, caption="Slowbro")
    st.markdown(
        """
        And just like that, **Slowpoke evolved into Slowbro!**
        Shellder never let go, and Slowbro never bothered to
        check — they've been stuck together ever since.
        """
    )

st.markdown("---")

_bro_col1, _bro_col2 = st.columns([1, 3])

with _bro_col1:
    st.image(_slowbro_url, width=120)

with _bro_col2:
    st.markdown(
        """
        Now, this Slowbro spends every day running **discrete choice models** for everyone:
        Mixed Logit, Latent Class, GMNL, Bootstrap...
        each one more exhausting than the last.

        > *Slowbro is tired. When Slowbro is tired, Slowbro needs coffee.* ☕
        """
    )
    # ── Buy me a coffee placeholder ──
    # TODO: Replace with actual Buy Me a Coffee widget/link
    st.button("☕ Buy Slowbro a Coffee", disabled=True, help="Coming soon!")

# ── Begin the journey ─────────────────────────────────────────────
_NAV_IMG = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites"
    "/pokemon/other/official-artwork/80.png"
)

st.divider()
_j1, _j2 = st.columns([1, 4])
with _j1:
    st.image(_NAV_IMG, width=80)
with _j2:
    st.markdown(
        "<p style='margin:0; color:gray; font-size:0.85em;'>"
        "Ready? Slowbro will walk you through every step.</p>",
        unsafe_allow_html=True,
    )
    if st.button(
        "📊 Begin the journey with Slowbro",
        key="_nav_begin",
        use_container_width=True,
        type="primary",
    ):
        st.switch_page("pages/1_📊_Data.py")
