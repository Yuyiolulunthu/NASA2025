# Web/components/top_banner.py
import streamlit as st

def render_top_banner(key_prefix: str = "top"):
    # ... your layout code (logo, title, etc.) ...

    demo_clicked = st.button(
        "Demo",
        key=f"{key_prefix}-demo",
        use_container_width=True,
    )
    analyze_clicked = st.button(
        "Analyze",
        key=f"{key_prefix}-analyze",
        use_container_width=True,
    )
    vetting_clicked = st.button(
        "Vetting",
        key=f"{key_prefix}-vetting",
        use_container_width=True,
    )

    # If you have a “hamburger/more” menu, key those widgets too:
    with st.popover("☰ More", key=f"{key_prefix}-more"):
        about = st.button("About our models", key=f"{key_prefix}-about")
        analyze_demo = st.button("Analyze (Demo)", key=f"{key_prefix}-analyze-demo")
        vet_demo = st.button("Vetting (Demo)", key=f"{key_prefix}-vet-demo")

    return {
        "demo": demo_clicked,
        "analyze": analyze_clicked,
        "vetting": vetting_clicked,
        "about": about,
        "analyze_demo": analyze_demo,
        "vet_demo": vet_demo,
    }
