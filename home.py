import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fama French Five Factor Model",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide default sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Logo URL
logo_path = https://raw.githubusercontent.com/ewiggom-pixel/GU-Hedge-Fund/37053e6bbddf5f44f6fd8443dfd12f8516f2c73a/hoyalytics_logo-removebg-preview.png

# --- CSS ---
st.markdown("""
<style>
.stApp {
   background: linear-gradient(180deg, #000000 0%, #072f5f 100%);
}

/* Navigation links */
.nav-link {
    color: white !important;
    text-decoration: none;
    font-weight: 600;
    font-size: 1.2rem;
    margin: 0 1.5rem;
    transition: color 0.2s ease;
}
.nav-link:hover {
    color: black !important;
    text-decoration: none;
}

/* Hero Section */
.hero h1 {
    font-size: 3.75rem;
    text-align: center;
    margin: 2rem 0 1rem;
    color: #cbeaff;  /* Updated title color */
}
.hero p {
    font-size: 1.3rem;
    color: #ccc;
    text-align: center;
    margin-bottom: 2.5rem;
}

/* CTA Button */
.cta-button {
    background: linear-gradient(90deg, #58cced,#3895d3);
    padding: 0.75rem 4rem;
    border-radius: 8px;  
    color: white !important;  
    font-weight: 600;
    text-decoration: none;
    display: inline-block;
    font-size: 1rem;
    border: none;
}
.cta-button:hover {
    opacity: 0.85;
}

/* Logo container */
#logo-container {
    margin-top: -50px;
}
#logo-container img {
    max-width: 150px;
    height: auto;
}

/* Dashboard Button as HTML */
.dashboard-btn {
    display: flex;
    justify-content: center;
    margin-top: 2rem;
}
.dashboard-btn a {
    background: linear-gradient(90deg, #3895d3, #58cced);
    border-radius: 999px;
    color: white !important;
    padding: 0.75rem 2rem;
    font-size: 1.125rem;
    font-weight: 600;
    text-decoration: none;
    display: inline-block;
    transition: opacity 0.2s ease, transform 0.15s ease-in-out;
}
.dashboard-btn a:hover {
    opacity: 0.85;
    transform: scale(1.12);
}
</style>
""", unsafe_allow_html=True)

#NAV BAR 
col1, col2, col3 = st.columns([2,6,2])

# Logo
with col1:
    st.markdown(f"""
        <div id="logo-container">
            <img src="{logo_path}">
        </div>
    """, unsafe_allow_html=True)

# NAV links
with col2:
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 2.5rem; padding-left: 2rem;">
        <a class="nav-link" href="https://hoyalytics-website.herokuapp.com/" target="_blank">Hoyalytics</a>
        <a class="nav-link" href="https://www.instagram.com/hoyalytics/" target="_blank">Instagram</a>
        <a class="nav-link" href="https://github.com/ewiggom-pixel/GU-Hedge-Fund" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# CTA button
with col3:
    st.markdown(
        '<div style="text-align: right;"><a class="cta-button" href="https://hoyalytics-website.herokuapp.com/join" target="_blank">Join Us</a></div>',
        unsafe_allow_html=True
    )

# HERO SECTION
st.markdown("""
<div class="hero">
  <h1>Fama French Five Factor Model</h1>
  <p>Track your portfolio’s alpha and factor exposures in real time; upload your portfolio and watch a full history of performance, risk, and factor sensitivities unfold over time.</p>
</div>
""", unsafe_allow_html=True)

# DASHBOARD BUTTON
st.markdown("""
<div class="dashboard-btn">
    <a href="/CSV" target="_self">Try It Now</a>
</div>
""", unsafe_allow_html=True)
