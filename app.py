import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit.components.v1 as components
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Kano River EWS",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== LIGHT PROFESSIONAL THEME CSS ====================
st.markdown("""
<style>
    /* Light theme */
    .stApp {
        background: #f5f9fc;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: #1e2a3a;
    }

    /* Cards */
    .glass {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 20, 40, 0.08);
        border: 1px solid #e8edf2;
        transition: box-shadow 0.2s;
    }
    .glass:hover {
        box-shadow: 0 8px 32px rgba(0, 20, 40, 0.12);
    }

    .input-glass {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        transition: border-color 0.3s;
    }
    .input-glass:hover {
        border-color: #4a90d9;
    }

    /* Title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .hero-sub {
        color: #4a6a8a;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        margin-top: -0.2rem;
    }

    /* Input labels */
    .input-label {
        color: #2d3748;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        display: block;
        margin-bottom: 4px;
    }
    .input-value {
        color: #2b6cb0;
        font-weight: 600;
        font-size: 1.2rem;
        display: block;
        margin-top: 2px;
    }

    /* Custom number inputs */
    .stNumberInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #d2dce6 !important;
        border-radius: 10px !important;
        color: #1a202c !important;
        padding: 0.5rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #4a90d9 !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 217, 0.2) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.7rem 2.5rem !important;
        border: none !important;
        border-radius: 50px !important;
        box-shadow: 0 4px 16px rgba(49, 130, 206, 0.3) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.3px;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(49, 130, 206, 0.4) !important;
        background: linear-gradient(135deg, #4299e1, #3182ce) !important;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: all 0.2s;
    }
    .metric-card:hover {
        border-color: #bdd3eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a202c;
        line-height: 1.2;
    }
    .metric-label {
        color: #4a6a8a;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .metric-unit {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 400;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 1.2rem;
        border-radius: 60px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.3px;
    }
    .status-safe { background: #e6f7e6; color: #2f855a; border: 1px solid #b2e0b2; }
    .status-warning { background: #fef3e2; color: #b7791f; border: 1px solid #fbd38d; }
    .status-danger { background: #fee2e2; color: #c53030; border: 1px solid #feb2b2; animation: pulse-danger 1.5s ease-in-out infinite; }
    @keyframes pulse-danger {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Footer */
    .footer {
        color: #a0aec0;
        font-size: 0.7rem;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
        letter-spacing: 0.3px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Report container - for print */
    .report-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .report-container h3 {
        color: #1a365d;
        margin-top: 0;
    }
    .report-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0f4f8;
    }
    .report-label {
        font-weight: 500;
        color: #4a6a8a;
    }
    .report-value {
        font-weight: 600;
        color: #1a202c;
    }

    @media print {
        .no-print { display: none !important; }
        .report-container { border: none; box-shadow: none; }
        .stApp { background: white; }
        .report-container { page-break-inside: avoid; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv("final_kano_dataset.csv").dropna()
        features = ['PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'RH2M', '3Day_Rain_Sum']
        X = df[features]
        y = df['Streamflow_m3s']
        model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42)
        model.fit(X, y)
        return model
    except:
        # Demo fallback
        st.warning("⚠️ Dataset not found. Using synthetic demo model for testing.")
        import numpy as np
        np.random.seed(42)
        X = np.random.rand(100, 5) * 100
        y = X[:, 0] * 2.5 + X[:, 4] * 1.8 + np.random.randn(100) * 15
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

model = load_and_train_model()

# ==================== HEADER ====================
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">🌊 Kano River EWS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Flood Early Warning · Streamflow Prediction</div>', unsafe_allow_html=True)
with col_status:
    st.markdown("""
    <div style="text-align:right; padding-top:0.5rem;">
        <span style="color:#a0aec0; font-size:0.6rem; letter-spacing:0.5px;">⚡ SYSTEM ACTIVE</span>
        <div style="display:flex; justify-content:flex-end; gap:6px; margin-top:4px;">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#3182ce; box-shadow:0 0 12px rgba(49,130,206,0.4);"></span>
            <span style="color:#a0aec0; font-size:0.6rem;">live</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== INPUT SECTION ====================
st.markdown('<div style="color:#4a6a8a; font-weight:500; letter-spacing:0.5px; margin-bottom:1rem;">📡 METEOROLOGICAL INPUTS</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    cols = st.columns(5)
    with cols[0]:
        st.markdown('<span class="input-label">☔ Precipitation</span>', unsafe_allow_html=True)
        rain = st.number_input("", min_value=0.0, value=12.5, step=0.5, key="rain", label_visibility="collapsed")
        st.markdown(f'<span class="input-value">{rain:.1f} mm</span>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<span class="input-label">🌧️ 3-Day Rain Sum</span>', unsafe_allow_html=True)
        rain_3day = st.number_input("", min_value=0.0, value=28.0, step=1.0, key="rain3", label_visibility="collapsed")
        st.markdown(f'<span class="input-value">{rain_3day:.1f} mm</span>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<span class="input-label">💧 Humidity</span>', unsafe_allow_html=True)
        humidity = st.number_input("", min_value=0.0, max_value=100.0, value=62.0, step=1.0, key="hum", label_visibility="collapsed")
        st.markdown(f'<span class="input-value">{humidity:.0f} %</span>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<span class="input-label">🌡️ Max Temp</span>', unsafe_allow_html=True)
        tmax = st.number_input("", value=34.5, step=0.5, key="tmax", label_visibility="collapsed")
        st.markdown(f'<span class="input-value">{tmax:.1f} °C</span>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown('<span class="input-label">🌡️ Min Temp</span>', unsafe_allow_html=True)
        tmin = st.number_input("", value=21.5, step=0.5, key="tmin", label_visibility="collapsed")
        st.markdown(f'<span class="input-value">{tmin:.1f} °C</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== PREDICT BUTTON ====================
center_btn = st.columns([2, 2, 2])
with center_btn[1]:
    predict_clicked = st.button("🚀 PREDICT STREAMFLOW", type="primary", use_container_width=True)

# ==================== INITIALIZE SESSION STATE ====================
if 'prediction' not in st.session_state:
    st.session_state.prediction = 85.0
    st.session_state.water_height = 35
    st.session_state.color_theme = "rgba(0, 119, 190, 0.85)"
    st.session_state.wave_speed = 0.02
    st.session_state.wave_amplitude = 4
    st.session_state.particle_speed = 1.5
    st.session_state.status_text = "NORMAL FLOW"
    st.session_state.status_class = "status-safe"
    st.session_state.alert_type = "success"
    st.session_state.risk_level = "Safe"

if predict_clicked:
    input_data = pd.DataFrame([[rain, tmax, tmin, humidity, rain_3day]],
                              columns=['PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'RH2M', '3Day_Rain_Sum'])
    pred = model.predict(input_data)[0]
    st.session_state.prediction = max(0, pred)

    if pred >= 300:
        st.session_state.color_theme = "rgba(200, 30, 30, 0.85)"   # red
        st.session_state.wave_speed = 0.15
        st.session_state.wave_amplitude = 14
        st.session_state.particle_speed = 8
        st.session_state.status_text = "🚨 FLOOD DISASTER"
        st.session_state.status_class = "status-danger"
        st.session_state.alert_type = "error"
        st.session_state.risk_level = "Critical"
    elif pred >= 150:
        st.session_state.color_theme = "rgba(230, 130, 0, 0.85)"    # orange
        st.session_state.wave_speed = 0.07
        st.session_state.wave_amplitude = 8
        st.session_state.particle_speed = 4
        st.session_state.status_text = "⚠️ HIGH WATER ALERT"
        st.session_state.status_class = "status-warning"
        st.session_state.alert_type = "warning"
        st.session_state.risk_level = "High"
    else:
        st.session_state.color_theme = "rgba(0, 140, 220, 0.85)"    # blue
        st.session_state.wave_speed = 0.02
        st.session_state.wave_amplitude = 4
        st.session_state.particle_speed = 1.5
        st.session_state.status_text = "✅ NORMAL FLOW"
        st.session_state.status_class = "status-safe"
        st.session_state.alert_type = "success"
        st.session_state.risk_level = "Safe"

    st.session_state.water_height = max(15, min(90, (pred / 500) * 100))

# ==================== RESULTS DISPLAY ====================
pred = st.session_state.prediction
water_height = st.session_state.water_height
color_theme = st.session_state.color_theme
wave_speed = st.session_state.wave_speed
wave_amplitude = st.session_state.wave_amplitude
particle_speed = st.session_state.particle_speed
status_text = st.session_state.status_text
status_class = st.session_state.status_class
alert_type = st.session_state.alert_type
risk_level = st.session_state.risk_level

# Show alert
if alert_type == "error":
    st.error(f"🚨 **FLOOD DISASTER WARNING** ({pred:.1f} m³/s) – Critical water levels detected. High risk of river breaking banks.")
elif alert_type == "warning":
    st.warning(f"⚠️ **HIGH WATER ALERT** ({pred:.1f} m³/s) – Elevated streamflow. Monitor low-lying areas.")
else:
    st.success(f"✅ **NORMAL FLOW** ({pred:.1f} m³/s) – River operating within safe capacity.")

# Metrics Row
st.markdown('<div style="color:#4a6a8a; font-weight:500; letter-spacing:0.5px; margin:1.5rem 0 0.8rem 0;">📊 PREDICTION METRICS</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{pred:.1f} <span class="metric-unit">m³/s</span></div>
        <div class="metric-label">Discharge</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    level_text = "🟢 SAFE" if pred < 150 else ("🟠 HIGH" if pred < 300 else "🔴 CRITICAL")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.6rem;">{level_text}</div>
        <div class="metric-label">Risk Level</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    pct = min(100, (pred / 500) * 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{pct:.0f}%</div>
        <div class="metric-label">Capacity</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.1rem;">{datetime.now().strftime('%H:%M:%S')}</div>
        <div class="metric-label">Updated</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== RIVER ANIMATION (Canvas) ====================
st.markdown('<div style="color:#4a6a8a; font-weight:500; letter-spacing:0.5px; margin-bottom:0.8rem;">🌊 HYDRAULIC FLOW PROFILE · REAL-TIME</div>', unsafe_allow_html=True)

canvas_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    * {{ margin: 0; padding: 0; }}
    .river-container {{
        position: relative;
        width: 100%;
        max-width: 1000px;
        margin: 0 auto;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        background: #f0f5fa;
    }}
    canvas {{
        display: block;
        width: 100%;
        height: auto;
        background: #f0f5fa;
    }}
    .canvas-label {{
        position: absolute;
        top: 12px;
        left: 16px;
        font-family: 'Segoe UI', sans-serif;
        color: #2d3748;
        font-size: 0.7rem;
        letter-spacing: 0.5px;
        font-weight: 600;
        background: rgba(255,255,255,0.8);
        padding: 3px 14px;
        border-radius: 20px;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.5);
    }}
    .canvas-status {{
        position: absolute;
        top: 12px;
        right: 16px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 3px 14px;
        border-radius: 20px;
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.5);
        color: #1a202c;
    }}
    .canvas-flow {{
        position: absolute;
        bottom: 12px;
        left: 16px;
        font-family: 'Segoe UI', sans-serif;
        color: #4a6a8a;
        font-size: 0.7rem;
        background: rgba(255,255,255,0.7);
        padding: 3px 14px;
        border-radius: 20px;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.3);
    }}
    .canvas-flow strong {{
        color: #1a202c;
    }}
</style>
</head>
<body>
<div class="river-container">
    <canvas id="riverCanvas" width="1000" height="400"></canvas>
    <div class="canvas-label">🌊 Kano River · Cross-section</div>
    <div class="canvas-status">{status_text}</div>
    <div class="canvas-flow">Discharge: <strong>{pred:.1f} m³/s</strong> · Level: <strong>{water_height:.0f}%</strong></div>
</div>

<script>
    const canvas = document.getElementById('riverCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    const targetHeight = {water_height / 100};
    const speed = {wave_speed};
    const amplitude = {wave_amplitude};
    const pSpeed = {particle_speed};
    const waterColor = '{color_theme}';

    let offset = 0;
    let currentWaterLevel = 0.15;
    let particles = [];
    const numParticles = 50;
    const bankLeft = W * 0.12;
    const bankRight = W * 0.88;

    for (let i = 0; i < numParticles; i++) {{
        particles.push({{
            x: bankLeft + Math.random() * (bankRight - bankLeft),
            y: H * 0.3 + Math.random() * H * 0.5,
            size: 1.5 + Math.random() * 2.5,
            speed: pSpeed * (0.7 + Math.random() * 0.6),
            phase: Math.random() * Math.PI * 2
        }});
    }}

    function drawSky() {{
        const grad = ctx.createLinearGradient(0, 0, 0, H * 0.6);
        grad.addColorStop(0, '#c9dbe9');
        grad.addColorStop(0.4, '#dce6ef');
        grad.addColorStop(0.8, '#eaf0f5');
        grad.addColorStop(1, '#f0f5fa');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H * 0.7);
    }}

    function drawBanks() {{
        // Left bank
        ctx.beginPath();
        ctx.moveTo(0, H * 0.5);
        ctx.quadraticCurveTo(W * 0.05, H * 0.42, bankLeft, H * 0.48);
        ctx.quadraticCurveTo(bankLeft + 10, H * 0.52, bankLeft, H * 0.70);
        ctx.lineTo(0, H * 0.70);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, H*0.4, 0, H*0.7);
        grad.addColorStop(0, '#9bb87f');
        grad.addColorStop(0.5, '#7da36a');
        grad.addColorStop(1, '#5a7a4a');
        ctx.fillStyle = grad;
        ctx.fill();

        // Right bank
        ctx.beginPath();
        ctx.moveTo(W, H * 0.48);
        ctx.quadraticCurveTo(W * 0.95, H * 0.40, bankRight, H * 0.46);
        ctx.quadraticCurveTo(bankRight - 10, H * 0.50, bankRight, H * 0.68);
        ctx.lineTo(W, H * 0.68);
        ctx.closePath();
        const grad2 = ctx.createLinearGradient(W, H*0.4, W, H*0.7);
        grad2.addColorStop(0, '#9bb87f');
        grad2.addColorStop(0.5, '#7da36a');
        grad2.addColorStop(1, '#5a7a4a');
        ctx.fillStyle = grad2;
        ctx.fill();

        // Grass details
        for (let i = 0; i < 20; i++) {{
            const x = bankLeft - 5 - Math.random() * 15;
            const y = H * 0.48 + Math.random() * 0.08 * H;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x - 3, y - 10 - Math.random() * 10);
            ctx.lineTo(x + 3, y - 8 - Math.random() * 10);
            ctx.closePath();
            ctx.fillStyle = `rgba(100, 160, 80, ${{0.3 + Math.random()*0.4}})`;
            ctx.fill();
        }}
        for (let i = 0; i < 20; i++) {{
            const x = bankRight + 5 + Math.random() * 15;
            const y = H * 0.46 + Math.random() * 0.08 * H;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x - 3, y - 10 - Math.random() * 10);
            ctx.lineTo(x + 3, y - 8 - Math.random() * 10);
            ctx.closePath();
            ctx.fillStyle = `rgba(100, 160, 80, ${{0.3 + Math.random()*0.4}})`;
            ctx.fill();
        }}
    }}

    function drawWater(level) {{
        const waterY = H * (1 - level * 0.85);
        const waves = [
            {{ amp: amplitude * 1.2, freq: 0.018, speed: speed * 1.8, phase: 0 }},
            {{ amp: amplitude * 0.8, freq: 0.035, speed: speed * 1.2, phase: 1.2 }},
            {{ amp: amplitude * 0.5, freq: 0.06, speed: speed * 0.7, phase: 2.8 }},
            {{ amp: amplitude * 0.3, freq: 0.09, speed: speed * 0.4, phase: 4.5 }}
        ];

        ctx.beginPath();
        ctx.moveTo(bankLeft, H);
        for (let x = bankLeft; x <= bankRight; x += 1) {{
            let y = waterY;
            for (const w of waves) {{
                y += w.amp * Math.sin(x * w.freq + offset * w.speed + w.phase);
            }}
            const bankY = H * (0.50 + 0.06 * Math.sin(x * 0.01 + 0.5));
            const maxY = Math.min(H * 0.92, bankY + 20);
            y = Math.min(y, maxY);
            ctx.lineTo(x, y);
        }}
        ctx.lineTo(bankRight, H);
        ctx.closePath();
        ctx.fillStyle = waterColor;
        ctx.fill();

        // Reflection
        ctx.beginPath();
        for (let x = bankLeft; x <= bankRight; x += 1) {{
            let y = waterY + 10;
            for (const w of waves) {{
                y += w.amp * 0.6 * Math.sin(x * w.freq + offset * w.speed * 0.8 + w.phase + 0.5);
            }}
            const bankY = H * (0.50 + 0.06 * Math.sin(x * 0.01 + 0.5));
            const maxY = Math.min(H * 0.92, bankY + 20);
            y = Math.min(y, maxY);
            ctx.lineTo(x, y);
        }}
        ctx.lineTo(bankRight, H);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.fill();

        // Sparkles
        for (let i = 0; i < 30; i++) {{
            const x = bankLeft + 10 + ((i * 73 + 17) % (bankRight - bankLeft - 20));
            let y = waterY;
            for (const w of waves) {{
                y += w.amp * Math.sin(x * w.freq + offset * w.speed + w.phase + i * 0.5);
            }}
            const bankY = H * (0.50 + 0.06 * Math.sin(x * 0.01 + 0.5));
            const maxY = Math.min(H * 0.92, bankY + 20);
            y = Math.min(y, maxY);
            if (i % 3 === 0) {{
                ctx.fillStyle = `rgba(255,255,255,${{0.05 + 0.06 * Math.sin(offset + i)}})`;
                ctx.fillRect(x, y - 1, 4, 2);
            }}
        }}
    }}

    function drawParticles(level) {{
        const waterY = H * (1 - level * 0.85);
        const waves = [
            {{ amp: amplitude * 1.2, freq: 0.018, speed: speed * 1.8, phase: 0 }},
            {{ amp: amplitude * 0.8, freq: 0.035, speed: speed * 1.2, phase: 1.2 }},
            {{ amp: amplitude * 0.5, freq: 0.06, speed: speed * 0.7, phase: 2.8 }},
            {{ amp: amplitude * 0.3, freq: 0.09, speed: speed * 0.4, phase: 4.5 }}
        ];

        particles.forEach(p => {{
            p.x += p.speed * 0.5;
            if (p.x > bankRight) p.x = bankLeft;
            let ySurf = waterY;
            for (const w of waves) {{
                ySurf += w.amp * Math.sin(p.x * w.freq + offset * w.speed + w.phase + p.phase);
            }}
            const bankY = H * (0.50 + 0.06 * Math.sin(p.x * 0.01 + 0.5));
            const maxY = Math.min(H * 0.92, bankY + 20);
            ySurf = Math.min(ySurf, maxY);
            const targetY = ySurf + 4 + Math.sin(offset * 0.5 + p.phase) * 3;
            p.y += (targetY - p.y) * 0.05;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(180, 200, 150, 0.5)`;
            ctx.fill();
        }});
    }}

    function drawRipples(level) {{
        const waterY = H * (1 - level * 0.85);
        for (let i = 0; i < 5; i++) {{
            const cx = bankLeft + 30 + ((i * 97 + 23) % (bankRight - bankLeft - 60));
            let cy = waterY + 15 + Math.sin(offset * 0.2 + i * 1.2) * 10;
            const bankY = H * (0.50 + 0.06 * Math.sin(cx * 0.01 + 0.5));
            const maxY = Math.min(H * 0.92, bankY + 20);
            cy = Math.min(cy, maxY - 5);
            const r = 8 + Math.sin(offset * 0.4 + i) * 4;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(255,255,255,0.15)`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }}
    }}

    function drawWaterLevelMarkers(level) {{
        const y = H * (1 - level * 0.85);
        ctx.beginPath();
        ctx.moveTo(bankRight + 10, y);
        ctx.lineTo(bankRight + 26, y);
        ctx.strokeStyle = '#3182ce';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(bankRight + 18, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(49, 130, 206, 0.2)';
        ctx.fill();
    }}

    function drawBed() {{
        ctx.fillStyle = '#8b6b4d';
        ctx.fillRect(0, H - 10, W, 10);
        for (let i = 0; i < 100; i++) {{
            const x = Math.random() * W;
            const y = H - 8 - Math.random() * 4;
            ctx.fillStyle = `rgba(120, 80, 50, ${{0.2 + Math.random()*0.3}})`;
            ctx.fillRect(x, y, 2 + Math.random()*3, 1 + Math.random()*2);
        }}
    }}

    function animate() {{
        offset += speed * 0.8;
        currentWaterLevel += (targetHeight - currentWaterLevel) * 0.04;
        if (Math.abs(currentWaterLevel - targetHeight) < 0.001) currentWaterLevel = targetHeight;

        ctx.clearRect(0, 0, W, H);
        drawSky();
        drawBanks();
        drawWater(currentWaterLevel);
        drawRipples(currentWaterLevel);
        drawParticles(currentWaterLevel);
        drawWaterLevelMarkers(currentWaterLevel);
        drawBed();

        requestAnimationFrame(animate);
    }}

    animate();
</script>
</body>
</html>
"""

components.html(canvas_html, height=430)

# ==================== REPORT GENERATION ====================
st.markdown("---")
st.markdown('<div style="color:#4a6a8a; font-weight:500; letter-spacing:0.5px; margin-bottom:0.8rem;">📄 PREDICTION REPORT</div>', unsafe_allow_html=True)

# Build the report content
report_html = f"""
<div class="report-container" id="reportContainer">
    <h3>🌊 Kano River Flood Early Warning System – Prediction Report</h3>
    <p style="color:#4a6a8a; font-size:0.9rem; margin-bottom:1.2rem;">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
        <div>
            <h4 style="color:#1a365d; margin:0 0 0.5rem 0;">Input Parameters</h4>
            <div class="report-row"><span class="report-label">Precipitation</span><span class="report-value">{rain:.1f} mm</span></div>
            <div class="report-row"><span class="report-label">3-Day Rain Sum</span><span class="report-value">{rain_3day:.1f} mm</span></div>
            <div class="report-row"><span class="report-label">Humidity</span><span class="report-value">{humidity:.0f} %</span></div>
            <div class="report-row"><span class="report-label">Max Temperature</span><span class="report-value">{tmax:.1f} °C</span></div>
            <div class="report-row"><span class="report-label">Min Temperature</span><span class="report-value">{tmin:.1f} °C</span></div>
        </div>
        <div>
            <h4 style="color:#1a365d; margin:0 0 0.5rem 0;">Prediction Results</h4>
            <div class="report-row"><span class="report-label">Streamflow</span><span class="report-value">{pred:.1f} m³/s</span></div>
            <div class="report-row"><span class="report-label">Risk Level</span><span class="report-value" style="color:{'#2f855a' if risk_level=='Safe' else '#b7791f' if risk_level=='High' else '#c53030'};">{risk_level}</span></div>
            <div class="report-row"><span class="report-label">Capacity Utilization</span><span class="report-value">{min(100, (pred/500)*100):.0f}%</span></div>
            <div class="report-row"><span class="report-label">Status</span><span class="report-value">{status_text}</span></div>
        </div>
    </div>
    <div style="margin-top:1rem; padding-top:0.8rem; border-top:1px solid #e2e8f0; color:#718096; font-size:0.8rem; text-align:center;">
        This report is auto-generated by the Kano River Early Warning System.
    </div>
</div>
"""

st.markdown(report_html, unsafe_allow_html=True)

# Print button - uses JavaScript to print the report container
st.markdown("""
<div class="no-print" style="display:flex; justify-content:center; margin-top:0.8rem;">
    <button onclick="printReport()" style="background:#3182ce; color:white; border:none; border-radius:50px; padding:0.6rem 2rem; font-weight:600; font-size:0.9rem; cursor:pointer; box-shadow:0 4px 16px rgba(49,130,206,0.3); transition:all 0.2s;">
        🖨️ Print Report
    </button>
</div>
<script>
function printReport() {
    var printContents = document.getElementById('reportContainer').innerHTML;
    var originalContents = document.body.innerHTML;
    document.body.innerHTML = '<div style="max-width:800px; margin:2rem auto; padding:1rem;">' + printContents + '</div>';
    window.print();
    document.body.innerHTML = originalContents;
    location.reload(); // needed to restore Streamlit functionality
}
</script>
""", unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    Kano River Flood Early Warning System · Powered by Machine Learning · © 2026
</div>
""", unsafe_allow_html=True)