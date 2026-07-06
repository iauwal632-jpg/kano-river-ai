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
    .stApp {
        background: #f5f9fc;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: #1e2a3a;
    }
    .glass {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 20, 40, 0.08);
        border: 1px solid #e8edf2;
    }
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
        margin-top: -0.2rem;
    }
    .input-label {
        color: #2d3748;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        display: block;
        margin-bottom: 4px;
    }
    .input-value {
        color: #2b6cb0;
        font-weight: 600;
        font-size: 1.2rem;
        display: block;
    }
    .stNumberInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #d2dce6 !important;
        border-radius: 10px !important;
        color: #1a202c !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #4a90d9 !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 217, 0.2) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.7rem 2.5rem !important;
        width: 100%;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(49, 130, 206, 0.4) !important;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a202c;
    }
    .metric-label {
        color: #4a6a8a;
        font-size: 0.7rem;
        text-transform: uppercase;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

# ==================== STATE MANAGEMENT ====================
if 'prediction' not in st.session_state:
    st.session_state.prediction = 85.0

if predict_clicked:
    input_data = pd.DataFrame([[rain, tmax, tmin, humidity, rain_3day]], columns=['PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'RH2M', '3Day_Rain_Sum'])
    st.session_state.prediction = max(0, model.predict(input_data)[0])

pred = st.session_state.prediction

# Determine alert parameters
if pred >= 300:
    color_theme = "rgba(200, 30, 30, 0.85)"
    wave_speed, wave_amp, p_speed = 0.15, 14, 8
    status_text, alert_type, risk_level = "🚨 FLOOD DISASTER", "error", "Critical"
elif pred >= 150:
    color_theme = "rgba(230, 130, 0, 0.85)"
    wave_speed, wave_amp, p_speed = 0.07, 8, 4
    status_text, alert_type, risk_level = "⚠️ HIGH WATER ALERT", "warning", "High"
else:
    color_theme = "rgba(0, 140, 220, 0.85)"
    wave_speed, wave_amp, p_speed = 0.02, 4, 1.5
    status_text, alert_type, risk_level = "✅ NORMAL FLOW", "success", "Safe"

water_height = max(15, min(90, (pred / 500) * 100))

# Show Alert
if alert_type == "error":
    st.error(f"🚨 **FLOOD DISASTER WARNING** ({pred:.1f} m³/s) – Critical water levels detected. Evacuation recommended.")
elif alert_type == "warning":
    st.warning(f"⚠️ **HIGH WATER ALERT** ({pred:.1f} m³/s) – Elevated streamflow. Monitor low-lying areas.")
else:
    st.success(f"✅ **NORMAL FLOW** ({pred:.1f} m³/s) – River operating within safe capacity.")

# Metrics Row
st.markdown('<div style="margin:1.5rem 0 0.8rem 0;"></div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{pred:.1f} <span style="font-size:0.9rem;color:#718096">m³/s</span></div><div class="metric-label">Discharge</div></div>', unsafe_allow_html=True)
with m2:
    level_text = "🟢 SAFE" if pred < 150 else ("🟠 HIGH" if pred < 300 else "🔴 CRITICAL")
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.6rem;">{level_text}</div><div class="metric-label">Risk Level</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{min(100, (pred / 500) * 100):.0f}%</div><div class="metric-label">Capacity</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.1rem;">{datetime.now().strftime("%H:%M:%S")}</div><div class="metric-label">Updated</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== RIVER ANIMATION (Canvas) ====================
st.markdown('<div style="color:#4a6a8a; font-weight:500; letter-spacing:0.5px; margin-bottom:0.8rem;">🌊 HYDRAULIC FLOW PROFILE · REAL-TIME</div>', unsafe_allow_html=True)

canvas_html = f"""
<!DOCTYPE html>
<html><body style="margin:0; background:#f5f9fc;">
<div style="position:relative; width:100%; border-radius:16px; overflow:hidden; border:1px solid #e2e8f0;">
    <canvas id="riverCanvas" width="1000" height="350" style="display:block; width:100%; background:#f0f5fa;"></canvas>
    <div style="position:absolute; top:12px; left:16px; font-family:sans-serif; font-size:0.7rem; font-weight:600; background:rgba(255,255,255,0.8); padding:3px 14px; border-radius:20px;">🌊 Kano River Cross-section</div>
</div>
<script>
    const canvas = document.getElementById('riverCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const targetHeight = {water_height / 100};
    let offset = 0, currentWaterLevel = 0.15;
    
    function draw() {{
        offset += {wave_speed};
        currentWaterLevel += (targetHeight - currentWaterLevel) * 0.05;
        ctx.clearRect(0, 0, W, H);
        
        // Sky
        let grad = ctx.createLinearGradient(0,0,0,H);
        grad.addColorStop(0, '#c9dbe9'); grad.addColorStop(1, '#f0f5fa');
        ctx.fillStyle = grad; ctx.fillRect(0,0,W,H);
        
        // Water
        let waterY = H * (1 - currentWaterLevel * 0.85);
        ctx.beginPath(); ctx.moveTo(0, H);
        for(let x=0; x<=W; x+=5) {{
            let y = waterY + Math.sin(x*0.015 + offset)*{wave_amp} + Math.sin(x*0.03 + offset*1.5)*({wave_amp}*0.5);
            ctx.lineTo(x, Math.min(y, H*0.9));
        }}
        ctx.lineTo(W, H); ctx.closePath();
        ctx.fillStyle = '{color_theme}'; ctx.fill();
        
        requestAnimationFrame(draw);
    }}
    draw();
</script>
</body></html>
"""
components.html(canvas_html, height=370)

# ==================== BULLETPROOF PRINT COMPONENT ====================
st.markdown("---")
st.markdown('<div style="color:#4a6a8a; font-weight:500; letter-spacing:0.5px; margin-bottom:0.8rem;">📄 PREDICTION REPORT</div>', unsafe_allow_html=True)

# We isolate the entire print logic inside an HTML component so it doesn't touch Streamlit's DOM
print_component = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; text-align: center; margin: 0; padding: 10px; }}
        .btn {{ background: #3182ce; color: white; border: none; padding: 10px 30px; border-radius: 50px; font-weight: bold; cursor: pointer; transition: 0.2s; }}
        .btn:hover {{ background: #2b6cb0; transform: translateY(-2px); }}
        #printArea {{ display: none; }} /* Hidden from UI, used only for printing */
    </style>
</head>
<body>
    
    <button class="btn" onclick="printReport()">🖨️ Print Official Report</button>
    
    <div id="printArea">
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 30px; color: #333;">
            <h2 style="color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 10px;">🌊 Kano River Flood Early Warning System</h2>
            <p><strong>Official Prediction Report</strong> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                <tr>
                    <td style="vertical-align: top; width: 50%;">
                        <h3 style="color: #2b6cb0;">Meteorological Inputs</h3>
                        <p><strong>Precipitation:</strong> {rain:.1f} mm</p>
                        <p><strong>3-Day Rain Sum:</strong> {rain_3day:.1f} mm</p>
                        <p><strong>Relative Humidity:</strong> {humidity:.0f}%</p>
                        <p><strong>Max Temperature:</strong> {tmax:.1f} °C</p>
                        <p><strong>Min Temperature:</strong> {tmin:.1f} °C</p>
                    </td>
                    <td style="vertical-align: top; width: 50%; background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <h3 style="color: #2b6cb0; margin-top:0;">Prediction Results</h3>
                        <h1 style="color: #1a202c; margin: 10px 0;">{pred:.1f} m³/s</h1>
                        <p style="font-size: 1.2rem; color: {'#c53030' if risk_level == 'Critical' else '#b7791f' if risk_level == 'High' else '#2f855a'}; font-weight: bold;">{status_text}</p>
                        <p><strong>Risk Level:</strong> {risk_level}</p>
                        <p><strong>Channel Capacity:</strong> {min(100, (pred / 500) * 100):.0f}% Full</p>
                    </td>
                </tr>
            </table>
            
            <div style="margin-top: 40px; text-align: center; color: #718096; font-size: 0.9rem; border-top: 1px solid #eee; padding-top: 20px;">
                Auto-generated by Rantan Technologies AI Engine · Aliko Dangote University of Science and Technology
            </div>
        </div>
    </div>

    <script>
        function printReport() {{
            // 1. Grab the HTML of the report
            const printContent = document.getElementById('printArea').innerHTML;
            
            // 2. Open a temporary, hidden popup window
            const printWindow = window.open('', '', 'height=800,width=800');
            
            // 3. Write the report HTML into the popup
            printWindow.document.write('<html><head><title>Print Report - Kano River EWS</title></head><body>');
            printWindow.document.write(printContent);
            printWindow.document.write('</body></html>');
            
            // 4. Trigger print and close the popup silently
            printWindow.document.close();
            printWindow.focus();
            
            // Short delay ensures the browser renders the CSS before printing
            setTimeout(function() {{
                printWindow.print();
                printWindow.close();
            }}, 250);
        }}
    </script>
</body>
</html>
"""
components.html(print_component, height=100)

# ==================== FOOTER ====================
st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.7rem; padding:2rem 0; margin-top:2rem; border-top:1px solid #e2e8f0;">
    Kano River Flood Early Warning System · Powered by Machine Learning · © 2026
</div>
""", unsafe_allow_html=True)
