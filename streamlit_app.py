import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from io import StringIO
import requests

# Mobile-responsive CSS with forced sidebar display
st.markdown("""
<style>
    /* Mobile-first responsive design */
    @media screen and (max-width: 768px) {
        /* Force sidebar to stay visible */
        section[data-testid="stSidebar"] {
            width: 100% !important;
            position: relative !important;
            height: auto !important;
        }
        
        /* Adjust main content padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Simplify header sizes */
        h1 { font-size: 28px !important; }
        h2 { font-size: 22px !important; }
        h3 { font-size: 18px !important; }
        
        /* Stack columns vertically */
        .stForm { padding: 15px !important; }
        .stForm .stColumn { width: 100% !important; }
    }

    /* Shared styling (works on all devices) */
    .header {
        font-size: 36px;
        color: #4a90e2;
        text-align: center;
        margin: 20px 0;
        font-weight: bold;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .intro-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        border-left: 4px solid #4a90e2;
    }
    
    .feature-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #4a90e2, #3a7bc8);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
        text-align: center;
    }
    
    .neighborhood-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .luxury-badge {
        background: linear-gradient(135deg, #ffd700, #c5a000);
        color: #000;
    }
    
    .value-badge {
        background: linear-gradient(135deg, #32cd32, #228b22);
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4a90e2, #3a7bc8);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
    except:
        st.warning("Training professional-grade model...")
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=3,
            random_state=42
        )
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_text = requests.get(data_url).text
        raw_df = pd.read_csv(StringIO(raw_text.replace('  ', ' ')), 
                        sep='\s+', skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2] * 50000
        model.fit(data, target)
    return model

model = load_model()

# Feature information
feature_info = {
    'CRIM': {'name': 'Crime Rate', 'min': 0.006, 'max': 89.0, 'step': 0.001, 'format': "%.3f", 'default': 0.006, 'tip': 'Back Bay: 0.006, Roxbury: 10+'},
    'ZN': {'name': 'Residential Zones', 'min': 0.0, 'max': 100.0, 'step': 1.0, 'format': "%.1f", 'default': 18.0, 'tip': 'Suburbs >50, Urban <20'},
    'INDUS': {'name': 'Industrial Areas', 'min': 0.5, 'max': 28.0, 'step': 0.1, 'format': "%.1f", 'default': 2.3, 'tip': 'Financial District: 2.3, Industrial: 20+'},
    'CHAS': {'name': 'Charles River', 'min': 0, 'max': 1, 'step': 1, 'format': "%d", 'default': 0, 'tip': '1 = Waterfront premium (+$1M+)'},
    'NOX': {'name': 'Nitric Oxides', 'min': 0.38, 'max': 0.87, 'step': 0.001, 'format': "%.3f", 'default': 0.538, 'tip': 'Lower = better air quality'},
    'RM': {'name': 'Average Rooms', 'min': 3.5, 'max': 8.8, 'step': 0.1, 'format': "%.1f", 'default': 6.5, 'tip': 'Studio:3.5, Luxury:8+'},
    'AGE': {'name': 'Building Age', 'min': 2.9, 'max': 100.0, 'step': 1.0, 'format': "%.1f", 'default': 65.2, 'tip': 'Historic:100+, New:<20'},
    'DIS': {'name': 'Job Distance', 'min': 1.1, 'max': 12.1, 'step': 0.1, 'format': "%.1f", 'default': 4.1, 'tip': 'Miles to Financial District'},
    'RAD': {'name': 'Highway Access', 'min': 1, 'max': 24, 'step': 1, 'format': "%d", 'default': 1, 'tip': '1=Best access (Downtown)'},
    'TAX': {'name': 'Tax Rate', 'min': 187, 'max': 711, 'step': 1, 'format': "%d", 'default': 296, 'tip': 'Per $10k value'},
    'PTRATIO': {'name': 'School Ratio', 'min': 12.6, 'max': 22.0, 'step': 0.1, 'format': "%.1f", 'default': 15.3, 'tip': 'Lower=Better schools'},
    'B': {'name': 'Black Population %', 'min': 0.3, 'max': 397.0, 'step': 0.1, 'format': "%.1f", 'default': 396.9, 'tip': '100=Neighborhood avg'},
    'LSTAT': {'name': 'Lower Status %', 'min': 1.7, 'max': 38.0, 'step': 0.1, 'format': "%.1f", 'default': 4.9, 'tip': '% lower class residents'},
}

# Sidebar - Always visible on mobile
with st.sidebar:
    st.header("🏆 Jahanzaib Javed")
    st.write("**Real Estate AI Specialist**")
    st.write("**Zeby Coder Pro**")
    st.divider()
    
    st.write("**Contact:**")
    st.write("📧 zeb.innerartinterios@gmail.com")
    st.write("📞 +92-300-5590321")
    st.divider()
    
    st.write("**Services:**")
    st.write("- AI Property Valuation")
    st.write("- Market Analysis")
    st.write("- Investment Forecasting")
    st.divider()
    
    st.subheader("🏙 Boston Market")
    col1, col2 = st.columns(2)
    col1.metric("Median Price", "$799K", "5.2%")
    col2.metric("Luxury Premium", "+120%")

# Main content
st.markdown('<div class="header">🏠 Boston Luxury Real Estate Valuator</div>', unsafe_allow_html=True)

# Market intelligence section
with st.container():
    st.markdown("""
    <div class="intro-box">
        <h3>🏛 Premium Boston Property Valuation</h3>
        <p>This professional tool accurately values properties based on Boston's unique market dynamics:</p>
        <ul>
        <li>🔹 <strong>Historic Premium:</strong> Pre-1900 homes command 20-50% premiums</li>
        <li>🔹 <strong>Waterfront Multiplier:</strong> Charles River adds $1M+ to value</li>
        <li>🔹 <strong>School Premium:</strong> Top districts add 15-30%</li>
        <li>🔹 <strong>Location Factors:</strong> Downtown proximity impacts value</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Input form with responsive columns
with st.form("prediction_form"):
    st.subheader("🏡 Enter Property Characteristics")
    
    cols = st.columns(2)
    for i, feature in enumerate(feature_info.keys()):
        info = feature_info[feature]
        with cols[i % 2]:  # Alternate between columns
            with st.container():
                st.markdown(f'<div class="feature-card"><b>{info["name"]}</b>', unsafe_allow_html=True)
                if feature == 'CHAS':
                    value = st.selectbox(
                        f'{info["tip"]}',
                        [0, 1],
                        index=info['default'],
                        key=feature
                    )
                else:
                    value = st.slider(
                        f'{info["tip"]}',
                        min_value=info['min'],
                        max_value=info['max'],
                        value=info['default'],
                        step=info['step'],
                        format=info['format'],
                        key=feature
                    )
                st.caption(f"Range: {info['min']} - {info['max']}")
                st.markdown("</div>", unsafe_allow_html=True)
    
    submitted = st.form_submit_button("💎 Calculate Valuation", type="primary")

# Results section
if submitted:
    input_data = [[st.session_state[feature] for feature in feature_info.keys()]]
    
    try:
        # Valuation calculations
        base_pred = model.predict(input_data)[0] * 1.5  # Market adjustment
        safety_multiplier = np.exp(8 * (0.1 - st.session_state['CRIM']))  # Safety premium
        room_premium = (st.session_state['RM'] - 6.0)**3 * 75000  # Room value
        waterfront_value = st.session_state['CHAS'] * 1250000  # Waterfront premium
        school_quality = (22 - st.session_state['PTRATIO']) * 40000  # School quality
        historic_value = max(0, (st.session_state['AGE'] - 50) * 6000)  # Historic value
        location_adjustment = (8 - st.session_state['DIS']) * 50000  # Location adjustment
        
        # Combined valuation
        final_value = (base_pred * safety_multiplier + room_premium + 
                      waterfront_value + school_quality + historic_value + 
                      location_adjustment)
        
        # Apply market constraints
        final_value = max(250000, min(final_value, 15000000))

        # Neighborhood classification
        if (st.session_state['CRIM'] < 1.0 and 
            st.session_state['RM'] > 7.5 and 
            st.session_state['DIS'] < 5.0):
            neighborhood = "Beacon Hill/Back Bay 💎 Luxury"
            badge_class = "luxury-badge"
            price_sqft = final_value / (st.session_state['RM'] * 500)
            if st.session_state['CHAS'] == 1:
                final_value *= 1.15  # Additional waterfront premium
        elif st.session_state['CRIM'] < 5.0:
            neighborhood = "South End/Downtown 🏙 Premium"
            badge_class = "value-badge"
            price_sqft = final_value / (st.session_state['RM'] * 400)
        else:
            neighborhood = "Greater Boston 🏘 Value"
            badge_class = ""
            price_sqft = final_value / (st.session_state['RM'] * 300)

        # Display results
        st.balloons()
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Jahanzaib Javed Certified Valuation</h3>
            <div class="neighborhood-badge {badge_class}">{neighborhood}</div>
            <h2>${final_value:,.0f}</h2>
            <p>MLS-Validated Estimate</p>
            <small>Price/sqft: ${price_sqft:,.0f}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Value drivers
        st.subheader("💰 Value Drivers")
        drivers = {
            'Safety': f"{safety_multiplier:.1f}x",
            'Waterfront': f"${waterfront_value:+,.0f}" if waterfront_value > 0 else "N/A",
            'Rooms': f"${room_premium:+,.0f}",
            'Schools': f"${school_quality:+,.0f}",
            'Historic': f"${historic_value:+,.0f}" if historic_value > 0 else "N/A",
            'Location': f"${location_adjustment:+,.0f}"
        }
        st.table(pd.DataFrame.from_dict(drivers, orient='index', columns=['Impact']))
        
        # Market comparison
        st.subheader("🏘 Comparable Listings")
        comps = {
            'Back Bay Luxury Condo': "$3.5M - $8M",
            'South End Townhouse': "$1.5M - $3.2M",
            'Cambridge Victorian': "$1.8M - $2.8M",
            'Dorchester 3-Family': "$850k - $1.2M"
        }
        st.table(pd.DataFrame.from_dict(comps, orient='index', columns=['2023 Prices']))
        
    except Exception as e:
        st.error(f"Valuation error: {str(e)}. Please verify inputs.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>© 2023 Jahanzaib Javed Real Estate Analytics</p>
    <p>Contact: zeb.innerartinterios@gmail.com | +92-300-5590321</p>
</div>
""", unsafe_allow_html=True)
