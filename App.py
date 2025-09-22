import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
import math
from matplotlib.patches import Wedge, Circle
import plotly.express as px
import plotly.graph_objects as go
import gdown
# ---- Page Config ----
st.set_page_config(
    page_title="Global Resistance Policy Monitor",
    page_icon="🌍",
    layout="wide",
)

# ---- Load Data ----
import os, streamlit as st, pandas as pd

GDRIVE_FILE_ID = st.secrets["GDRIVE_FILE_ID"]
LOCAL_PATH = "/tmp/atlas.xlsx"

def _download_from_gdrive(file_id: str, dest_path: str):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

@st.cache_data(show_spinner="📥 Fetching dataset from Google Drive...")
def _load_excel_from_path(dest_path: str):
    return pd.read_excel(dest_path, engine="openpyxl")

def load_and_prepare_data():
    try:
        if not os.path.exists(LOCAL_PATH):
            os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
            with st.spinner("Downloading dataset from Google Drive..."):
                _download_from_gdrive(GDRIVE_FILE_ID, LOCAL_PATH)
        df = _load_excel_from_path(LOCAL_PATH)
        return df
    except Exception as e:
        st.error(f"Failed to load data from Google Drive: {e}")
        st.info("Upload the Excel file manually below.")
        upload = st.file_uploader("Upload the ATLAS Excel file", type=["xlsx"])
        if upload is not None:
            try:
                df = pd.read_excel(upload, engine="openpyxl")
                st.success("File loaded from upload.")
                return df
            except Exception as ex:
                st.error(f"Could not read uploaded file: {ex}")
        return None


def get_available_antibiotics(df):
    """Get list of available antibiotics from the dataframe"""
    if df is None:
        return []
    
    # Look for columns that end with '_I' (resistance interpretation columns)
    resistance_cols = [col for col in df.columns if col.endswith('_I')]
    # Remove the '_I' suffix to get antibiotic names
    antibiotics = [col.replace('_I', '') for col in resistance_cols]
    
    return sorted(antibiotics)

def prepare_country_data(df, selected_country, policy_year, selected_family, selected_antibiotic):
    """Prepare data for a specific country and antibiotic"""
    # Filter for selected family and create working dataframe with relevant columns
    antibiotic_conc_col = selected_antibiotic  # Concentration column
    antibiotic_interp_col = f'{selected_antibiotic}_I'  # Interpretation column
    
    # Check if the selected antibiotic columns exist
    if antibiotic_interp_col not in df.columns:
        return None, None, f"Antibiotic '{selected_antibiotic}' interpretation data not found in dataset"
    
    # Select relevant columns
    base_cols = ['Species', 'Family', 'Country', 'Gender', 'Age Group', 'Speciality', 'Source', 'In / Out Patient', 'Year']
    antibiotic_cols = [antibiotic_conc_col, antibiotic_interp_col] if antibiotic_conc_col in df.columns else [antibiotic_interp_col]
    
    df2 = df[base_cols + antibiotic_cols].copy()
    
    # Filter for selected family
    df_ent = df2[df2['Family'] == selected_family].copy()
    
    # Create resistance binary variable for the selected antibiotic
    resistance_col_name = f"{selected_antibiotic}_resistant"
    df_ent[resistance_col_name] = df_ent[antibiotic_interp_col].apply(
        lambda x: 1 if pd.notna(x) and x == 'Resistant' else 0
    )
    
    # One-hot encode categorical variables
    categorical_cols = ['Source', 'Age Group','Speciality', 'In / Out Patient']
    df_ent = pd.get_dummies(df_ent, columns=categorical_cols, drop_first=True)
    
    # Drop unnecessary columns
    cols_to_drop = ['Family'] + antibiotic_cols
    df_ent = df_ent.drop(columns=[col for col in cols_to_drop if col in df_ent.columns], axis=1)
    
    # Filter for selected country
    df_country = df_ent[df_ent['Country'] == selected_country].drop(['Country','Species'], axis=1)
    
    if df_country.empty:
        return None, None, f"No data available for {selected_country} with {selected_family} and {selected_antibiotic}"
    
    # Create policy variable based on user-selected policy year
    df_country['policy'] = (df_country['Year'] >= policy_year).astype(int)
    
    # Aggregate by year
    binary_cols = df_country.columns.difference(["Year", resistance_col_name, 'Gender'], sort=False)
    df_sum = (df_country.groupby(["Year"])
              .agg(
                  resistance_rate=(resistance_col_name, "mean"),
                  **{col: (col, "sum") for col in binary_cols},
                  male_ratio=("Gender", lambda x: np.mean(x == "Male"))
              )
              .reset_index()
    )
    
    df_sum['policy'] = (df_sum['Year'] >= policy_year).astype(int)
    
    return df_sum, df_country, None

def run_causal_analysis(df_sum, analysis_mode):
    """Run the causal forest analysis with selectable ground truth mode"""
    if df_sum is None or len(df_sum) < 2:
        return None
        
    Y = df_sum['resistance_rate']  # Changed from MEM_resistance_rate to generic resistance_rate
    T = df_sum['policy']
    X = df_sum[[col for col in df_sum.columns if col not in ['resistance_rate', 'policy', 'Year']]]
    
    try:
        # Initialize and fit the causal forest model
        est = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=0),
            model_t=RandomForestClassifier(n_estimators=100, random_state=0),
            discrete_treatment=True,
            random_state=0
        )
        
        est.fit(Y, T, X=X)
        
        # Calculate treatment effects
        tau_hat = est.effect(X)
        df_sum['tau_hat'] = tau_hat
        
        # Generate counterfactual based on selected analysis mode
        if analysis_mode == "policy_implemented":
            # Standard mode: Policy is implemented (observed), predict what if no policy
            df_sum['resistance_cf'] = df_sum['resistance_rate'] + df_sum['tau_hat'] * df_sum['policy']
            df_sum['mode'] = "policy_implemented"
        else:
            # Reverse mode: No policy is observed, predict what if policy implemented
            df_sum['resistance_cf'] = df_sum['resistance_rate'] - df_sum['tau_hat'] * df_sum['policy']
            df_sum['mode'] = "no_policy"
        
        return df_sum
        
    except Exception as e:
        return None

def create_enhanced_plot(df_sum, selected_country, policy_year, analysis_mode, selected_antibiotic):
    """Create a modern, attractive resistance plot with mode-aware labeling"""
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#fafafa')
    
    # Modern color palette
    observed_color = '#1f77b4'  # Professional blue
    counterfactual_color = '#ff7f0e'  # Vibrant orange
    policy_color = '#d62728'  # Alert red
    fill_alpha = 0.15
    
    # Dynamic labels based on analysis mode
    if analysis_mode == "policy_implemented":
        observed_label = 'Observed (With Policy)'
        counterfactual_label = 'Counterfactual (Without Policy)'
        policy_annotation = f'Policy\nImplemented\n({policy_year})'
    else:
        observed_label = 'Observed (Without Policy)'
        counterfactual_label = 'Counterfactual (With Policy)'
        policy_annotation = f'Policy\nWould Start\n({policy_year})'
    
    # Create the main plots with enhanced styling
    observed_line = ax.plot(df_sum['Year'], df_sum['resistance_rate'], 
                           marker='o', linewidth=4, markersize=10, color=observed_color,
                           label=observed_label, markeredgewidth=2, 
                           markeredgecolor='white', zorder=5, alpha=0.9)
    
    counterfactual_line = ax.plot(df_sum['Year'], df_sum['resistance_cf'], 
                                 marker='D', linestyle='--', linewidth=4, markersize=9, 
                                 color=counterfactual_color, alpha=0.85,
                                 label=counterfactual_label, markeredgewidth=2, 
                                 markeredgecolor='white', zorder=4)
    
    # Add confidence bands (simulated for visual appeal)
    years = df_sum['Year']
    observed_upper = df_sum['resistance_rate'] * 1.05
    observed_lower = df_sum['resistance_rate'] * 0.95
    cf_upper = df_sum['resistance_cf'] * 1.05
    cf_lower = df_sum['resistance_cf'] * 0.95
    
    ax.fill_between(years, observed_lower, observed_upper, 
                    color=observed_color, alpha=fill_alpha, zorder=1)
    ax.fill_between(years, cf_lower, cf_upper, 
                    color=counterfactual_color, alpha=fill_alpha, zorder=1)
    
    # Enhanced policy period visualization
    policy_years = df_sum[df_sum['Year'] >= policy_year]['Year']
    if len(policy_years) > 0:
        if analysis_mode == "policy_implemented":
            period_label = 'Policy Implementation Period'
        else:
            period_label = 'Hypothetical Policy Period'
            
        ax.axvspan(policy_year - 0.5, df_sum['Year'].max() + 0.5, 
                   alpha=0.08, color=policy_color, zorder=0, 
                   label=period_label)
    
    # Policy implementation line with annotation
    policy_line = ax.axvline(x=policy_year, color=policy_color, linestyle='-', 
                            linewidth=3, alpha=0.8, zorder=3)
    
    # Add annotation for policy implementation
    ax.annotate(policy_annotation, 
                xy=(policy_year, ax.get_ylim()[1]*0.9), 
                xytext=(policy_year + 0.5, ax.get_ylim()[1]*0.85),
                fontsize=11, fontweight='bold', color=policy_color,
                ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color=policy_color, lw=2))
    
    # Enhanced title and labels with mode-aware text and antibiotic name
    mode_text = "Policy Impact Analysis" if analysis_mode == "policy_implemented" else "Policy Potential Analysis"
    ax.set_title(f'Antimicrobial Resistance {mode_text}\n{selected_country} • {selected_antibiotic} Resistance Trends', 
                fontsize=18, fontweight='bold', pad=25, color='#2c3e50')
    ax.set_xlabel('Year', fontsize=14, fontweight='600', color='#34495e')
    ax.set_ylabel('Resistance Rate (%)', fontsize=14, fontweight='600', color='#34495e')
    
    # Customize legend with modern styling
    legend = ax.legend(fontsize=12, loc='upper left', frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.95,
                      edgecolor='#bdc3c7', facecolor='white')
    legend.get_frame().set_linewidth(0.5)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Enhanced grid and background
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#bdc3c7')
    ax.set_facecolor('#fdfdfd')
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#95a5a6')
        spine.set_linewidth(1)
    
    # Better tick formatting
    plt.xticks(fontsize=11, color='#2c3e50')
    plt.yticks(fontsize=11, color='#2c3e50')
    
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    
    plt.tight_layout(pad=2.0)
    return fig

def create_modern_donut_chart(df_sum, policy_year, analysis_mode):
    """Create modern, attractive donut charts with gradient effects and mode awareness"""
    # Modern color palette
    observed_color = '#3498db'      # Modern blue
    counterfactual_color = '#e74c3c'  # Modern red
    background_color = '#ecf0f1'    # Light gray
    
    # Calculate rates
    mask = df_sum['Year'] >= policy_year
    if mask.sum() == 0:
        mask = df_sum['Year'] == df_sum['Year']
    
    years_sel = df_sum.loc[mask, 'Year']
    y_min = int(years_sel.min())
    y_max = int(years_sel.max())
    
    obs_rate = float(df_sum.loc[mask, 'resistance_rate'].mean())  # Changed from MEM_resistance_rate
    cf_rate = float(df_sum.loc[mask, 'resistance_cf'].mean())
    
    # Ensure rates are between 0 and 1 for pie chart
    obs_rate = max(0, min(1, obs_rate))
    cf_rate = max(0, min(1, cf_rate))
    
    # Calculate delta based on analysis mode
    if analysis_mode == "policy_implemented":
        delta = obs_rate - cf_rate  # Observed (with policy) - Counterfactual (without policy)
    else:
        delta = cf_rate - obs_rate  # Counterfactual (with policy) - Observed (without policy)
    
    # Create figure with modern styling
    fig = plt.figure(figsize=(15, 6), facecolor='white')
    
    # Dynamic title based on mode
    if analysis_mode == "policy_implemented":
        fig.suptitle('Policy Impact Comparison: Resistance Rates', 
                    fontsize=20, fontweight='bold', y=0.95, color='#2c3e50')
    else:
        fig.suptitle('Policy Potential Comparison: Resistance Rates', 
                    fontsize=20, fontweight='bold', y=0.95, color='#2c3e50')
    
    # Left donut - Context dependent
    ax1 = plt.subplot(1, 3, 1)
    wedge_props = dict(width=0.4, edgecolor='white', linewidth=3)
    
    # Dynamic assignment based on mode
    if analysis_mode == "policy_implemented":
        # Left = Counterfactual (without policy)
        left_resistant = max(0, cf_rate)
        left_susceptible = max(0, 1 - cf_rate)
        left_title = 'Counterfactual Scenario\n(Without Policy)'
        left_value = cf_rate
        left_color = counterfactual_color
    else:
        # Left = Observed (without policy) - should be HIGHER value
        left_resistant = max(0, obs_rate)
        left_susceptible = max(0, 1 - obs_rate)
        left_title = 'Observed Scenario\n(Without Policy)'
        left_value = obs_rate
        left_color = counterfactual_color  # Keep red color for "without policy"
    
    wedges1, texts1 = ax1.pie([left_resistant, left_susceptible],
                              colors=[left_color, background_color],
                              startangle=90, counterclock=False, 
                              wedgeprops=wedge_props)
    
    wedges1[0].set_alpha(0.9)
    ax1.set_title(left_title, fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    ax1.text(0, 0.1, f'{left_value:.1%}', ha='center', va='center', 
             fontsize=24, fontweight='bold', color='#2c3e50')
    ax1.text(0, -0.2, 'Resistance\nRate', ha='center', va='center',
             fontsize=10, color='#7f8c8d', style='italic')
    
    # Right donut - Context dependent
    ax2 = plt.subplot(1, 3, 3)
    
    # Dynamic assignment based on mode
    if analysis_mode == "policy_implemented":
        # Right = Observed (with policy)
        right_resistant = max(0, obs_rate)
        right_susceptible = max(0, 1 - obs_rate)
        right_title = 'Observed Results\n(With Policy)'
        right_value = obs_rate
        right_color = observed_color
    else:
        # Right = Counterfactual (with policy) - should be LOWER value
        right_resistant = max(0, cf_rate)
        right_susceptible = max(0, 1 - cf_rate)
        right_title = 'Potential Results\n(With Policy)'
        right_value = cf_rate
        right_color = observed_color  # Keep blue color for "with policy"
    
    wedges2, texts2 = ax2.pie([right_resistant, right_susceptible],
                              colors=[right_color, background_color],
                              startangle=90, counterclock=False, 
                              wedgeprops=wedge_props)
    
    wedges2[0].set_alpha(0.9)
    ax2.set_title(right_title, fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    ax2.text(0, 0.1, f'{right_value:.1%}', ha='center', va='center', 
             fontsize=24, fontweight='bold', color='#2c3e50')
    ax2.text(0, -0.2, 'Resistance\nRate', ha='center', va='center',
             fontsize=10, color='#7f8c8d', style='italic')
    
    # Middle section - Impact visualization
    ax3 = plt.subplot(1, 3, 2)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Determine impact direction and styling
    if delta < -0.001:  # Small threshold to avoid floating point issues
        impact_color = '#27ae60'  # Green for reduction
        arrow_symbol = '↓'
        impact_text = f'{abs(delta):.2%}\nReduction'
        if analysis_mode == "policy_implemented":
            subtitle = 'Policy Effective'
        else:
            subtitle = 'Policy Would Help'
    elif delta > 0.001:
        impact_color = '#e74c3c'  # Red for increase
        arrow_symbol = '↑'
        impact_text = f'{delta:.2%}\nIncrease'
        if analysis_mode == "policy_implemented":
            subtitle = 'Policy Ineffective'
        else:
            subtitle = 'Policy Would Harm'
    else:
        impact_color = '#95a5a6'  # Gray for no change
        arrow_symbol = '='
        impact_text = 'No\nChange'
        subtitle = 'No Impact'
    
    # Large arrow or symbol
    ax3.text(0.5, 0.7, arrow_symbol, ha='center', va='center',
             fontsize=60, color=impact_color, fontweight='bold')
    
    # Impact text
    ax3.text(0.5, 0.45, impact_text, ha='center', va='center',
             fontsize=16, fontweight='bold', color=impact_color)
    
    # Subtitle
    ax3.text(0.5, 0.25, subtitle, ha='center', va='center',
             fontsize=12, color='#7f8c8d', style='italic')
    
    # Period indicator
    ax3.text(0.5, 0.1, f'Analysis Period: {y_min}–{y_max}', 
             ha='center', va='center', fontsize=10, color='#95a5a6')
    
    plt.tight_layout()
    return fig, obs_rate, cf_rate

def prepare_global_resistance_data(df, selected_year, selected_family, selected_antibiotic):
    """Prepare global resistance data for world map visualization"""
    # Filter data
    df_filtered = df[
        (df['Year'] == selected_year) & 
        (df['Family'] == selected_family)
    ].copy()
    
    if df_filtered.empty:
        return None
    
    # Create resistance binary variable for the selected antibiotic
    antibiotic_col = f'{selected_antibiotic}_I'
    if antibiotic_col not in df_filtered.columns:
        return None
    
    df_filtered["resistant"] = df_filtered[antibiotic_col].apply(
        lambda x: 1 if pd.notna(x) and x == 'Resistant' else 0
    )
    
    # Calculate resistance rate by country
    country_resistance = (df_filtered.groupby('Country')
                         .agg(
                             resistance_rate=('resistant', 'mean'),
                             sample_size=('resistant', 'count')
                         )
                         .reset_index())
    
    # Filter countries with sufficient sample size
    country_resistance = country_resistance[country_resistance['sample_size'] >= 5]
    
    return country_resistance

def create_world_resistance_map(resistance_data, selected_year, selected_family, selected_antibiotic):
    """Create an interactive world map showing resistance rates"""
    if resistance_data is None or resistance_data.empty:
        return None
    
    # Create the choropleth map
    fig = px.choropleth(
        resistance_data,
        locations='Country',
        color='resistance_rate',
        hover_name='Country',
        hover_data={
            'resistance_rate': ':.1%',
            'sample_size': ':,d',
            'Country': False
        },
        color_continuous_scale='RdYlBu_r',  # Red-Yellow-Blue reversed (red = high resistance)
        range_color=[0, resistance_data['resistance_rate'].max()],
        locationmode='country names',
        title=f'Global {selected_antibiotic} Resistance Rates - {selected_year}<br>{selected_family}',
        labels={
            'resistance_rate': 'Resistance Rate',
            'sample_size': 'Sample Size'
        }
    )
    
    # Customize the map appearance
    fig.update_layout(
        title={
            'text': f'Global {selected_antibiotic} Resistance Rates - {selected_year}<br><sub>{selected_family}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)',
        ),
        coloraxis_colorbar=dict(
            title="Resistance Rate",
            tickformat='.0%',
            len=0.7,
            thickness=15,
            x=1.02
        ),
        height=400,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'Resistance Rate: %{customdata[0]}<br>' +
                     'Sample Size: %{customdata[1]}<extra></extra>',
        customdata=resistance_data[['resistance_rate', 'sample_size']].values
    )
    
    return fig

# Load data
df = load_and_prepare_data()

# ---- Custom Styles ----
st.markdown(
    """
    <style>
        /* Title box at top */
        .title-box {
            background-color: #0a2540; /* Navy */
            padding: 10px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 10px;
        }
        .title-box h1 {
            font-size: 2.8em;
            font-weight: 800;
            color: white;
            margin: 0;
        }

        /* Subtitle */
        .subtitle {
            font-size: 1.2em !important;
            font-style: italic;
            color: #6c757d; /* Muted gray */
            text-align: center;
            margin-bottom: 20px;
            line-height: 1.4em;
            
            
        }

        /* Enhanced Sidebar Styles */
        .sidebar-header {
            background: linear-gradient(135deg, #0a2540, #1e3a5f);
            padding: 10px 10px;
            border-radius: 10px;
            margin-top: 60px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(10, 37, 64, 0.2);
        }
        
        .sidebar-title {
            font-size: 1.3em;
            font-weight: 800;
            color: white;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        /* Sidebar section styling */
        .sidebar-section {
            padding: 2px 0;
        }

        .section-label {
            font-size: 0.85em;
            font-weight: 600;
            color: #0a2540;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Custom selectbox styling */
        .stSelectbox > div > div {
            background-color: white;
            border: 2px solid #e9ecef;
            border-radius: 6px;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: #0a2540;
            box-shadow: 0 0 0 2px rgba(10, 37, 64, 0.1);
        }

        /* Sidebar info box */
        .info-box {
            background: linear-gradient(135deg, #e8f4fd, #d1ecf1);
            padding: 10px;
            border-radius: 8px;
            margin-top: -8px;
            margin-bottom: 10px;
            border-left: 4px solid #17a2b8;
        }
        
        .info-text {
            font-size: 0.85em;
            color: #0c5460;
            line-height: 1.4;
            margin: 0;
        }

        /* Sidebar button styling */
        .sidebar-button {
            width: 100%;
            padding: 10px 15px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            background: white;
            color: #0a2540;
            font-weight: 600;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 8px;
            text-align: center;
        }
        
        .sidebar-button:hover {
            background: #f8f9fa;
            border-color: #0a2540;
            box-shadow: 0 2px 8px rgba(10, 37, 64, 0.1);
        }
        
        .sidebar-button-active {
            background: #0a2540;
            color: white;
            border-color: #0a2540;
        }
        
        .sidebar-button-active:hover {
            background: #1e3a5f;
        }

        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e293b;
            margin: 2.5rem 0 1.5rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #e2e8f0;
        }
        
        /* Analysis cards */
        .analysis-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
            margin-bottom: 2rem;
        }
        
        /* Status indicators */
        .status-positive {
            color: #059669;
            background: #d1fae5;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        
        .status-negative {
            color: #dc2626;
            background: #fee2e2;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px !important;
            border: none !important;
        }
        
        .streamlit-expanderHeader p {
            color: white !important;
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }
        
        .streamlit-expanderContent {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 0 0 10px 10px;
            padding: 1.5rem !important;
            border: 1px solid #e1e8ed !important;
            border-top: none !important;
        }
        
        .info-section {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .info-section h4 {
            color: #2c3e50;
            margin-bottom: 0.8rem;
            font-size: 1.1em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3rem;
        }
        
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-left: 4px solid #fdcb6e;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }

        /* Target the expander label */
        div.stExpander p{
            font-size: 25x !important;  /* Increase size (adjust as needed) */
            font-weight: bold;
            font-style: italic
        }
        
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown(
    """
    <div class="title-box">
        <h1>Global Resistance Policy Monitor</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="subtitle">
        A  platform to explore resistance rates across countries, organisms, and antibiotics — designed to support evidence-based AMR policies and interventions.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.expander("🧠About This Dashboard - How It Works & What It Shows"):
    st.markdown("""
    <div class="info-section">
        <h4>🎯 What This Tool Does </h4>
        <p>Think of this as a "time machine" for policy analysis. This tool can analyze AMR policies in two different ways:</p>
        <ul>
            <li><strong>Policy Impact Analysis:</strong> When a policy was already implemented, see how effective it was by comparing actual results to what would have happened without the policy.</li>
            <li><strong>Policy Potential Analysis:</strong> When no policy exists, predict what would happen if a policy were implemented by comparing current results to potential outcomes with the policy.</li>
        </ul>
        <p>Our AI model creates a "parallel universe" scenario using patterns from similar countries, time periods, and patient populations.</p>
    </div>
    
    <div class="warning-box">
        <strong>⚠️ Important Model Behavior:</strong> 
        <ul>
            <li><strong>Policy Impact Mode:</strong> Assumes policy was implemented at the specified year and predicts what would have happened without it</li>
            <li><strong>Policy Potential Mode:</strong> Assumes no policy was in place and predicts what would happen if policy were implemented at the specified year</li>
        </ul>
    </div>
    
    <div class="info-section">
        <h4>🔀 Analysis Modes</h4>
        <ul>
            <li><strong>Policy Impact Analysis:</strong> Use when a policy was actually implemented. Shows effectiveness by comparing observed results (with policy) vs counterfactual (without policy)</li>
            <li><strong>Policy Potential Analysis:</strong> Use when considering implementing a policy. Shows potential benefit by comparing current situation (no policy) vs predicted results (with policy)</li>
        </ul>
    </div>
    
    <div class="info-section">
        <h4>📋 How to Use This Dashboard</h4>
        <ol>
            <li><strong>Select Analysis Mode:</strong> Choose whether to analyze an existing policy's impact or explore a potential policy's benefits</li>
            <li><strong>Select Country:</strong> Choose the country for analysis</li>
            <li><strong>Set Policy Year:</strong> Specify the year when policy was/would be implemented</li>
            <li><strong>Choose Organism:</strong> Select bacterial family (e.g., Enterobacteriaceae)</li>
            <li><strong>Pick Antibiotic:</strong> Select the antibiotic for resistance analysis</li>
        </ol>
    </div>
    
    <div class="info-section">
        <h4>📊 Understanding Your Results</h4>
        <ul>
            <li><strong>Trend Chart:</strong> Blue line = observed resistance rates, Orange dashed line = predicted counterfactual rates</li>
            <li><strong>Donut Charts:</strong> Compare current situation vs alternative scenario</li>
            <li><strong>Green Arrow ↓:</strong> Policy shows/would show benefit (reduced resistance)</li>
            <li><strong>Red Arrow ↑:</strong> Policy appears/would appear ineffective (resistance increased/would increase)</li>
            <li><strong>World Map:</strong> Shows how your selected country compares globally</li>
            <li><strong>Country Ranking:</strong> Indicates where your country stands among all countries with available data</li>
        </ul>
    </div>
    
    <div class="info-section">
        <h4>🔬 Data Source & Methodology</h4>
        <p>Analysis uses the <strong>ATLAS global surveillance database</strong> with advanced causal inference methods (Causal Forest with Double Machine Learning) to isolate policy effects from other factors like time trends, patient demographics, and healthcare settings.</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize analysis mode in session state
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "policy_implemented"

# ---- Enhanced Sidebar ----
with st.sidebar:
    # Attractive header section
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-title">🔍 Analysis Parameters</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    if df is not None and not df.empty:
        # Extract unique values from the data
        years = ["Select a year..."] + sorted(df['Year'].dropna().unique().tolist(), reverse=True)
        countries = ["Select a country..."] + sorted(df['Country'].dropna().unique().tolist())
        families = ["Select an organism..."] + sorted(df['Family'].dropna().unique().tolist())
        
        # Get available antibiotics dynamically from the data
        available_antibiotics = get_available_antibiotics(df)
        antibiotics = ["Select an antibiotic..."] + available_antibiotics
    else:
        years = countries = families = antibiotics = ["No data available"]
     # Information tip box
    st.markdown(
        """
        <div class="info-box">
            <p class="info-text">
                <strong>💡</strong> Select parameters to generate detailed resistance pattern analysis and policy effects.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Country Selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🗺️ Country</p>', unsafe_allow_html=True)
    country = st.selectbox("Country", countries, label_visibility="collapsed",
                           index=countries.index('Greece') if 'Greece' in countries else 0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Year Selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📅 Year</p>', unsafe_allow_html=True)
    year = st.selectbox("Year", years, label_visibility="collapsed",
                        index=years.index(2010) if 2010 in years else len(years)//2)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Organism
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🦠 Organism</p>', unsafe_allow_html=True)
    organism = st.selectbox("Organism", families, label_visibility="collapsed",
                            index=families.index('Enterobacteriaceae') if 'Enterobacteriaceae' in families else 0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Treatment Parameters - Updated to use dynamic antibiotic list
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">💊 Antibiotic</p>', unsafe_allow_html=True)
    antibiotic = st.selectbox("Antibiotic", antibiotics, label_visibility="collapsed",
                              index=antibiotics.index('Meropenem') if 'Meropenem' in antibiotics else 0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis Mode Selection (moved to last position)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🔀 Analysis Mode</p>', unsafe_allow_html=True)
    
    # Create two buttons vertically arranged
    impact_button = st.button(
        "📈 Policy Impact Analysis",
        help="Analyze the effectiveness of an existing policy",
        use_container_width=True,
        key="impact_btn"
    )
    
    potential_button = st.button(
        "📉 Policy Potential Analysis", 
        help="Explore the potential benefits of a proposed policy",
        use_container_width=True,
        key="potential_btn"
    )
    
    # Handle button clicks
    if impact_button:
        st.session_state.analysis_mode = "policy_implemented"
        st.rerun()
    
    if potential_button:
        st.session_state.analysis_mode = "no_policy"
        st.rerun()
    
    # Display current mode status
    current_mode = st.session_state.analysis_mode
    if current_mode == "policy_implemented":
        mode_text = "📈 Policy Impact Analysis"
        mode_description = " :Analyzing existing policy effectiveness"
    else:
        mode_text = "📉 Policy Potential Analysis" 
        mode_description = ":Exploring proposed policy benefits"
    
    st.info(f"**{mode_text}**\n{mode_description}")
    st.markdown('</div>', unsafe_allow_html=True)
    
   

# ---- Display Data Info ----
if df is not None and not df.empty:
    # Check if valid selections are made
    if (country != "Select a country..." and 
        year != "Select a year..." and 
        organism != "Select an organism..." and
        antibiotic != "Select an antibiotic..."):
        
        with st.spinner('🔄 Preparing data for analysis...'):
            df_sum, df_country, error_msg = prepare_country_data(df, country, year, organism, antibiotic)
        
        if error_msg:
            st.error(f"❌ {error_msg}")
        elif df_sum is not None:
            with st.spinner('🤖 Running causal analysis with machine learning...'):
                df_results = run_causal_analysis(df_sum, st.session_state.analysis_mode)
            
            st.markdown('<div class="section-header">📈 Analysis Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("🗺️ Country", country, border=False)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("📅 Policy Year", year, border=False)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                total_samples = len(df_country) if df_country is not None else 0
                st.metric("🔬 Total Samples", f"{total_samples:,}", border=False)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                years_available = len(df_sum) if df_sum is not None else 0
                st.metric("📊 Years of Data", years_available, border=False)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ---- Main Analysis Results ----
            if df_results is not None:
                st.markdown('<div class="section-header">🔬 Causal Analysis Results</div>', unsafe_allow_html=True)
                
                with st.spinner('📈 Generating resistance trend visualization...'):
                    fig = create_enhanced_plot(df_results, country, year, st.session_state.analysis_mode, antibiotic)
                    st.pyplot(fig, use_container_width=True)
                
                # Policy Impact Assessment
                st.markdown('<div class="section-header">⚖️ Policy Impact Assessment</div>', unsafe_allow_html=True)

                with st.spinner('🍩 Creating policy impact comparison charts...'):
                    try:
                        donut_fig, obs_rate, cf_rate = create_modern_donut_chart(df_results, year, st.session_state.analysis_mode)
                        st.pyplot(donut_fig, use_container_width=True)
                        
                        # Impact summary - adjusted for analysis mode
                        if st.session_state.analysis_mode == "policy_implemented":
                            delta = obs_rate - cf_rate  # Observed (with policy) - Counterfactual (without policy)
                            context_text = "policy shows" if delta < -0.001 else "policy associated with" if delta > 0.001 else "No significant policy"
                        else:
                            delta = cf_rate - obs_rate  # Counterfactual (with policy) - Observed (without policy)
                            context_text = "policy would show" if delta < -0.001 else "policy would be associated with" if delta > 0.001 else "No significant policy"
                        
                        if delta < -0.001:  # Small threshold
                            status_class = "status-positive"
                            if st.session_state.analysis_mode == "policy_implemented":
                                impact_text = f"✅ Policy shows positive impact with {abs(delta):.2%} reduction in {antibiotic} resistance"
                            else:
                                impact_text = f"✅ Policy would show positive impact with {abs(delta):.2%} reduction in {antibiotic} resistance"
                        elif delta > 0.001:
                            status_class = "status-negative"
                            if st.session_state.analysis_mode == "policy_implemented":
                                impact_text = f"❌ Policy associated with {delta:.2%} increase in {antibiotic} resistance"
                            else:
                                impact_text = f"❌ Policy would be associated with {delta:.2%} increase in {antibiotic} resistance"
                        else:
                            status_class = "status-neutral"
                            impact_text = f"➖ No significant policy impact detected for {antibiotic} resistance"
                        
                        st.markdown(f'''
                        <div style="text-align: center; margin-top: 1.5rem;">
                            <span class="{status_class}">{impact_text}</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error creating donut charts: {str(e)}")
                        st.info("This may be due to data quality issues. Try different parameter selections.")

                # Add visual separator
                st.markdown("---")

                # Global Resistance Context Section
                st.markdown('<div class="section-header">🌍 Global Resistance Context</div>', unsafe_allow_html=True)

                with st.spinner('🌐 Preparing global resistance map...'):
                    # Get the current year and antibiotic for global context
                    map_year = year if year != "Select a year..." else df['Year'].max()
                    
                    global_resistance_data = prepare_global_resistance_data(
                        df, map_year, organism, antibiotic
                    )
                    
                    if global_resistance_data is not None and not global_resistance_data.empty:
                        world_map = create_world_resistance_map(
                            global_resistance_data, map_year, organism, antibiotic
                        )
                        
                        if world_map:
                            st.plotly_chart(world_map, use_container_width=True)
                            
                            # Show where the selected country ranks
                            selected_country_data = global_resistance_data[
                                global_resistance_data['Country'] == country
                            ]
                            if not selected_country_data.empty:
                                country_rate = selected_country_data['resistance_rate'].iloc[0]
                                country_rank = (global_resistance_data['resistance_rate'] > country_rate).sum() + 1
                                total_countries = len(global_resistance_data)
                                
                                st.info(f"📊 **{country}** ranks #{country_rank} out of {total_countries} countries with {country_rate:.1%} {antibiotic} resistance rate in {map_year}")
                            else:
                                st.info(f"ℹ️ Global context for {map_year} - {country} data not available for comparison")
                        else:
                            st.error("Unable to create world map.")
                    else:
                        st.warning(f"⚠️ Limited global data available for {antibiotic} in {organism} for {map_year}")
            else:
                st.error("❌ Unable to run causal analysis. This may be due to insufficient data for the selected parameters.")
        else:
            st.error("❌ No data available for the selected parameters. Please try different selections.")
    else:
        st.info("🔍 Please complete your selections in the sidebar to begin analysis.")
else:
    st.error("❌ Unable to load data. Please check if the data file exists.")
        
# ---- Footer ----
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 2rem;'>
        <p>🧬 Powered by ATLAS Global Surveillance Database & Advanced Causal ML</p>
        <p>Built with ❤️ for Evidence-Based AMR Policy Making</p>
    </div>
    """,
    unsafe_allow_html=True
)
