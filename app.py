import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import glob
import os
from itertools import combinations
from collections import Counter

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="ACA Smart Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e2e8f0; padding: 20px;
        border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); transition: 0.3s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px); box-shadow: 0 8px 15px rgba(0,0,0,0.08); border-color: #3b82f6;
    }
    div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.9rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #1e293b !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; border-radius: 8px; background-color: #f8fafc; 
        border: 1px solid #cbd5e1; padding: 0 20px;
    }
    .stTabs [data-baseweb="tab"] p { color: #334155 !important; font-weight: 600; font-size: 14px; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; border-color: #3b82f6 !important; }
    .stTabs [aria-selected="true"] p { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. CHART HELPER ---
def make_clean(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#94a3b8"),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest", 
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#000000"), bordercolor="#e2e8f0"),
        xaxis=dict(showgrid=False, showline=True, linecolor="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=None)
    )
    return fig

def group_small_slices(df, cat_col, val_col, top_n=8):
    df_grouped = df.groupby(cat_col)[val_col].sum().reset_index().sort_values(val_col, ascending=False)
    if len(df_grouped) <= top_n: return df_grouped
    top_df = df_grouped.head(top_n).copy()
    others_val = df_grouped.iloc[top_n:][val_col].sum()
    others_df = pd.DataFrame({cat_col: ['OTHERS'], val_col: [others_val]})
    return pd.concat([top_df, others_df], ignore_index=True)

# --- 3. DATA ENGINE & NLP ---
def parse_aca_date(val):
    try:
        if pd.isna(val) or str(val).strip() == "": return pd.NaT
        s_val = str(val).replace('.','').replace('-','')
        if s_val.isdigit():
            return pd.to_datetime('1899-12-30') + pd.to_timedelta(float(val), 'D')
        return pd.to_datetime(val, errors='coerce')
    except: return pd.NaT

def detect_client_type(name):
    if pd.isna(name): return "Unknown"
    name = str(name).upper()
    corp_keywords = ['PT ', 'PT.', 'CV ', 'CV.', 'UD ', 'UD.', 'CORP', 'INC.', 'LTD', 'YAYASAN', 'KOPERASI', 'BANK', 'FINANCE', 'HOTEL', 'RS ', 'RUMAH SAKIT']
    if any(keyword in name for keyword in corp_keywords):
        return "🏢 Corporate (B2B)"
    return "👤 Retail (B2C)"

@st.cache_data
def load_mega_data():
    files = glob.glob(os.path.join("data_produksi", "*.csv"))
    if not files: return None
        
    dfs = []
    for f in files:
        try:
            df_t = pd.read_csv(f, sep=None, engine='python', encoding='utf-8-sig', dtype=str, on_bad_lines='skip')
            df_t.columns = [str(c).strip().upper() for c in df_t.columns]
            dfs.append(df_t)
        except: continue
    
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)

    df['TGL_IN'] = df['DATE_INPUT'].apply(parse_aca_date)
    df['TGL_APP'] = df['USER_APPROVE_DATE'].apply(parse_aca_date)
    
    def clean_col(col_name):
        if col_name in df.columns:
            return pd.to_numeric(
                df[col_name].astype(str).str.replace(',', '', regex=False).str.replace('Rp', '', regex=False), 
                errors='coerce'
            ).fillna(0)
        return 0

    df['PREMIUM'] = clean_col('PREMIUM_GROSS')
    col_tsi = 'TSI' if 'TSI' in df.columns else 'TSI_OC'
    df['TSI_VAL'] = clean_col(col_tsi)

    df['RATE_PCT'] = df.apply(lambda x: (x['PREMIUM'] / x['TSI_VAL'] * 100) if x['TSI_VAL'] > 0 else 0, axis=1)
    df['SLA_HARI'] = (df['TGL_APP'] - df['TGL_IN']).dt.days
    df = df.dropna(subset=['TGL_IN'])
    
    df['TAHUN'] = df['TGL_IN'].dt.year.astype(str)
    df['BULAN_NUM'] = df['TGL_IN'].dt.month
    df['BULAN_NAMA'] = df['TGL_IN'].dt.strftime('%b')
    df['HARI'] = df['TGL_IN'].dt.day_name()
    df['TANGGAL'] = df['TGL_IN'].dt.day 
    df['JAM_INPUT'] = df['TGL_IN'].dt.hour
    
    text_cols = ['SEGMENT', 'TOC_DESCRIPTION', 'INPUT_NAME', 'TRANSACTION_TYPE', 'INSURED_NAME', 'MO_NAME', 'SOURCE_NAME']
    for c in text_cols:
        if c not in df.columns: df[c] = "Unknown"
        else: df[c] = df[c].fillna("Unknown").str.upper().str.strip()

    df['STATUS_SLA'] = df['SLA_HARI'].apply(lambda x: "Same Day" if x <= 0 else ("1 Day" if x <= 1 else "> 1 Day"))
    df['CLIENT_TYPE'] = df['INSURED_NAME'].apply(detect_client_type)
    
    return df

# --- 4. RENDERER ---
df_raw = load_mega_data()

if df_raw is not None:
    # --- SIDEBAR ---
    st.sidebar.markdown("### ⚙️ Filter Dashboard")
    all_years = sorted(df_raw['TAHUN'].unique())
    sel_years = st.sidebar.multiselect("Tahun", all_years, default=all_years, key="filter_tahun")
    
    df_filtered_year = df_raw[df_raw['TAHUN'].isin(sel_years)]
    all_users = sorted(df_filtered_year['INPUT_NAME'].unique())
    sel_users = st.sidebar.multiselect("Petugas", all_users, default=all_users, key="filter_petugas")
    
    df = df_filtered_year[df_filtered_year['INPUT_NAME'].isin(sel_users)] if sel_users else df_filtered_year
    st.sidebar.caption(f"Loaded: {len(df):,} Rows")

    # --- HEADER ---
    st.title("🛡️ Executive Production Dashboard")
    st.caption("Monitoring performa real-time PT Asuransi Central Asia - Branch Bogor")
    st.markdown("---")
    
    # --- KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi", f"{len(df):,}", "Polis")
    c2.metric("Gross Premium", f"{df['PREMIUM'].sum()/1e9:.2f} M", "Miliar IDR")
    
    valid_sla = df.dropna(subset=['SLA_HARI'])
    avg_sla = valid_sla['SLA_HARI'].mean() if not valid_sla.empty else 0
    c3.metric("Rerata SLA", f"{avg_sla:.2f} Hari", "Kecepatan")
    
    renew_prem = df[df['TRANSACTION_TYPE'].str.contains('RENEWAL', na=False)]['PREMIUM'].sum()
    ratio = (renew_prem/df['PREMIUM'].sum()*100) if df['PREMIUM'].sum() > 0 else 0
    c4.metric("Renewal Ratio", f"{ratio:.1f}%", "Retensi")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- TABS ---
    tabs = st.tabs([
        "💎 Insights", "🔮 AI Center", "👥 Nasabah (AI)", 
        "📈 Strategis", "🎯 Produk", "⚙️ Operasional", 
        "📅 Waktu", "🚀 Produktivitas", "⚖️ Risk & Rate", "🔍 Database"
    ])

    # --- TAB 0: INSIGHTS ---
    with tabs[0]:
        st.markdown("### 🧠 Deep Dive Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("⚠️ Pareto Risk: Concentration")
            mo_col = 'MO_NAME' if 'MO_NAME' in df.columns else 'SOURCE_NAME'
            df_pareto = df.groupby(mo_col)['PREMIUM'].sum().reset_index().sort_values('PREMIUM', ascending=False)
            df_pareto['CUM_PCT'] = df_pareto['PREMIUM'].cumsum() / df_pareto['PREMIUM'].sum() * 100
            
            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(x=df_pareto.head(20)[mo_col], y=df_pareto.head(20)['PREMIUM'], name='Premium', marker_color='#3b82f6'))
            fig_par.add_trace(go.Scatter(x=df_pareto.head(20)[mo_col], y=df_pareto.head(20)['CUM_PCT'], name='Cum %', yaxis='y2', mode='lines+markers', line=dict(color='#ef4444')))
            fig_par.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110], showgrid=False), hovermode="x unified")
            st.plotly_chart(make_clean(fig_par), use_container_width=True)
            
        with col2:
            st.subheader("📊 Dependency Ratio")
            top_20_count = int(len(df_pareto) * 0.2) or 1
            rev_top = df_pareto.head(top_20_count)['PREMIUM'].sum()
            total_rev = df_pareto['PREMIUM'].sum()
            pareto_ratio = (rev_top / total_rev * 100) if total_rev > 0 else 0
            
            st.markdown(f"""
            <div style="background-color:#f1f5f9; padding:20px; border-radius:10px;">
                <h4 style="margin:0; color:#64748b;">Pareto Index</h4>
                <h1 style="margin:0; font-size:3rem; color:#1e293b;">{pareto_ratio:.1f}%</h1>
                <p style="color:#475569;">dari Omzet berasal dari <b>Top 20%</b> Mitra.</p>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 1: AI CENTER (UPDATED) ---
    with tabs[1]:
        st.markdown("### 🔮 Advanced AI Analytics Hub")
        st.info("Fitur AI ini menggunakan data historis Anda untuk memprediksi tren dan memberikan rekomendasi bisnis.")
        
        # ROW 1: FORECAST & SEASONALITY
        c_ai1, c_ai2 = st.columns([3, 2])
        with c_ai1:
            st.subheader("📈 Revenue Forecast (Trend Projection)")
            df_ml = df.groupby('TGL_IN')['PREMIUM'].sum().reset_index().sort_values('TGL_IN')
            df_ml['Date_Ord'] = df_ml['TGL_IN'].map(pd.Timestamp.toordinal)
            
            if len(df_ml) > 1:
                z = np.polyfit(df_ml['Date_Ord'], df_ml['PREMIUM'], 1)
                p = np.poly1d(z)
                last_date = df_ml['TGL_IN'].max()
                future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
                
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=df_ml['TGL_IN'], y=df_ml['PREMIUM'], name='Actual', line=dict(color='#cbd5e1')))
                fig_fc.add_trace(go.Scatter(x=df_ml['TGL_IN'], y=p(df_ml['Date_Ord']), name='Trend', line=dict(color='#3b82f6', width=2)))
                fig_fc.add_trace(go.Scatter(x=future_dates, y=p([d.toordinal() for d in future_dates]), name='Forecast (30 Days)', line=dict(color='#f59e0b', dash='dot')))
                st.plotly_chart(make_clean(fig_fc), use_container_width=True)
            else: st.warning("Data belum cukup.")
            
        with c_ai2:
            st.subheader("📅 Seasonality Index (AI Pattern)")
            # Menghitung indeks musiman: Rata-rata bulan X / Rata-rata Total
            if not df.empty:
                monthly_avg = df.groupby('BULAN_NUM')['PREMIUM'].mean()
                total_avg = df['PREMIUM'].mean()
                seasonality = (monthly_avg / total_avg).reset_index(name='Index')
                seasonality['Month'] = seasonality['BULAN_NUM'].apply(lambda x: datetime(2023, x, 1).strftime('%b'))
                
                # Highlight best and worst
                colors = ['#ef4444' if x < seasonality['Index'].mean() else '#10b981' for x in seasonality['Index']]
                
                fig_seas = go.Figure(go.Bar(
                    x=seasonality['Month'], 
                    y=seasonality['Index'],
                    marker_color=colors,
                    text=seasonality['Index'].apply(lambda x: f"{x:.1f}x"),
                    textposition='auto'
                ))
                fig_seas.add_hline(y=1, line_dash="dot", annotation_text="Average Baseline", annotation_position="top right")
                st.plotly_chart(make_clean(fig_seas), use_container_width=True)

        st.markdown("---")
        
        # ROW 2: CROSS-SELLING & ANOMALY
        c_ai3, c_ai4 = st.columns(2)
        with c_ai3:
            st.subheader("🛒 Smart Bundling (Top Pairs)")
            st.caption("Rekomendasi paket produk yang paling sering dibeli bersamaan oleh satu nasabah.")
            
            # --- PROSES DATA (Sama seperti sebelumnya) ---
            transactions = df.groupby('INSURED_NAME')['TOC_DESCRIPTION'].apply(list)
            pair_counts = Counter()
            for products in transactions:
                unique_prods = sorted(list(set(products)))
                if len(unique_prods) > 1:
                    pair_counts.update(combinations(unique_prods, 2))
            
            if pair_counts:
                # Ambil TOP 10 Saja supaya chart fokus & rapi
                df_pairs = pd.DataFrame(pair_counts.most_common(10), columns=['Pair', 'Frequency'])
                
                # Buat nama Label yang mudah dibaca: "Produk A + Produk B"
                df_pairs['Bundling Name'] = df_pairs['Pair'].apply(lambda x: f"{x[0]} + {x[1]}")
                
                # --- VISUALISASI: HORIZONTAL BAR CHART ---
                # Sort ascending=True agar bar terpanjang ada di paling atas chart
                fig_bar = px.bar(
                    df_pairs.sort_values('Frequency', ascending=True), 
                    x='Frequency', 
                    y='Bundling Name',
                    text='Frequency', # Menampilkan angka di ujung bar
                    orientation='h',  # Horizontal supaya teks panjang terbaca
                    color='Frequency',
                    color_continuous_scale='Blues'
                )
                
                fig_bar.update_layout(
                    xaxis_title="Jumlah Kejadian (Transaksi)",
                    yaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    height=350, # Tinggi chart disesuaikan
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                # Agar teks angka ada di luar bar jika barnya pendek
                fig_bar.update_traces(textposition='outside') 
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # --- REKOMENDASI TEKS (Actionable) ---
                if not df_pairs.empty:
                    top_combo = df_pairs.iloc[0]['Pair']
                    st.success(f"💡 **Strategi:** Fokus tawarkan **{top_combo[1]}** kepada nasabah yang sudah memiliki **{top_combo[0]}**.")

            else:
                st.warning("Belum cukup data nasabah dengan kepemilikan > 1 jenis polis untuk analisis bundling.")

        with c_ai4:
            st.subheader("🚨 Rate Anomaly Detector")
            st.caption("Mendeteksi polis dengan Rate Premi yang tidak wajar (Outlier).")
            df_anom = df[(df['PREMIUM'] > 0) & (df['TSI_VAL'] > 0)].copy()
            df_anom['RATE_Z'] = (df_anom['RATE_PCT'] - df_anom['RATE_PCT'].mean()) / df_anom['RATE_PCT'].std()
            anomalies = df_anom[df_anom['RATE_Z'].abs() > 3].sort_values('RATE_Z', ascending=False)
            st.metric("Suspicious Policies", len(anomalies), delta_color="inverse")
            if not anomalies.empty:
                st.dataframe(anomalies[['POLICYNO', 'INSURED_NAME', 'RATE_PCT', 'TOC_DESCRIPTION']], hide_index=True, height=250)
            
        # ROW 3: NLP & HEATMAP
        st.markdown("---")
        c_ai5, c_ai6 = st.columns(2)
        with c_ai5:
             st.subheader("🏢 NLP Entity Classifier")
             df_client = df.groupby('CLIENT_TYPE')['PREMIUM'].sum().reset_index()
             fig_client = px.pie(df_client, values='PREMIUM', names='CLIENT_TYPE', color='CLIENT_TYPE', hole=0.6,
                                color_discrete_map={'🏢 Corporate (B2B)': '#3b82f6', '👤 Retail (B2C)': '#10b981'})
             st.plotly_chart(make_clean(fig_client), use_container_width=True)
             
        with c_ai6:
            st.subheader("🔥 Product-Market Fit Heatmap")
            df_heat = df.groupby(['SEGMENT', 'TOC_DESCRIPTION'])['PREMIUM'].sum().reset_index()
            top_seg = df.groupby('SEGMENT')['PREMIUM'].sum().nlargest(8).index
            top_toc = df.groupby('TOC_DESCRIPTION')['PREMIUM'].sum().nlargest(8).index
            df_heat = df_heat[df_heat['SEGMENT'].isin(top_seg) & df_heat['TOC_DESCRIPTION'].isin(top_toc)]
            fig_heat = px.density_heatmap(df_heat, x='SEGMENT', y='TOC_DESCRIPTION', z='PREMIUM', color_continuous_scale='Blues')
            fig_heat.update_layout(coloraxis_showscale=False)
            st.plotly_chart(make_clean(fig_heat), use_container_width=True)

    # --- TAB 2: NASABAH (RFM AI) ---
    with tabs[2]:
        st.subheader("👥 Smart Customer Segmentation (RFM Model)")
        st.caption("AI Clustering berdasarkan: Recency (Terakhir Beli), Frequency (Sering Beli), Monetary (Nilai Beli).")
        
        snapshot_date = df['TGL_IN'].max() + timedelta(days=1)
        rfm = df.groupby('INSURED_NAME').agg({
            'TGL_IN': lambda x: (snapshot_date - x.max()).days,
            'POLICYNO': 'count',
            'PREMIUM': 'sum'
        }).rename(columns={'TGL_IN': 'Recency', 'POLICYNO': 'Frequency', 'PREMIUM': 'Monetary'})
        
        try:
            r_groups = pd.qcut(rfm['Recency'], q=4, labels=range(4, 0, -1), duplicates='drop')
            f_groups = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=range(1, 5))
            m_groups = pd.qcut(rfm['Monetary'], q=4, labels=range(1, 5))
            rfm = rfm.assign(R=r_groups.values, F=f_groups.values, M=m_groups.values)
            
            def segment_me(row):
                score = row['R'] + row['F'] + row['M']
                if score >= 10: return '🏆 Champions (VIP)'
                elif score >= 7: return '💎 Loyal Customers'
                elif row['R'] == 1 and row['M'] >= 3: return '💤 Sleeping Giants'
                elif row['R'] >= 3 and row['F'] == 1: return '🌱 New Customers'
                else: return '⚠️ At Risk'
                
            rfm['Segment'] = rfm.apply(segment_me, axis=1)
            
            c_rfm1, c_rfm2 = st.columns([1, 2])
            with c_rfm1:
                seg_counts = rfm['Segment'].value_counts().reset_index()
                seg_counts.columns = ['Segment', 'Count']
                fig_pie = px.pie(seg_counts, values='Count', names='Segment', hole=0.5, 
                                 color='Segment', color_discrete_map={
                                     '🏆 Champions (VIP)': '#10b981', '💎 Loyal Customers': '#3b82f6',
                                     '💤 Sleeping Giants': '#f59e0b', '🌱 New Customers': '#8b5cf6', '⚠️ At Risk': '#ef4444'
                                 })
                st.plotly_chart(make_clean(fig_pie), use_container_width=True)
            with c_rfm2:
                sel_seg = st.selectbox("Filter Segmen:", rfm['Segment'].unique())
                st.dataframe(
                    rfm[rfm['Segment'] == sel_seg].sort_values('Monetary', ascending=False).style.format({'Monetary': "Rp {:,.0f}"}), 
                    use_container_width=True, height=300
                )
        except:
            st.warning("Data belum cukup bervariasi untuk analisis RFM.")

    # --- TAB 3: STRATEGIS ---
    with tabs[3]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Product Mix")
            df_p = group_small_slices(df, 'TOC_DESCRIPTION', 'PREMIUM')
            df_p['CUM'] = 100 * df_p['PREMIUM'].cumsum() / df_p['PREMIUM'].sum()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_p['TOC_DESCRIPTION'], y=df_p['PREMIUM'], name='Premium', marker_color='#3b82f6'))
            fig.add_trace(go.Scatter(x=df_p['TOC_DESCRIPTION'], y=df_p['CUM'], name='Cum %', yaxis='y2', line=dict(color='#f59e0b')))
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110], showgrid=False))
            st.plotly_chart(make_clean(fig), use_container_width=True)
        with col_b:
            st.subheader("Monthly Trend")
            trend = df.groupby(['TAHUN', 'BULAN_NUM', 'BULAN_NAMA'])['PREMIUM'].sum().reset_index().sort_values(['TAHUN', 'BULAN_NUM'])
            fig_trend = px.line(trend, x='BULAN_NAMA', y='PREMIUM', color='TAHUN', markers=True, line_shape='spline')
            st.plotly_chart(make_clean(fig_trend), use_container_width=True)

    # --- TAB 4: PRODUK ---
    with tabs[4]:
        col_a, col_b = st.columns([2,1])
        with col_a:
            st.subheader("Portfolio Heatmap")
            fig_tree = px.treemap(df[df['PREMIUM']>0], path=[px.Constant("ACA"), 'SEGMENT', 'TOC_DESCRIPTION'], values='PREMIUM', color='SEGMENT', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_tree.update_traces(textinfo="label+percent parent")
            st.plotly_chart(make_clean(fig_tree), use_container_width=True)
        with col_b:
            st.subheader("Top Sales Leaderboard")
            top_sales = df.groupby('INPUT_NAME')['PREMIUM'].sum().reset_index().sort_values('PREMIUM', ascending=False).head(10)
            fig_bar = px.bar(top_sales, x='PREMIUM', y='INPUT_NAME', orientation='h', text_auto='.2s', color='PREMIUM', color_continuous_scale='Blues')
            fig_bar.update_layout(coloraxis_showscale=False)
            st.plotly_chart(make_clean(fig_bar), use_container_width=True)

    # --- TAB 5: OPERASIONAL ---
    with tabs[5]:
        col_a, col_b = st.columns([1,2])
        with col_a:
            st.subheader("SLA Compliance")
            fig_pie = px.pie(df, names='STATUS_SLA', hole=0.6, color_discrete_sequence=['#10b981', '#3b82f6', '#ef4444'])
            st.plotly_chart(make_clean(fig_pie), use_container_width=True)
        with col_b:
            st.subheader("Efficiency (Avg SLA Days)")
            sla_user = df.dropna(subset=['SLA_HARI']).groupby('INPUT_NAME')['SLA_HARI'].mean().reset_index().sort_values('SLA_HARI').head(10)
            fig_spd = px.bar(sla_user, x='SLA_HARI', y='INPUT_NAME', orientation='h', text_auto='.2f', color='SLA_HARI', color_continuous_scale='Mint')
            fig_spd.update_layout(coloraxis_showscale=False)
            st.plotly_chart(make_clean(fig_spd), use_container_width=True)

    # --- TAB 6: WAKTU ---
    with tabs[6]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Daily Volume")
            day_counts = df['HARI'].value_counts().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()
            fig_day = px.bar(day_counts, x='HARI', y='count', color='count', color_continuous_scale='Viridis')
            fig_day.update_layout(coloraxis_showscale=False)
            st.plotly_chart(make_clean(fig_day), use_container_width=True)
        with col_b:
            st.subheader("Peak Hour Heatmap")
            if 'JAM_INPUT' in df.columns:
                heat = df.groupby(['JAM_INPUT', 'HARI']).size().reset_index(name='Volume')
                fig_heat = px.density_heatmap(heat, x='JAM_INPUT', y='HARI', z='Volume', color_continuous_scale='Blues', nbinsx=24)
                fig_heat.update_layout(coloraxis_showscale=False)
                st.plotly_chart(make_clean(fig_heat), use_container_width=True)

    # --- TAB 7: PRODUKTIVITAS ---
    with tabs[7]:
        st.subheader("Staff Productivity Matrix")
        prod = df.groupby('INPUT_NAME').agg({'PREMIUM': 'sum', 'POLICYNO': 'count'}).reset_index()
        prod['RATA_RATA_POLIS'] = prod['PREMIUM'] / prod['POLICYNO']
        prod_viz = prod[prod['PREMIUM'] > 0]
        fig_prod = px.scatter(prod_viz, x='POLICYNO', y='RATA_RATA_POLIS', size='PREMIUM', color='PREMIUM', hover_name='INPUT_NAME', color_continuous_scale='Turbo')
        st.plotly_chart(make_clean(fig_prod), use_container_width=True)

    # --- TAB 8: RISK & RATE ---
    with tabs[8]:
        st.subheader("Pricing Discipline")
        df_risk = df[(df['PREMIUM'] > 0) & (df['TSI_VAL'] > 100_000_000) & (df['RATE_PCT'] < 5)]
        if not df_risk.empty:
            fig_risk = px.scatter(df_risk, x='TSI_VAL', y='RATE_PCT', color='TOC_DESCRIPTION', hover_data=['POLICYNO', 'INSURED_NAME'], log_x=True, opacity=0.7)
            st.plotly_chart(make_clean(fig_risk), use_container_width=True)
        
        st.subheader("Top 10 Clients")
        top_client = df.groupby('INSURED_NAME')['PREMIUM'].sum().reset_index().sort_values('PREMIUM', ascending=False).head(10)
        fig_cli = px.bar(top_client, x='PREMIUM', y='INSURED_NAME', orientation='h', color='PREMIUM', color_continuous_scale='Spectral')
        fig_cli.update_layout(coloraxis_showscale=False)
        st.plotly_chart(make_clean(fig_cli), use_container_width=True)

    # --- TAB 9: DATABASE ---
    with tabs[9]:
        st.subheader("Data Explorer")
        st.dataframe(df, use_container_width=True, height=600)

else:
    st.info("👋 Selamat Datang! Silakan masukkan file CSV ke dalam folder 'data_produksi' untuk memulai.") 