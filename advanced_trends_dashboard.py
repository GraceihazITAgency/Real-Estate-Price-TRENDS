import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GRACE-IHAZ PROPERTIES & IT SOLUTIONS")
print("Advanced Price Trends & Market Analytics Dashboard")
print("="*80)

# Load dataset
print("\n[Loading Data...]")
try:
    df = pd.read_csv('grace_ihaz_property_dataset.csv')
    print(f"✓ Dataset loaded: {len(df)} properties")
except FileNotFoundError:
    print("ERROR: Dataset not found. Please run the dataset generator first.")
    exit()

# Data preprocessing
df['date_listed'] = pd.to_datetime(df['date_listed'])
df['listing_month'] = df['date_listed'].dt.to_period('M').astype(str)
df['listing_year'] = df['date_listed'].dt.year
df['listing_quarter'] = df['date_listed'].dt.to_period('Q').astype(str)
df['property_age'] = 2025 - df['year_built']
df['price_per_sqm'] = df['price_ngn'] / df['size_sqm']

# Separate Sale and Rent
df_sale = df[df['listing_type'] == 'Sale'].copy()
df_rent = df[df['listing_type'] == 'Rent'].copy()

# ============================================================================
# 1. INTERACTIVE PRICE TRENDS OVER TIME
# ============================================================================
print("\n[1/10] Creating Price Trends Over Time...")

# Monthly average price trends by state
monthly_state_prices = df_sale.groupby(['listing_month', 'state'])['price_ngn'].mean().reset_index()

fig1 = px.line(
    monthly_state_prices,
    x='listing_month',
    y='price_ngn',
    color='state',
    title='Property Price Trends Over Time by State',
    labels={'price_ngn': 'Average Price (₦)', 'listing_month': 'Month'},
    markers=True
)
fig1.update_layout(
    height=600,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig1.write_html('trend_1_price_over_time.html')
print("✓ Saved: trend_1_price_over_time.html")

# ============================================================================
# 2. PRICE GROWTH RATE ANALYSIS
# ============================================================================
print("[2/10] Creating Price Growth Rate Analysis...")

# Calculate quarterly growth rates
quarterly_prices = df_sale.groupby(['listing_quarter', 'state'])['price_ngn'].mean().reset_index()
quarterly_prices['growth_rate'] = quarterly_prices.groupby('state')['price_ngn'].pct_change() * 100

fig2 = px.bar(
    quarterly_prices.dropna(),
    x='listing_quarter',
    y='growth_rate',
    color='state',
    title='Quarterly Price Growth Rate by State (%)',
    labels={'growth_rate': 'Growth Rate (%)', 'listing_quarter': 'Quarter'},
    barmode='group'
)
fig2.update_layout(height=600)
fig2.write_html('trend_2_growth_rate.html')
print("✓ Saved: trend_2_growth_rate.html")

# ============================================================================
# 3. PRICE DISTRIBUTION EVOLUTION
# ============================================================================
print("[3/10] Creating Price Distribution Evolution...")

# Price distribution by year
fig3 = px.box(
    df_sale,
    x='listing_year',
    y='price_ngn',
    color='state',
    title='Price Distribution Evolution by Year',
    labels={'price_ngn': 'Price (₦)', 'listing_year': 'Year'}
)
fig3.update_layout(height=600)
fig3.write_html('trend_3_price_distribution.html')
print("✓ Saved: trend_3_price_distribution.html")

# ============================================================================
# 4. HEATMAP: AVERAGE PRICES BY STATE AND PROPERTY TYPE
# ============================================================================
print("[4/10] Creating State-Property Type Price Heatmap...")

state_type_pivot = df_sale.pivot_table(
    values='price_ngn',
    index='state',
    columns='property_type',
    aggfunc='mean'
).fillna(0)

fig4 = go.Figure(data=go.Heatmap(
    z=state_type_pivot.values / 1_000_000,
    x=state_type_pivot.columns,
    y=state_type_pivot.index,
    colorscale='Viridis',
    text=np.round(state_type_pivot.values / 1_000_000, 1),
    texttemplate='₦%{text}M',
    textfont={"size": 10},
    colorbar=dict(title="Price (₦M)")
))
fig4.update_layout(
    title='Average Property Prices: State vs Property Type Heatmap',
    xaxis_title='Property Type',
    yaxis_title='State',
    height=500
)
fig4.write_html('trend_4_state_type_heatmap.html')
print("✓ Saved: trend_4_state_type_heatmap.html")

# ============================================================================
# 5. PRICE PER SQUARE METER TRENDS
# ============================================================================
print("[5/10] Creating Price per SQM Trends...")

price_per_sqm_trend = df_sale.groupby(['listing_month', 'state'])['price_per_sqm'].mean().reset_index()

fig5 = px.area(
    price_per_sqm_trend,
    x='listing_month',
    y='price_per_sqm',
    color='state',
    title='Price per Square Meter Trends by State',
    labels={'price_per_sqm': 'Price per SQM (₦)', 'listing_month': 'Month'}
)
fig5.update_layout(height=600)
fig5.write_html('trend_5_price_per_sqm.html')
print("✓ Saved: trend_5_price_per_sqm.html")

# ============================================================================
# 6. CITY-LEVEL PRICE TRENDS (TOP 15 CITIES)
# ============================================================================
print("[6/10] Creating City-Level Price Trends...")

top_cities = df_sale.groupby('city')['price_ngn'].mean().nlargest(15).index
city_trend_data = df_sale[df_sale['city'].isin(top_cities)].groupby(['listing_month', 'city'])['price_ngn'].mean().reset_index()

fig6 = px.line(
    city_trend_data,
    x='listing_month',
    y='price_ngn',
    color='city',
    title='Price Trends: Top 15 Most Expensive Cities',
    labels={'price_ngn': 'Average Price (₦)', 'listing_month': 'Month'},
    markers=True
)
fig6.update_layout(height=700)
fig6.write_html('trend_6_city_trends.html')
print("✓ Saved: trend_6_city_trends.html")

# ============================================================================
# 7. PROPERTY AGE VS PRICE TREND ANALYSIS
# ============================================================================
print("[7/10] Creating Property Age vs Price Analysis...")

age_bins = [0, 2, 5, 10, 20]
age_labels = ['0-2 years', '3-5 years', '6-10 years', '10+ years']
df_sale['age_group'] = pd.cut(df_sale['property_age'], bins=age_bins, labels=age_labels)

age_price_trend = df_sale.groupby(['listing_month', 'age_group'])['price_ngn'].mean().reset_index()

fig7 = px.line(
    age_price_trend,
    x='listing_month',
    y='price_ngn',
    color='age_group',
    title='Price Trends by Property Age Group',
    labels={'price_ngn': 'Average Price (₦)', 'listing_month': 'Month', 'age_group': 'Age Group'},
    markers=True
)
fig7.update_layout(height=600)
fig7.write_html('trend_7_age_trends.html')
print("✓ Saved: trend_7_age_trends.html")

# ============================================================================
# 8. BEDROOM COUNT PRICE TRENDS
# ============================================================================
print("[8/10] Creating Bedroom Count Price Trends...")

bedroom_trend = df_sale.groupby(['listing_month', 'bedrooms'])['price_ngn'].mean().reset_index()

fig8 = px.line(
    bedroom_trend,
    x='listing_month',
    y='price_ngn',
    color='bedrooms',
    title='Price Trends by Number of Bedrooms',
    labels={'price_ngn': 'Average Price (₦)', 'listing_month': 'Month', 'bedrooms': 'Bedrooms'},
    markers=True
)
fig8.update_layout(height=600)
fig8.write_html('trend_8_bedroom_trends.html')
print("✓ Saved: trend_8_bedroom_trends.html")

# ============================================================================
# 9. MARKET ACTIVITY: LISTING VOLUME TRENDS
# ============================================================================
print("[9/10] Creating Market Activity Analysis...")

listing_volume = df.groupby(['listing_month', 'listing_type']).size().reset_index(name='count')

fig9 = go.Figure()

for listing_type in listing_volume['listing_type'].unique():
    type_data = listing_volume[listing_volume['listing_type'] == listing_type]
    fig9.add_trace(go.Scatter(
        x=type_data['listing_month'],
        y=type_data['count'],
        mode='lines+markers',
        name=listing_type,
        fill='tonexty' if listing_type == 'Rent' else None
    ))

fig9.update_layout(
    title='Market Activity: Property Listing Volume Over Time',
    xaxis_title='Month',
    yaxis_title='Number of Listings',
    height=600,
    hovermode='x unified'
)
fig9.write_html('trend_9_market_activity.html')
print("✓ Saved: trend_9_market_activity.html")

# ============================================================================
# 10. COMPREHENSIVE DASHBOARD: MULTI-METRIC TRENDS
# ============================================================================
print("[10/10] Creating Comprehensive Multi-Metric Dashboard...")

# Create subplots
fig10 = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Average Price Trend',
        'Price per SQM Trend',
        'Market Volume',
        'Price Distribution',
        'State Comparison',
        'Property Type Distribution'
    ),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "bar"}, {"type": "box"}],
        [{"type": "bar"}, {"type": "pie"}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. Average price trend
monthly_avg = df_sale.groupby('listing_month')['price_ngn'].mean().reset_index()
fig10.add_trace(
    go.Scatter(x=monthly_avg['listing_month'], y=monthly_avg['price_ngn'],
               mode='lines+markers', name='Avg Price', line=dict(color='blue')),
    row=1, col=1
)

# 2. Price per SQM trend
monthly_sqm = df_sale.groupby('listing_month')['price_per_sqm'].mean().reset_index()
fig10.add_trace(
    go.Scatter(x=monthly_sqm['listing_month'], y=monthly_sqm['price_per_sqm'],
               mode='lines+markers', name='Price/SQM', line=dict(color='green')),
    row=1, col=2
)

# 3. Market volume
volume_data = df.groupby('listing_month').size().reset_index(name='count')
fig10.add_trace(
    go.Bar(x=volume_data['listing_month'], y=volume_data['count'],
           name='Volume', marker_color='orange'),
    row=2, col=1
)

# 4. Price distribution (recent 3 months)
recent_months = df_sale['listing_month'].unique()[-3:]
recent_data = df_sale[df_sale['listing_month'].isin(recent_months)]
for month in recent_months:
    month_data = recent_data[recent_data['listing_month'] == month]
    fig10.add_trace(
        go.Box(y=month_data['price_ngn'], name=month),
        row=2, col=2
    )

# 5. State comparison (current average)
state_avg = df_sale.groupby('state')['price_ngn'].mean().sort_values(ascending=False)
fig10.add_trace(
    go.Bar(x=state_avg.index, y=state_avg.values,
           name='State Avg', marker_color='purple'),
    row=3, col=1
)

# 6. Property type distribution
type_counts = df['property_type'].value_counts()
fig10.add_trace(
    go.Pie(labels=type_counts.index, values=type_counts.values, name='Type'),
    row=3, col=2
)

fig10.update_layout(
    title_text="Grace-Ihaz Properties: Comprehensive Market Dashboard",
    showlegend=False,
    height=1400
)

fig10.write_html('trend_10_comprehensive_dashboard.html')
print("✓ Saved: trend_10_comprehensive_dashboard.html")

# ============================================================================
# GENERATE STATIC MATPLOTLIB DASHBOARD (HIGH-RES PNG)
# ============================================================================
print("\n[BONUS] Creating Static High-Resolution Dashboard...")

fig_static = plt.figure(figsize=(24, 16))
gs = fig_static.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# 1. Price trends by state
ax1 = fig_static.add_subplot(gs[0, :])
for state in df_sale['state'].unique():
    state_data = df_sale[df_sale['state'] == state].groupby('listing_month')['price_ngn'].mean()
    ax1.plot(range(len(state_data)), state_data.values / 1_000_000, marker='o', label=state, linewidth=2)
ax1.set_title('Property Price Trends by State', fontsize=16, fontweight='bold')
ax1.set_xlabel('Time Period', fontsize=12)
ax1.set_ylabel('Average Price (₦ Millions)', fontsize=12)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(alpha=0.3)

# 2. State comparison
ax2 = fig_static.add_subplot(gs[1, 0])
state_avg = df_sale.groupby('state')['price_ngn'].mean().sort_values(ascending=False)
colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(state_avg)))
ax2.barh(range(len(state_avg)), state_avg.values / 1_000_000, color=colors_bar)
ax2.set_yticks(range(len(state_avg)))
ax2.set_yticklabels(state_avg.index)
ax2.set_xlabel('Average Price (₦ Millions)', fontsize=11)
ax2.set_title('Average Price by State', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Property type distribution
ax3 = fig_static.add_subplot(gs[1, 1])
type_counts = df['property_type'].value_counts()
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
wedges, texts, autotexts = ax3.pie(type_counts.values, labels=type_counts.index,
                                     autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax3.set_title('Property Type Distribution', fontsize=13, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 4. Price per SQM by state
ax4 = fig_static.add_subplot(gs[1, 2])
price_sqm_state = df_sale.groupby('state')['price_per_sqm'].mean().sort_values(ascending=False)
ax4.bar(range(len(price_sqm_state)), price_sqm_state.values / 1000, color='coral')
ax4.set_xticks(range(len(price_sqm_state)))
ax4.set_xticklabels(price_sqm_state.index, rotation=45, ha='right')
ax4.set_ylabel('Price per SQM (₦ Thousands)', fontsize=11)
ax4.set_title('Price per SQM by State', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Bedroom price trends
ax5 = fig_static.add_subplot(gs[2, 0])
bedroom_avg = df_sale.groupby('bedrooms')['price_ngn'].mean()
ax5.plot(bedroom_avg.index, bedroom_avg.values / 1_000_000, marker='o', 
         linewidth=3, markersize=10, color='darkblue')
ax5.set_xlabel('Number of Bedrooms', fontsize=11)
ax5.set_ylabel('Average Price (₦ Millions)', fontsize=11)
ax5.set_title('Price by Bedroom Count', fontsize=13, fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Condition impact
ax6 = fig_static.add_subplot(gs[2, 1])
condition_order = ['New', 'Renovated', 'Fair', 'Needs Repair']
condition_avg = df_sale.groupby('condition')['price_ngn'].mean().reindex(condition_order)
colors_cond = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
bars = ax6.bar(range(len(condition_avg)), condition_avg.values / 1_000_000, color=colors_cond)
ax6.set_xticks(range(len(condition_avg)))
ax6.set_xticklabels(condition_order, rotation=45, ha='right')
ax6.set_ylabel('Average Price (₦ Millions)', fontsize=11)
ax6.set_title('Price by Condition', fontsize=13, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

# 7. Market volume over time
ax7 = fig_static.add_subplot(gs[2, 2])
monthly_volume = df.groupby('listing_month').size()
ax7.fill_between(range(len(monthly_volume)), monthly_volume.values, alpha=0.5, color='purple')
ax7.plot(range(len(monthly_volume)), monthly_volume.values, color='darkviolet', linewidth=2)
ax7.set_xlabel('Time Period', fontsize=11)
ax7.set_ylabel('Number of Listings', fontsize=11)
ax7.set_title('Market Activity Over Time', fontsize=13, fontweight='bold')
ax7.grid(alpha=0.3)

# 8. Amenities comparison
ax8 = fig_static.add_subplot(gs[3, :])
amenities_impact = pd.DataFrame({
    'Feature': ['Serviced: No', 'Serviced: Yes', 'Gated: No', 'Gated: Yes'],
    'Price': [
        df_sale[df_sale['serviced'] == 0]['price_ngn'].mean(),
        df_sale[df_sale['serviced'] == 1]['price_ngn'].mean(),
        df_sale[df_sale['gated_estate'] == 0]['price_ngn'].mean(),
        df_sale[df_sale['gated_estate'] == 1]['price_ngn'].mean()
    ]
})
colors_amenity = ['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71']
bars = ax8.bar(amenities_impact['Feature'], amenities_impact['Price'] / 1_000_000, color=colors_amenity)
ax8.set_ylabel('Average Price (₦ Millions)', fontsize=12)
ax8.set_title('Impact of Amenities on Property Price', fontsize=14, fontweight='bold')
ax8.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, amenities_impact['Price'])):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
             f'₦{val/1_000_000:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('GRACE-IHAZ PROPERTIES: COMPREHENSIVE MARKET ANALYTICS DASHBOARD',
             fontsize=20, fontweight='bold', y=0.995)

plt.savefig('comprehensive_trends_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comprehensive_trends_dashboard.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("PRICE TRENDS ANALYSIS SUMMARY")
print("="*80)

print("\n1. OVERALL MARKET TRENDS:")
print("-" * 80)
first_month_avg = df_sale.groupby('listing_month')['price_ngn'].mean().iloc[0]
last_month_avg = df_sale.groupby('listing_month')['price_ngn'].mean().iloc[-1]
overall_growth = ((last_month_avg - first_month_avg) / first_month_avg) * 100
print(f"First Month Average: ₦{first_month_avg:,.0f}")
print(f"Latest Month Average: ₦{last_month_avg:,.0f}")
print(f"Overall Growth: {overall_growth:+.2f}%")

print("\n2. STATE GROWTH RATES:")
print("-" * 80)
for state in df_sale['state'].unique():
    state_data = df_sale[df_sale['state'] == state].groupby('listing_month')['price_ngn'].mean()
    if len(state_data) > 1:
        state_growth = ((state_data.iloc[-1] - state_data.iloc[0]) / state_data.iloc[0]) * 100
        print(f"{state:12s}: {state_growth:+.2f}%")

print("\n3. PRICE PER SQM TRENDS:")
print("-" * 80)
sqm_by_state = df_sale.groupby('state')['price_per_sqm'].mean().sort_values(ascending=False)
for state, price in sqm_by_state.items():
    print(f"{state:12s}: ₦{price:,.0f} per sqm")

print("\n4. MOST APPRECIATING PROPERTY TYPES:")
print("-" * 80)
for prop_type in df_sale['property_type'].unique()[:5]:
    type_data = df_sale[df_sale['property_type'] == prop_type].groupby('listing_month')['price_ngn'].mean()
    if len(type_data) > 1:
        type_growth = ((type_data.iloc[-1] - type_data.iloc[0]) / type_data.iloc[0]) * 100
        print(f"{prop_type:25s}: {type_growth:+.2f}%")

print("\n5. FILES GENERATED:")
print("-" * 80)
files = [
    'trend_1_price_over_time.html',
    'trend_2_growth_rate.html',
    'trend_3_price_distribution.html',
    'trend_4_state_type_heatmap.html',
    'trend_5_price_per_sqm.html',
    'trend_6_city_trends.html',
    'trend_7_age_trends.html',
    'trend_8_bedroom_trends.html',
    'trend_9_market_activity.html',
    'trend_10_comprehensive_dashboard.html',
    'comprehensive_trends_dashboard.png'
]
for i, file in enumerate(files, 1):
    print(f"{i:2d}. {file}")

print("\n" + "="*80)
print("TREND ANALYSIS COMPLETE!")
print("="*80)
print("\nOpen the HTML files in your browser for interactive exploration!")
print("View the PNG file for high-resolution static dashboard!")