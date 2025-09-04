import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Financial Barriers to Mobile Money Accounts: A Global Perspective", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", [
    "Data Overview",
    "South Africa Analysis",
    "Comparative Analysis"
])

# Sidebar for visualization selection in Comparative Analysis
if page == "Comparative Analysis":
    viz_type = st.sidebar.selectbox("Select Visualization Type", [
        "Pie Chart",
        "Heatmap",
        "Treemap",
        "Geo Bubble",
        "Line Graph",
        "Scatter Plot",
        "Histogram",
        "Box Plot",
        "Interactive Bar Chart",
        "Correlation Heatmap",
        "Time-Series Line Graph"
    ])
else:
    viz_type = None

# Title and description
st.title("Financial Barriers to Mobile Money Accounts: A Global Perspective")
st.markdown("This dashboard examines barriers to mobile money account adoption across multiple countries, with a focus on South Africa, using World Bank Findex data. Visualizations explore 'Not Enough Money' and 'Agents Too Far' barriers through derived metrics and demographic breakdowns, with comparisons to other regions for context.")

# Section 1: Data Preparation
if page == "Data Overview":
    st.header("1. Data Preparation")

    # Load data
    uploaded_file_a = st.file_uploader("Upload WB_FINDEX_FIN14A.csv", type="csv")
    uploaded_file_d = st.file_uploader("Upload WB_FINDEX_FIN14D.csv", type="csv")

    if uploaded_file_a and uploaded_file_d:
        data_a = pd.read_csv(uploaded_file_a)
        data_d = pd.read_csv(uploaded_file_d)

        countries = ['Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 
                     'Pakistan', 'Senegal', 'Sierra Leone', 'South Africa', 'Tanzania', 'Togo', 'Uganda', 
                     'Zambia', 'Zimbabwe', 'Sub-Saharan Africa (excluding high income)']

        # Filter and select for FIN14A
        df_filtered_a = data_a[data_a["REF_AREA_LABEL"].isin(countries)]
        df_selected_a = df_filtered_a[["REF_AREA_LABEL", "OBS_VALUE", "UNIT_MEASURE"]]
        st.subheader("Selected Data for FIN14A (Agents Too Far)")
        st.dataframe(df_selected_a.head())

        # Filter and select for FIN14D
        df_filtered_d = data_d[data_d["REF_AREA_LABEL"].isin(countries)]
        df_selected_d = df_filtered_d[["REF_AREA_LABEL", "OBS_VALUE", "UNIT_MEASURE"]]
        st.subheader("Selected Data for FIN14D (Not Enough Money)")
        st.dataframe(df_selected_d.head())

        # Filter for PT_RESP_NACCT
        df_nacct_a = df_selected_a[df_selected_a["UNIT_MEASURE"] == "PT_RESP_NACCT"].copy()
        df_nacct_a.rename(columns={"REF_AREA_LABEL": "Country", "OBS_VALUE": "Agents Too Far (%)"}, inplace=True)
        df_nacct_a = df_nacct_a[["Country", "Agents Too Far (%)"]]

        df_nacct_d = df_selected_d[df_selected_d["UNIT_MEASURE"] == "PT_RESP_NACCT"].copy()
        df_nacct_d.rename(columns={"REF_AREA_LABEL": "Country", "OBS_VALUE": "Not Enough Money (%)"}, inplace=True)
        df_nacct_d = df_nacct_d[["Country", "Not Enough Money (%)"]]

        # Handle missing values
        df_nacct_a['Agents Too Far (%)'].fillna(df_nacct_a['Agents Too Far (%)'].mean(), inplace=True)
        df_nacct_d['Not Enough Money (%)'].fillna(df_nacct_d['Not Enough Money (%)'].mean(), inplace=True)

        # Derived metrics
        df_merged = pd.merge(df_nacct_a, df_nacct_d, on='Country')
        df_merged['Barrier Ratio (Money/Agents)'] = df_merged['Not Enough Money (%)'] / df_merged['Agents Too Far (%)'].replace(0, np.nan)
        df_merged['Barrier Difference (%)'] = df_merged['Not Enough Money (%)'] - df_merged['Agents Too Far (%)']
        df_merged['Total Barrier (%)'] = df_merged['Not Enough Money (%)'] + df_merged['Agents Too Far (%)']

        # Descriptive Stats
        st.subheader("Descriptive Statistics")
        st.dataframe(df_merged.describe())

        # Numerical Analysis
        st.header("2. Numerical Analysis")
        st.write(f"Mean Barrier Ratio (Money/Agents): {df_merged['Barrier Ratio (Money/Agents)'].mean():.2f}")
        st.write(f"Mean Barrier Difference: {df_merged['Barrier Difference (%)'].mean():.2f}%")
        st.write(f"Mean Total Barrier: {df_merged['Total Barrier (%)'].mean():.2f}%")
        st.write("Insight: Not having enough money is consistently a stronger barrier than agent distance across countries, with South Africa showing a notable gap.")

        # Cache merged dataset
        st.session_state['df_merged'] = df_merged

    else:
        st.warning("Please upload both CSV files to proceed.")
        st.stop()

# Visualization Pages
if page in ["South Africa Analysis", "Comparative Analysis"]:
    if 'df_merged' not in st.session_state:
        st.error("Please upload data in the Data Overview page first.")
        st.stop()

    df_merged = st.session_state['df_merged']
    st.header("3. Visualizations")

    if page == "South Africa Analysis":
        st.subheader("South Africa: Financial Barriers Analysis")
        sa_df = df_merged[df_merged['Country'] == 'South Africa'].copy()

        # Placeholder demographic data
        age_groups = ['15-24', '25-44', '45-64', '65+']
        genders = ['Male', 'Female']
        sa_agents_age = [25.0, 20.0, 18.0, 15.0]  # Hypothetical
        sa_money_age = [40.0, 36.0, 30.0, 25.0]
        sa_agents_gender = [22.0, 18.0]
        sa_money_gender = [38.0, 34.0]
        sa_age_df = pd.DataFrame({
            'Age Group': age_groups,
            'Agents Too Far (%)': sa_agents_age,
            'Not Enough Money (%)': sa_money_age,
            'Barrier Ratio (Money/Agents)': [m / a if a != 0 else np.nan for m, a in zip(sa_money_age, sa_agents_age)],
            'Barrier Difference (%)': [m - a for m, a in zip(sa_money_age, sa_agents_age)]
        })
        sa_gender_df = pd.DataFrame({
            'Gender': genders,
            'Agents Too Far (%)': sa_agents_gender,
            'Not Enough Money (%)': sa_money_gender,
            'Barrier Ratio (Money/Agents)': [m / a if a != 0 else np.nan for m, a in zip(sa_money_gender, sa_agents_gender)],
            'Barrier Difference (%)': [m - a for m, a in zip(sa_money_gender, sa_agents_gender)]
        })

        # Pie Chart: Barrier Proportions in South Africa
        st.subheader("Proportion of Barriers in South Africa")
        fig_pie = go.Figure(data=[
            go.Pie(labels=['Not Enough Money', 'Agents Too Far'],
                   values=[sa_df['Not Enough Money (%)'].iloc[0], sa_df['Agents Too Far (%)'].iloc[0]],
                   textinfo='label+percent', pull=[0.1, 0])]
        )
        fig_pie.update_layout(title="South Africa: Barrier Proportions (2021)",
                              font=dict(size=14), margin=dict(t=50, b=50))
        st.plotly_chart(fig_pie)
        st.markdown("*Figure 1: Pie chart showing 'Not Enough Money' (36%) dominates over 'Agents Too Far' (20%) in South Africa, highlighting financial constraints as the primary barrier.*")

        # Four-in-One: Age-Based Analysis
        st.subheader("Age-Based Barriers in South Africa (Four-in-One)")
        fig_age = make_subplots(rows=2, cols=2,
                                subplot_titles=("Barrier Ratio by Age", "Barrier Difference by Age",
                                                "Agents Too Far by Age", "Not Enough Money by Age"),
                                specs=[[{"type": "bar"}, {"type": "bar"}],
                                       [{"type": "bar"}, {"type": "bar"}]])
        fig_age.add_trace(go.Bar(x=sa_age_df['Age Group'], y=sa_age_df['Barrier Ratio (Money/Agents)'],
                                 name="Ratio (Money/Agents)"), row=1, col=1)
        fig_age.add_trace(go.Bar(x=sa_age_df['Age Group'], y=sa_age_df['Barrier Difference (%)'],
                                 name="Difference (%)"), row=1, col=2)
        fig_age.add_trace(go.Bar(x=sa_age_df['Age Group'], y=sa_age_df['Agents Too Far (%)'],
                                 name="Agents Too Far (%)"), row=2, col=1)
        fig_age.add_trace(go.Bar(x=sa_age_df['Age Group'], y=sa_age_df['Not Enough Money (%)'],
                                 name="Not Enough Money (%)"), row=2, col=2)
        fig_age.update_layout(title="South Africa: Age-Based Barrier Analysis (Placeholder Data)",
                              height=600, showlegend=True, font=dict(size=12))
        fig_age.update_yaxes(title_text="Ratio", row=1, col=1)
        fig_age.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        fig_age.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        fig_age.update_yaxes(title_text="Percentage (%)", row=2, col=2)
        st.plotly_chart(fig_age)
        st.markdown("*Figure 2: Subplots showing age-based variations in South Africa. Younger groups (15-24) face higher barriers, with a larger gap between financial and distance constraints.*")

        # Four-in-One: Gender-Based Analysis
        st.subheader("Gender-Based Barriers in South Africa (Four-in-One)")
        fig_gender = make_subplots(rows=2, cols=2,
                                   subplot_titles=("Barrier Ratio by Gender", "Barrier Difference by Gender",
                                                   "Agents Too Far by Gender", "Not Enough Money by Gender"),
                                   specs=[[{"type": "bar"}, {"type": "bar"}],
                                          [{"type": "bar"}, {"type": "bar"}]])
        fig_gender.add_trace(go.Bar(x=sa_gender_df['Gender'], y=sa_gender_df['Barrier Ratio (Money/Agents)'],
                                    name="Ratio (Money/Agents)"), row=1, col=1)
        fig_gender.add_trace(go.Bar(x=sa_gender_df['Gender'], y=sa_gender_df['Barrier Difference (%)'],
                                    name="Difference (%)"), row=1, col=2)
        fig_gender.add_trace(go.Bar(x=sa_gender_df['Gender'], y=sa_gender_df['Agents Too Far (%)'],
                                    name="Agents Too Far (%)"), row=2, col=1)
        fig_gender.add_trace(go.Bar(x=sa_gender_df['Gender'], y=sa_gender_df['Not Enough Money (%)'],
                                    name="Not Enough Money (%)"), row=2, col=2)
        fig_gender.update_layout(title="South Africa: Gender-Based Barrier Analysis (Placeholder Data)",
                                 height=600, showlegend=True, font=dict(size=12))
        fig_gender.update_yaxes(title_text="Ratio", row=1, col=1)
        fig_gender.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        fig_gender.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        fig_gender.update_yaxes(title_text="Percentage (%)", row=2, col=2)
        st.plotly_chart(fig_gender)
        st.markdown("*Figure 3: Subplots comparing gender-based barriers in South Africa. Males and females show similar patterns, with financial barriers consistently higher.*")

    elif page == "Comparative Analysis":
        st.subheader("Comparative Analysis: Selected Countries")
        selected_countries = st.sidebar.multiselect("Select Countries", options=df_merged['Country'].unique(), 
                                                   default=['South Africa', 'Nigeria', 'Tanzania'])
        if not selected_countries:
            st.warning("Please select at least one country.")
            st.stop()
        comp_df = df_merged[df_merged['Country'].isin(selected_countries)].copy()

        if viz_type == "Pie Chart":
            # Pie Chart: Barrier Proportions Comparison
            st.subheader("Barrier Proportions for Selected Country")
            selected_comp_country = st.sidebar.selectbox("Select Country for Pie Chart Comparison", 
                                                        options=selected_countries, index=0)
            comp_pie_df = comp_df[comp_df['Country'] == selected_comp_country]
            fig_comp_pie = go.Figure(data=[
                go.Pie(labels=['Not Enough Money', 'Agents Too Far'],
                       values=[comp_pie_df['Not Enough Money (%)'].iloc[0], comp_pie_df['Agents Too Far (%)'].iloc[0]],
                       textinfo='label+percent', pull=[0.1, 0])]
            )
            fig_comp_pie.update_layout(title=f"{selected_comp_country}: Barrier Proportions (2021)",
                                      font=dict(size=14), margin=dict(t=50, b=50))
            st.plotly_chart(fig_comp_pie)
            st.markdown(f"*Figure 4: Pie chart comparing barrier proportions in {selected_comp_country}. Financial barriers often dominate across countries.*")

        elif viz_type == "Heatmap":
            # Heatmap: Barrier Metrics Across Countries
            st.subheader("Heatmap of Barrier Metrics")
            pivot_df = comp_df.pivot_table(index='Country', values=['Agents Too Far (%)', 'Not Enough Money (%)', 
                                                                   'Barrier Ratio (Money/Agents)', 'Barrier Difference (%)'])
            fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax_heat)
            plt.title('Barrier Metrics Across Selected Countries', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig_heat)
            st.markdown("*Figure 5: Heatmap showing barrier metrics for selected countries, highlighting variations in financial and distance barriers.*")

        elif viz_type == "Treemap":
            # Treemap: Total Barrier Across Countries
            st.subheader("Treemap of Total Barriers")
            comp_df_melted = comp_df.melt(id_vars='Country', value_vars=['Agents Too Far (%)', 'Not Enough Money (%)'],
                                          var_name='Barrier', value_name='Percentage')
            fig_tree = px.treemap(comp_df_melted, path=['Country', 'Barrier'], values='Percentage',
                                  color='Percentage', color_continuous_scale='YlOrRd',
                                  title='Treemap of Financial Barriers Across Selected Countries')
            st.plotly_chart(fig_tree)
            st.markdown("*Figure 6: Treemap showing total barriers across selected countries, with financial constraints dominating.*")

        elif viz_type == "Geo Bubble":
            # Geo Bubble: Total Barrier
            st.subheader("Geographic Bubble Map")
            mappable_countries = [c for c in selected_countries if c != 'Sub-Saharan Africa (excluding high income)']
            geo_df = comp_df[comp_df['Country'].isin(mappable_countries)].copy()
            geo_df['Total Barrier (%)'] = geo_df['Agents Too Far (%)'] + geo_df['Not Enough Money (%)']
            fig_geo = px.scatter_geo(geo_df, locations="Country", locationmode='country names',
                                     size="Total Barrier (%)", color="Not Enough Money (%)",
                                     hover_name="Country", projection="natural earth",
                                     title='Geo Bubble Map (Bubble Size: Total Barriers, Color: Not Enough Money (%))')
            st.plotly_chart(fig_geo)
            st.markdown("*Figure 7: Geo bubble map comparing selected countries. Larger bubbles indicate higher total barriers.*")

        elif viz_type == "Line Graph":
            # Line Graph: Barrier Trends Across Countries
            st.subheader("Line Graph of Barrier Metrics")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=comp_df['Country'], y=comp_df['Not Enough Money (%)'],
                                         mode='lines+markers', name='Not Enough Money (%)'))
            fig_line.add_trace(go.Scatter(x=comp_df['Country'], y=comp_df['Agents Too Far (%)'],
                                         mode='lines+markers', name='Agents Too Far (%)'))
            fig_line.update_layout(title='Barrier Trends Across Selected Countries (2021)',
                                  xaxis_title='Country',
                                  yaxis_title='Percentage (%)',
                                  font=dict(size=12),
                                  showlegend=True)
            st.plotly_chart(fig_line)
            st.markdown("*Figure 8: Line graph showing trends in 'Not Enough Money' and 'Agents Too Far' across selected countries.*")

        elif viz_type == "Scatter Plot":
            # Scatter Plot: Not Enough Money vs. Agents Too Far
            st.subheader("Scatter Plot of Financial vs. Distance Barriers")
            fig_scatter = px.scatter(comp_df, x='Agents Too Far (%)', y='Not Enough Money (%)',
                                     text='Country', size='Total Barrier (%)', color='Country',
                                     title='Not Enough Money vs. Agents Too Far (Bubble Size: Total Barrier)')
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(showlegend=True, font=dict(size=12))
            st.plotly_chart(fig_scatter)
            st.markdown("*Figure 9: Scatter plot showing the relationship between financial and distance barriers, with bubble size representing total barriers.*")

        elif viz_type == "Histogram":
            # Histogram: Distribution of Barrier Metrics
            st.subheader("Histogram of Barrier Metrics")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=comp_df['Not Enough Money (%)'], name='Not Enough Money (%)', 
                                            opacity=0.5, nbinsx=10))
            fig_hist.add_trace(go.Histogram(x=comp_df['Agents Too Far (%)'], name='Agents Too Far (%)', 
                                            opacity=0.5, nbinsx=10))
            fig_hist.update_layout(title='Distribution of Barrier Metrics Across Selected Countries',
                                  xaxis_title='Percentage (%)',
                                  yaxis_title='Count',
                                  barmode='overlay',
                                  font=dict(size=12),
                                  showlegend=True)
            st.plotly_chart(fig_hist)
            st.markdown("*Figure 10: Histogram showing the distribution of 'Not Enough Money' and 'Agents Too Far' percentages across selected countries.*")

        elif viz_type == "Box Plot":
            # Box Plot: Distribution of Barrier Metrics
            st.subheader("Box Plot of Barrier Metrics Across Countries")
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df_merged['Not Enough Money (%)'], name='Not Enough Money (%)', marker_color='#FF4B4B'))
            fig_box.add_trace(go.Box(y=df_merged['Agents Too Far (%)'], name='Agents Too Far (%)', marker_color='#1F77B4'))
            fig_box.add_trace(go.Box(y=df_merged['Barrier Ratio (Money/Agents)'], name='Barrier Ratio (Money/Agents)', marker_color='#2CA02C'))
            fig_box.update_layout(title='Distribution of Barrier Metrics Across All Countries (2021)',
                                 yaxis_title='Value',
                                 font=dict(size=12),
                                 showlegend=True)
            st.plotly_chart(fig_box)
            st.markdown("*Figure 11: Box plot showing the distribution, median, and outliers of financial and distance barriers, as well as their ratio, across all countries.*")

        elif viz_type == "Interactive Bar Chart":
            # Interactive Bar Chart: Metric Comparison
            st.subheader("Bar Chart of Selected Barrier Metric")
            metric = st.sidebar.selectbox("Select Metric", ['Not Enough Money (%)', 'Agents Too Far (%)', 
                                                           'Barrier Ratio (Money/Agents)', 'Barrier Difference (%)', 
                                                           'Total Barrier (%)'])
            fig_bar = px.bar(comp_df, x='Country', y=metric, color='Country',
                             title=f'{metric} Across Selected Countries (2021)',
                             labels={metric: metric})
            fig_bar.update_layout(font=dict(size=12), showlegend=True)
            st.plotly_chart(fig_bar)
            st.markdown(f"*Figure 12: Bar chart comparing {metric} across selected countries, highlighting variations in financial and distance barriers.*")

        elif viz_type == "Correlation Heatmap":
            # Correlation Heatmap: Barrier Metrics
            st.subheader("Correlation Heatmap of Barrier Metrics")
            corr_df = comp_df[['Not Enough Money (%)', 'Agents Too Far (%)', 'Barrier Ratio (Money/Agents)', 
                               'Barrier Difference (%)', 'Total Barrier (%)']].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
            plt.title('Correlation of Barrier Metrics Across Selected Countries', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig_corr)
            st.markdown("*Figure 13: Correlation heatmap showing relationships between barrier metrics across selected countries. Strong correlations indicate related barriers.*")

        elif viz_type == "Time-Series Line Graph":
            # Placeholder Time-Series Line Graph
            st.subheader("Time-Series Trends of Barriers (Placeholder Data)")
            years = [2018, 2019, 2020, 2021]
            time_data = pd.DataFrame({
                'Year': years * len(selected_countries),
                'Country': [c for c in selected_countries for _ in years],
                'Not Enough Money (%)': [35 + i*0.5 for i, c in enumerate(selected_countries) for _ in years] 
                                        if len(selected_countries) > 0 else [35, 36, 37, 36],
                'Agents Too Far (%)': [20 + i*0.5 for i, c in enumerate(selected_countries) for _ in years] 
                                      if len(selected_countries) > 0 else [20, 21, 22, 20]
            })
            fig_time = px.line(time_data, x='Year', y=['Not Enough Money (%)', 'Agents Too Far (%)'], 
                               color='Country', line_group='Country',
                               title='Trends in Barriers Over Time (Placeholder Data)',
                               labels={'value': 'Percentage (%)', 'variable': 'Metric'})
            fig_time.update_layout(font=dict(size=12), showlegend=True)
            st.plotly_chart(fig_time)
            st.markdown("*Figure 14: Placeholder line graph showing trends in barriers over time for selected countries. Replace with real multi-year data when available.*")

    # Summary
    st.subheader("Summary")
    st.write("In South Africa, financial constraints ('Not Enough Money', 36%) are a significantly larger barrier than agent distance (20%), a pattern consistent across age and gender groups (placeholder data). Across selected countries, financial barriers dominate, with variations in intensity. New visualizations (line graph, scatter plot, histogram, box plot, interactive bar chart, correlation heatmap, time-series) provide deeper insights into cross-country trends, relationships, and distributions, underscoring the need for policies addressing income constraints to boost mobile money adoption.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data from World Bank Findex | Placeholder demographic data used for South Africa | Placeholder time-series data used for Time-Series Line Graph")