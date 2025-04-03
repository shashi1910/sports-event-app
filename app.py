import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import base64
from datetime import datetime
import os
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="The Sporting Pulse - CHRISPO '25",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3498DB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 0.5rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .dashboard-metrics {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        width: 23%;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar navigation
st.sidebar.image("https://img.freepik.com/free-vector/hand-drawn-flat-design-sports-logo-template_23-2149373179.jpg", width=100)
st.sidebar.title("CHRISPO '25")
st.sidebar.subheader("Inter-College Tournament")

# Navigation
page = st.sidebar.selectbox("Navigate To", 
                           ["Home Dashboard", "Data Explorer", 
                            "Feedback Analysis", "Image Gallery",
                            "Tournament Summary"])

# Helper Functions
@st.cache_data
def load_data():
    # Check if data exists, if not, generate it
    if not os.path.exists('chrispo25_data.csv'):
        # In a real app, we'd generate the data here, but we'll assume it's already generated
        st.error("Dataset not found. Please run the data generator script first.")
        return None
    
    data = pd.read_csv('chrispo25_data.csv')
    return data

def create_heatmap(df, x_col, y_col, value_col):
    # Create a cross-tabulation of the data
    heatmap_data = pd.crosstab(df[y_col], df[x_col], values=df[value_col], aggfunc='count')
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, ax=ax)
    
    plt.title(f"{y_col} vs {x_col} Participation", fontsize=16)
    plt.ylabel(y_col, fontsize=12)
    plt.xlabel(x_col, fontsize=12)
    
    return fig

def generate_wordcloud(text_data, title, colormap='viridis'):
    # Set up the word cloud parameters
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        colormap=colormap,
        max_font_size=150,
        random_state=42
    ).generate(text_data)
    
    # Create a figure and plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    return fig

def image_filters(image, filter_name):
    if filter_name == "Original":
        return image
    elif filter_name == "Grayscale":
        return ImageOps.grayscale(image)
    elif filter_name == "Sepia":
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = image.convert('RGB')
        sepia_img = np.array(sepia_img)
        sepia_img = sepia_img.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return Image.fromarray(sepia_img)
    elif filter_name == "Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_name == "Contour":
        return image.filter(ImageFilter.CONTOUR)
    elif filter_name == "Enhance":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    elif filter_name == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_name == "Emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif filter_name == "Edge Enhance":
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif filter_name == "Invert":
        return ImageOps.invert(image.convert('RGB'))

# Load data
data = load_data()

# Home Dashboard
if page == "Home Dashboard":
    st.markdown('<div class="main-header">The Sporting Pulse</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;">Comprehensive Analytics for CHRISPO \'25 Inter-College Tournament</div>', unsafe_allow_html=True)
    
    if data is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Tournament Overview</div>', unsafe_allow_html=True)
            
            # Key metrics
            total_participants = len(data)
            unique_colleges = data['College'].nunique()
            unique_states = data['State'].nunique()
            sports_count = data['Sport'].nunique()
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            metrics_col1.metric("Total Participants", f"{total_participants}")
            metrics_col2.metric("Colleges", f"{unique_colleges}")
            metrics_col3.metric("States", f"{unique_states}")
            metrics_col4.metric("Sports", f"{sports_count}")
            
            # Create tabs for different visualizations
            viz_tabs = st.tabs(["Sports Distribution", "Daily Participation", "Geographic Spread"])
            
            with viz_tabs[0]:
                st.subheader("Sport-wise Participation")
                
                sports_count = data['Sport'].value_counts().reset_index()
                sports_count.columns = ['Sport', 'Count']
                
                # Create interactive bar chart
                fig = px.bar(
                    sports_count, 
                    x='Sport', 
                    y='Count',
                    color='Count',
                    color_continuous_scale='Viridis',
                    template='plotly_white',
                    title='Participation by Sport'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tabs[1]:
                st.subheader("Day-wise Participation Trends")
                
                # Day-wise participation with time series line
                day_participation = data['Day'].value_counts().reset_index()
                day_participation.columns = ['Day', 'Count']
                day_participation['Day_Num'] = day_participation['Day'].str.extract('(\d+)').astype(int)
                day_participation = day_participation.sort_values('Day_Num')
                
                fig = px.line(
                    day_participation, 
                    x='Day', 
                    y='Count',
                    markers=True,
                    line_shape='spline',
                    template='plotly_white',
                    title='Daily Participation Trends'
                )
                
                fig.update_traces(line=dict(width=4), marker=dict(size=10))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tabs[2]:
                st.subheader("State-wise Participation")
                
                # State-wise participation choropleth would be ideal here
                # But using a simple horizontal bar chart for now
                state_counts = data['State'].value_counts().reset_index()
                state_counts.columns = ['State', 'Count']
                
                fig = px.bar(
                    state_counts.sort_values('Count'), 
                    y='State', 
                    x='Count',
                    orientation='h',
                    color='Count',
                    color_continuous_scale='Viridis',
                    template='plotly_white',
                    title='Participation by State'
                )
                
                fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="sub-header">Quick Stats</div>', unsafe_allow_html=True)
            
            # College leaderboard
            st.subheader("Top Participating Colleges")
            college_counts = data['College'].value_counts().head(5).reset_index()
            college_counts.columns = ['College', 'Participants']
            
            fig = px.bar(
                college_counts,
                x='Participants',
                y='College',
                orientation='h',
                color='Participants',
                color_continuous_scale='Blues',
                text='Participants'
            )
            
            fig.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance distribution
            st.subheader("Performance Distribution")
            performance_counts = data['Performance'].value_counts().reset_index()
            performance_counts.columns = ['Performance', 'Count']
            
            fig = px.pie(
                performance_counts,
                values='Count',
                names='Performance',
                color_discrete_sequence=px.colors.sequential.Plasma,
                hole=0.4
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent activity
            st.markdown('<div class="sub-header">Recent Participants</div>', unsafe_allow_html=True)
            st.dataframe(data[['Name', 'College', 'Sport', 'Day']].head(5), use_container_width=True)

# Data Explorer
elif page == "Data Explorer":
    st.markdown('<div class="main-header">Data Explorer</div>', unsafe_allow_html=True)
    
    if data is not None:
        st.markdown('<div class="sub-header">Interactive Data Filters</div>', unsafe_allow_html=True)
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_sports = st.multiselect("Select Sports", options=sorted(data['Sport'].unique()), default=None)
        
        with col2:
            selected_colleges = st.multiselect("Select Colleges", options=sorted(data['College'].unique()), default=None)
            
        with col3:
            selected_states = st.multiselect("Select States", options=sorted(data['State'].unique()), default=None)
        
        # Apply filters
        filtered_data = data.copy()
        
        if selected_sports:
            filtered_data = filtered_data[filtered_data['Sport'].isin(selected_sports)]
            
        if selected_colleges:
            filtered_data = filtered_data[filtered_data['College'].isin(selected_colleges)]
            
        if selected_states:
            filtered_data = filtered_data[filtered_data['State'].isin(selected_states)]
        
        # Show filtered data count
        st.write(f"Showing {len(filtered_data)} out of {len(data)} participants")
        
        # Create tabs for different analysis views
        tabs = st.tabs(["Participation Data", "Visual Analysis", "Cross-Tabulation"])
        
        with tabs[0]:
            # Allow searching and sorting
            search_term = st.text_input("Search by Name or College")
            
            if search_term:
                search_results = filtered_data[
                    filtered_data['Name'].str.contains(search_term, case=False) | 
                    filtered_data['College'].str.contains(search_term, case=False)
                ]
                st.dataframe(search_results, use_container_width=True)
            else:
                st.dataframe(filtered_data, use_container_width=True)
        
        with tabs[1]:
            chart_type = st.selectbox(
                "Select Chart Type", 
                ["Bar Chart", "Pie Chart", "Scatter Plot", "Heatmap", "Line Chart"]
            )
            
            if chart_type == "Bar Chart":
                bar_variable = st.selectbox("Select Variable", ["Sport", "College", "State", "Day", "Performance"])
                
                counts = filtered_data[bar_variable].value_counts().reset_index()
                counts.columns = [bar_variable, 'Count']
                
                # Sort by count
                counts = counts.sort_values('Count', ascending=False)
                
                fig = px.bar(
                    counts, 
                    x=bar_variable, 
                    y='Count',
                    color='Count',
                    title=f"Distribution by {bar_variable}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Pie Chart":
                pie_variable = st.selectbox("Select Variable", ["Sport", "College", "State", "Day", "Performance"])
                
                counts = filtered_data[pie_variable].value_counts().reset_index()
                counts.columns = [pie_variable, 'Count']
                
                # Limit to top 10 for readability if needed
                if len(counts) > 10 and pie_variable == "College":
                    counts = counts.head(10)
                    title = f"Top 10 {pie_variable} Distribution"
                else:
                    title = f"Distribution by {pie_variable}"
                
                fig = px.pie(
                    counts, 
                    values='Count', 
                    names=pie_variable,
                    title=title
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Scatter Plot":
                st.write("Showing Day vs Performance Level (marker size = count)")
                
                # Create a grouped dataframe for scatter plot
                scatter_df = filtered_data.groupby(['Day', 'Performance']).size().reset_index(name='Count')
                
                fig = px.scatter(
                    scatter_df,
                    x='Day',
                    y='Performance',
                    size='Count',
                    color='Performance',
                    title='Day vs Performance Distribution'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Heatmap":
                col1, col2 = st.columns(2)
                
                with col1:
                    heatmap_x = st.selectbox("Select X-axis", ["Sport", "Day", "State", "Performance"])
                
                with col2:
                    heatmap_y = st.selectbox("Select Y-axis", ["Day", "Sport", "State", "Performance"])
                
                if heatmap_x != heatmap_y:
                    fig = create_heatmap(filtered_data, heatmap_x, heatmap_y, 'Participant_ID')
                    st.pyplot(fig)
                else:
                    st.error("Please select different variables for X and Y axes")
                    
            elif chart_type == "Line Chart":
                # Time series data by day
                time_series = filtered_data.groupby(['Day', 'Sport']).size().reset_index(name='Count')
                
                fig = px.line(
                    time_series,
                    x='Day',
                    y='Count',
                    color='Sport',
                    markers=True,
                    title='Participation Trends by Day and Sport'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.subheader("Cross-Tabulation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                row_var = st.selectbox("Select Row Variable", ["Sport", "College", "State", "Day", "Performance"])
            
            with col2:
                col_var = st.selectbox("Select Column Variable", ["Day", "Sport", "State", "College", "Performance"])
            
            if row_var != col_var:
                # Create cross-tabulation
                crosstab = pd.crosstab(filtered_data[row_var], filtered_data[col_var])
                st.write(crosstab)
                
                # Visualize as heatmap
                st.subheader("Heatmap Visualization")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(crosstab, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, ax=ax)
                
                plt.title(f"{row_var} vs {col_var}", fontsize=16)
                plt.ylabel(row_var, fontsize=12)
                plt.xlabel(col_var, fontsize=12)
                
                st.pyplot(fig)
            else:
                st.error("Please select different variables for rows and columns")

# Feedback Analysis
elif page == "Feedback Analysis":
    st.markdown('<div class="main-header">Participant Feedback Analysis</div>', unsafe_allow_html=True)
    
    if data is not None:
        # Create tabs for different feedback analysis views
        feedback_tabs = st.tabs(["Sentiment Overview", "Sport-wise Feedback", "Word Clouds", "Text Explorer"])
        
        with feedback_tabs[0]:
            st.subheader("Feedback Snapshot")
            
            # Simplified sentiment analysis based on keywords
            def simple_sentiment(text):
                positive_words = ['great', 'excellent', 'amazing', 'good', 'wonderful', 
                                 'enjoyed', 'well-organized', 'fantastic', 'impressive']
                negative_words = ['disappointed', 'poor', 'bad', 'terrible', 'needs improvement',
                                 'frustrating', 'disorganized', 'inadequate']
                
                text = text.lower()
                
                pos_count = sum(1 for word in positive_words if word in text)
                neg_count = sum(1 for word in negative_words if word in text)
                
                if pos_count > neg_count:
                    return "Positive"
                elif neg_count > pos_count:
                    return "Negative"
                else:
                    return "Neutral"
            
            # Add sentiment column
            data['Sentiment'] = data['Feedback'].apply(simple_sentiment)
            
            # Show sentiment distribution
            sentiment_counts = data['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            # Create color mapping
            color_map = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
            sentiment_counts['Color'] = sentiment_counts['Sentiment'].map(color_map)
            
            fig = px.bar(
                sentiment_counts,
                x='Sentiment',
                y='Count',
                color='Sentiment',
                color_discrete_map=color_map,
                text='Count',
                title='Overall Sentiment Distribution'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show sentiment by sport
            sport_sentiment = pd.crosstab(data['Sport'], data['Sentiment']).reset_index()
            
            # Reshape for stacked visualization
            sport_sentiment_long = pd.melt(
                sport_sentiment, 
                id_vars=['Sport'], 
                value_vars=['Positive', 'Neutral', 'Negative'],
                var_name='Sentiment',
                value_name='Count'
            )
            
            fig = px.bar(
                sport_sentiment_long,
                x='Sport',
                y='Count',
                color='Sentiment',
                barmode='stack',
                color_discrete_map=color_map,
                title='Sentiment by Sport'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with feedback_tabs[1]:
            st.subheader("Sport-wise Feedback Analysis")
            
            selected_sport = st.selectbox("Select Sport", sorted(data['Sport'].unique()))
            
            # Filter data by selected sport
            sport_data = data[data['Sport'] == selected_sport]
            
            # Show sentiment distribution for selected sport
            sport_sentiment = sport_data['Sentiment'].value_counts().reset_index()
            sport_sentiment.columns = ['Sentiment', 'Count']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Create gauge chart for positive sentiment percentage
                positive_pct = (sport_data['Sentiment'] == 'Positive').mean() * 100
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = positive_pct,
                    title = {'text': "Positive Feedback %"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sport sentiment chart
                fig = px.pie(
                    sport_sentiment,
                    values='Count',
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map=color_map,
                    title=f'Sentiment Distribution for {selected_sport}'
                )
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show sample feedback
            st.subheader("Sample Feedback")
            
            feedback_sample = sport_data[['Name', 'College', 'Sentiment', 'Feedback']].sample(5)
            st.dataframe(feedback_sample, use_container_width=True)
            
            # Word frequency for selected sport
            st.subheader("Common Words in Feedback")
            
            all_feedback = " ".join(sport_data['Feedback'].tolist())
            wordcloud_fig = generate_wordcloud(all_feedback, f"{selected_sport} Feedback Word Cloud")
            st.pyplot(wordcloud_fig)
        
        with feedback_tabs[2]:
            st.subheader("Word Clouds by Sport")
            
            # Create a selection for word cloud colormap
            colormap = st.selectbox(
                "Select Color Theme",
                options=["viridis", "plasma", "inferno", "magma", "cividis", "blues", "reds", "greens"]
            )
            
            # Create columns for word clouds
            sports_list = sorted(data['Sport'].unique())
            
            # Display in 2 columns
            for i in range(0, len(sports_list), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(sports_list):
                        sport = sports_list[i]
                        sport_feedback = " ".join(data[data['Sport'] == sport]['Feedback'].tolist())
                        wc_fig = generate_wordcloud(sport_feedback, f"{sport} Feedback", colormap)
                        st.pyplot(wc_fig)
                
                with col2:
                    if i + 1 < len(sports_list):
                        sport = sports_list[i + 1]
                        sport_feedback = " ".join(data[data['Sport'] == sport]['Feedback'].tolist())
                        wc_fig = generate_wordcloud(sport_feedback, f"{sport} Feedback", colormap)
                        st.pyplot(wc_fig)
        
        with feedback_tabs[3]:
            st.subheader("Feedback Text Explorer")
            
            # Create filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_sport = st.selectbox("Filter by Sport", ["All"] + sorted(data['Sport'].unique()))
            
            with col2:
                filter_sentiment = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])
            
            with col3:
                filter_day = st.selectbox("Filter by Day", ["All"] + sorted(data['Day'].unique()))
            
            # Apply filters
            filtered_feedback = data.copy()
            
            if filter_sport != "All":
                filtered_feedback = filtered_feedback[filtered_feedback['Sport'] == filter_sport]
            
            if filter_sentiment != "All":
                filtered_feedback = filtered_feedback[filtered_feedback['Sentiment'] == filter_sentiment]
            
            if filter_day != "All":
                filtered_feedback = filtered_feedback[filtered_feedback['Day'] == filter_day]
            
            # Search in feedback
            search_feedback = st.text_input("Search in Feedback Text")
            
            if search_feedback:
                filtered_feedback = filtered_feedback[
                    filtered_feedback['Feedback'].str.contains(search_feedback, case=False)
                ]
            
            # Show filtered feedback
            st.write(f"Showing {len(filtered_feedback)} out of {len(data)} feedback entries")
            
            if not filtered_feedback.empty:
                st.dataframe(
                    filtered_feedback[['Name', 'College', 'Sport', 'Day', 'Sentiment', 'Feedback']], 
                    use_container_width=True
                )
                
                # Create a word cloud of filtered feedback
                if len(filtered_feedback) > 3:
                    st.subheader("Word Cloud of Filtered Feedback")
                    
                    filtered_text = " ".join(filtered_feedback['Feedback'].tolist())
                    wc_fig = generate_wordcloud(filtered_text, "Filtered Feedback Word Cloud", "plasma")
                    st.pyplot(wc_fig)
            else:
                st.info("No feedback matches your filters. Try adjusting your search criteria.")

# Modified section for the Image Gallery page
# This replaces the code in the Image Gallery section

# Modified section for the Image Gallery page
# This replaces the code in the Image Gallery section

# Image Gallery 
elif page == "Image Gallery":
    st.markdown('<div class="main-header">CHRISPO \'25 Image Gallery</div>', unsafe_allow_html=True)
    
    # In a real application, we would load actual event images
    # Here we'll use placeholder images and implement the processing functionality
    
    # Create tabs for different days
    day_tabs = st.tabs(["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"])
    
    # Sample image URLs from Unsplash (free, high-quality images)
    sample_images = {
        "Day 1": [
            "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",  # Basketball
            "https://images.unsplash.com/photo-1588492069485-d05b56b2831d?w=800&q=80",  # Volleyball
            "https://images.unsplash.com/photo-1517466787929-bc90951d0974?w=800&q=80"   # Football/Soccer
        ],
        "Day 2": [
            "https://plus.unsplash.com/premium_photo-1684820878202-52781d8e0ea9?q=80&w=2942&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",  # Tennis
            "https://images.unsplash.com/photo-1531415074968-036ba1b575da?w=800&q=80",  # Cricket
            "https://images.unsplash.com/photo-1519315901367-f34ff9154487?w=800&q=80"   # Swimming
        ],
        "Day 3": [
            "https://images.unsplash.com/photo-1517649763962-0c623066013b?w=800&q=80",  # Athletics
            "https://images.unsplash.com/photo-1547347298-4074fc3086f0?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",  # Table Tennis
            "https://images.unsplash.com/photo-1626224583764-f87db24ac4ea?w=800&q=80"   # Badminton
        ],
        "Day 4": [
            "https://images.unsplash.com/photo-1529699211952-734e80c4d42b?w=800&q=80",  # Chess
            "https://images.unsplash.com/photo-1574623452334-1e0ac2b3ccb4?w=800&q=80",  # Basketball action
            "https://images.unsplash.com/photo-1579952363873-27f3bade9f55?w=800&q=80"   # Football/Soccer action
        ],
        "Day 5": [
            "https://images.unsplash.com/photo-1521138054413-5a47d349b7af?w=800&q=80",  # Volleyball indoor
            "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800&q=80",  # Awards ceremony
            "https://images.unsplash.com/photo-1565958011703-44f9829ba187?w=800&q=80"   # Trophy
        ]
    }
    
    # Image processor
    def process_image(img_url, filter_name):
        # Download image
        try:
            import requests
            from PIL import Image
            import io
            
            response = requests.get(img_url)
            img = Image.open(io.BytesIO(response.content))
            
            # Apply the selected filter
            processed_img = image_filters(img, filter_name)
            
            return processed_img
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    
    # Display images for each day
    for i, tab in enumerate(day_tabs):
        day = f"Day {i+1}"
        
        with tab:
            st.markdown(f"<div class='sub-header'>Day {i+1} Highlights</div>", unsafe_allow_html=True)
            
            # Create image gallery
            col1, col2, col3 = st.columns(3)
            
            cols = [col1, col2, col3]
            
            for j, img_url in enumerate(sample_images[day]):
                with cols[j % 3]:
                    # Image container
                    st.image(img_url, caption=f"Event Image {j+1}", use_container_width=True)
            
            # Image processing section
            st.markdown("<div class='sub-header'>Image Processing Tool</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Select image - Add a unique key based on the day
                selected_img_idx = st.radio(
                    "Select Image to Process",
                    options=list(range(1, len(sample_images[day]) + 1)),
                    format_func=lambda x: f"Image {x}",
                    key=f"img_select_{day}"  # Add unique key parameter
                )
                
                # Select filter - Add a unique key based on the day
                selected_filter = st.selectbox(
                    "Select Filter",
                    options=["Original", "Grayscale", "Sepia", "Blur", "Contour", 
                             "Enhance", "Sharpen", "Emboss", "Edge Enhance", "Invert"],
                    key=f"filter_select_{day}"  # Add unique key parameter
                )
            
            with col2:
                # Display processed image
                selected_img_url = sample_images[day][selected_img_idx - 1]
                processed_img = process_image(selected_img_url, selected_filter)
                
                if processed_img:
                    st.image(processed_img, caption=f"{day} - Image {selected_img_idx} with {selected_filter} filter", use_container_width=True)
                    
                    # Add download button for processed image
                    buf = io.BytesIO()
                    processed_img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download Processed Image",
                        data=byte_im,
                        file_name=f"chrispo25_{day}_image{selected_img_idx}_{selected_filter}.png",
                        mime="image/png",
                        key=f"download_btn_{day}_{selected_img_idx}"  # Add unique key for download button
                    )




# Tournament Summary
elif page == "Tournament Summary":
    st.markdown('<div class="main-header">Tournament Summary</div>', unsafe_allow_html=True)
    
    if data is not None:
        # Create performance metrics with improved styling
        st.markdown('<div class="sub-header">Performance Dashboard</div>', unsafe_allow_html=True)
        
        # Create summary metrics in styled cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_participants = len(data)
            st.markdown(f"""
            <div style="background-color:#e8f4fd; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-size:1rem; color:#555;">Total Participants</div>
                <div style="font-size:2rem; font-weight:bold; color:#3498DB;">{total_participants:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_participants_per_day = len(data) / 5  # 5 days
            st.markdown(f"""
            <div style="background-color:#eafaf1; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-size:1rem; color:#555;">Avg Participants/Day</div>
                <div style="font-size:2rem; font-weight:bold; color:#2ECC71;">{avg_participants_per_day:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            most_popular_sport = data['Sport'].value_counts().idxmax()
            st.markdown(f"""
            <div style="background-color:#fef9e7; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-size:1rem; color:#555;">Most Popular Sport</div>
                <div style="font-size:2rem; font-weight:bold; color:#F1C40F;">{most_popular_sport}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Replace the sentiment calculation with a performance metric
            excellent_performance = (data['Performance'] == 'Excellent').mean() * 100
            st.markdown(f"""
            <div style="background-color:#f9ebeb; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-size:1rem; color:#555;">Excellent Performance</div>
                <div style="font-size:2rem; font-weight:bold; color:#E74C3C;">{excellent_performance:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create interactive summary visualization
        st.markdown('<div class="sub-header">Participation Summary</div>', unsafe_allow_html=True)
        
        viz_type = st.radio(
            "Select Visualization",
            options=["Bubble Chart", "Stacked Area Chart", "Sunburst", "Network"],
            horizontal=True,
            format_func=lambda x: f"📊 {x}" if x == "Bubble Chart" 
                        else f"📈 {x}" if x == "Stacked Area Chart"
                        else f"🔄 {x}" if x == "Sunburst"
                        else f"🕸️ {x}"
        )
        
        if viz_type == "Bubble Chart":
            # Create bubble chart of sports by day and college count
            bubble_data = data.groupby(['Day', 'Sport']).agg({
                'College': 'nunique',
                'Participant_ID': 'count'
            }).reset_index()
            
            bubble_data.columns = ['Day', 'Sport', 'Unique_Colleges', 'Participants']
            
            fig = px.scatter(
                bubble_data,
                x='Day',
                y='Sport',
                size='Participants',
                color='Unique_Colleges',
                hover_name='Sport',
                size_max=50,
                color_continuous_scale='Viridis',
                title='Participation Distribution by Day and Sport'
            )
            
            fig.update_layout(height=500)
            fig.update_traces(
                hovertemplate='<b>%{y}</b> on <b>%{x}</b><br>Participants: %{marker.size}<br>Colleges: %{marker.color}<extra></extra>'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Stacked Area Chart":
            # Create a stacked area chart showing participation over days by sport
            area_data = data.groupby(['Day', 'Sport']).size().reset_index(name='Count')
            
            # Make sure days are in correct order
            area_data['Day_Num'] = area_data['Day'].str.extract('(\d+)').astype(int)
            area_data = area_data.sort_values('Day_Num')
            
            fig = px.area(
                area_data,
                x='Day',
                y='Count',
                color='Sport',
                title='Participation Trends by Sport Over Days',
                template="plotly_white"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Sunburst":
            # Create a sunburst chart showing hierarchy of State -> College -> Sport
            st.write("Hierarchical view of participation from State to College to Sport")
            
            # Count participants for each path
            sunburst_data = data.groupby(['State', 'College', 'Sport']).size().reset_index(name='Count')
            
            fig = px.sunburst(
                sunburst_data,
                path=['State', 'College', 'Sport'],
                values='Count',
                color='Count',
                color_continuous_scale='Viridis',
                title='Hierarchical View of Participation',
                template="plotly_white"
            )
            
            fig.update_layout(height=700, margin=dict(t=30, l=0, r=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Network":
            # Create a simulated network visualization showing sport-college connections
            st.write("This visualization shows connections between sports and colleges based on participation counts")
            
            # Get counts for edges
            network_data = data.groupby(['Sport', 'College']).size().reset_index(name='Count')
            
            # Set a minimum threshold based on data
            min_count = max(2, network_data['Count'].quantile(0.7))
            
            # Filter for significant connections only (to reduce clutter)
            network_data = network_data[network_data['Count'] >= min_count]
            
            # Check if we have data
            if len(network_data) == 0:
                st.warning("Not enough significant connections found. Try adjusting the threshold.")
            else:
                # Create figure
                fig = go.Figure()
                
                # Add edges (connections)
                for _, row in network_data.iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row['Sport'], row['College']],
                            y=[0, 1],
                            mode='lines',
                            line=dict(width=row['Count'] / network_data['Count'].max() * 5, color='rgba(100,100,100,0.5)'),
                            hoverinfo='text',
                            hovertext=f"{row['Sport']} - {row['College']}: {row['Count']} participants",
                            showlegend=False
                        )
                    )
                
                # Add sport nodes
                fig.add_trace(
                    go.Scatter(
                        x=network_data['Sport'].unique(),
                        y=[0] * len(network_data['Sport'].unique()),
                        mode='markers+text',
                        marker=dict(size=15, color='#3498DB'),
                        text=network_data['Sport'].unique(),
                        textposition="bottom center",
                        name='Sports'
                    )
                )
                
                # Add college nodes
                fig.add_trace(
                    go.Scatter(
                        x=network_data['College'].unique(),
                        y=[1] * len(network_data['College'].unique()),
                        mode='markers+text',
                        marker=dict(size=10, color='#E74C3C'),
                        text=network_data['College'].unique(),
                        textposition="top center",
                        name='Colleges'
                    )
                )
                
                fig.update_layout(
                    title='Sport-College Connection Network',
                    showlegend=True,
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=50),
                    template="plotly_white",
                    plot_bgcolor='rgba(240,240,240,0.2)'
                )
                
                # Disable zoom and pan to keep layout fixed
                fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
                fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Add tournament performance analysis section
        st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
        
        # Create two columns layout for performance analysis
        perf_col1, perf_col2 = st.columns([3, 2])
        
        with perf_col1:
            # Performance by sport
            perf_sport = pd.crosstab(
                data['Sport'], 
                data['Performance'],
                normalize='index'
            ) * 100
            
            # Convert to long format for plotting
            perf_sport_long = perf_sport.reset_index().melt(
                id_vars=['Sport'],
                var_name='Performance',
                value_name='Percentage'
            )
            
            # Create a color mapping for performance
            perf_colors = {
                'Excellent': '#2ECC71',
                'Good': '#3498DB',
                'Average': '#F1C40F',
                'Below Average': '#E74C3C'
            }
            
            # Plot performance distribution by sport
            fig = px.bar(
                perf_sport_long,
                x='Sport',
                y='Percentage',
                color='Performance',
                color_discrete_map=perf_colors,
                title='Performance Distribution by Sport',
                template='plotly_white'
            )
            
            fig.update_layout(
                height=450,
                yaxis_title='Percentage (%)',
                xaxis_title='',
                legend_title='Performance Level',
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with perf_col2:
            # Overall performance distribution with nicer pie chart
            perf_counts = data['Performance'].value_counts().reset_index()
            perf_counts.columns = ['Performance', 'Count']
            
            # Calculate percentages
            perf_counts['Percentage'] = perf_counts['Count'] / perf_counts['Count'].sum() * 100
            
            fig = px.pie(
                perf_counts,
                values='Count',
                names='Performance',
                color='Performance',
                color_discrete_map=perf_colors,
                title='Overall Performance Distribution',
                hole=0.4
            )
            
            fig.update_layout(height=450)
            fig.update_traces(
                textinfo='percent+label',
                textposition='outside',
                textfont_size=12,
                pull=[0.05 if x == 'Excellent' else 0 for x in perf_counts['Performance']]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add day-wise performance trend
        st.subheader("Performance Trends by Day")
        
        # Create day-wise performance crosstab
        day_perf = pd.crosstab(data['Day'], data['Performance'])
        
        # Add day number for sorting
        day_perf['Day_Num'] = [int(day.split()[1]) for day in day_perf.index]
        day_perf = day_perf.sort_values('Day_Num').drop('Day_Num', axis=1)
        
        # Convert to long format for plotting
        day_perf_long = day_perf.reset_index().melt(
            id_vars=['Day'],
            var_name='Performance',
            value_name='Count'
        )
        
        # Add day number for proper ordering in the plot
        day_perf_long['Day_Num'] = day_perf_long['Day'].str.extract('(\d+)').astype(int)
        day_perf_long = day_perf_long.sort_values('Day_Num')
        
        # Create line chart
        fig = px.line(
            day_perf_long,
            x='Day',
            y='Count',
            color='Performance',
            markers=True,
            color_discrete_map=perf_colors,
            title='Performance Levels Across Tournament Days',
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            yaxis_title='Number of Participants',
            xaxis_title='',
            legend_title='Performance Level'
        )
        
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary report with improved styling
        st.markdown('<div class="sub-header">Tournament Highlights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top stats
            top_sport = data['Sport'].value_counts().idxmax()
            top_sport_count = data['Sport'].value_counts().max()
            top_college = data['College'].value_counts().idxmax()
            top_college_count = data['College'].value_counts().max()
            
            # Create a styled card for key findings
            st.markdown("""
            <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                <h3 style="color:#3498DB; border-bottom:2px solid #3498DB; padding-bottom:10px;">Key Findings</h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <ul style="list-style-type:none; padding:0;">
                    <li style="padding:8px 0; border-bottom:1px solid #eee;">
                        <span style="font-weight:bold;">Total participants:</span> {total_participants} across 5 days and 10 sports
                    </li>
                    <li style="padding:8px 0; border-bottom:1px solid #eee;">
                        <span style="font-weight:bold;">Most popular sport:</span> {top_sport} with {top_sport_count} participants
                    </li>
                    <li style="padding:8px 0; border-bottom:1px solid #eee;">
                        <span style="font-weight:bold;">Top participating college:</span> {top_college} with {top_college_count} participants
                    </li>
                    <li style="padding:8px 0; border-bottom:1px solid #eee;">
                        <span style="font-weight:bold;">Performance levels:</span> {', '.join(data['Performance'].unique())}
                    </li>
                    <li style="padding:8px 0;">
                        <span style="font-weight:bold;">Geographical reach:</span> Participants from {data['State'].nunique()} states
                    </li>
                </ul>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <h3 style="color:#3498DB; border-bottom:2px solid #3498DB; padding-bottom:10px; margin-top:20px;">Recommendations</h3>
                <ul style="list-style-type:none; padding:0;">
                    <li style="padding:8px 0; border-bottom:1px solid #eee;">
                        <span style="font-weight:bold;">📊</span> Focus marketing efforts on less represented sports for next year
                    </li>
                    <li style="padding:8px 0; border-bottom:1px solid #eee;">
                        <span style="font-weight:bold;">📅</span> Consider expanding event duration based on daily participation trends
                    </li>
                    <li style="padding:8px 0; border-bottom:1px solid #eee;">
                        <span style="font-weight:bold;">🏫</span> Target outreach to colleges with lower participation rates
                    </li>
                    <li style="padding:8px 0;">
                        <span style="font-weight:bold;">🔧</span> Improve facilities for sports with below average performance levels
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Download report option with improved styling
            st.markdown("""
            <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                <h3 style="color:#3498DB; border-bottom:2px solid #3498DB; padding-bottom:10px;">Generate Report</h3>
            </div>
            """, unsafe_allow_html=True)
            
            report_format = st.selectbox(
                "Select Report Format",
                options=["PDF", "Excel", "CSV"]
            )
            
            # Add options for report content
            report_sections = st.multiselect(
                "Select Report Sections",
                options=["Participant Statistics", "Performance Analysis", "Feedback Summary", "College Rankings", "State Distribution"],
                default=["Participant Statistics", "Performance Analysis"]
            )
            
            # Add a date selector for report date range
            report_date_range = st.date_input(
                "Select Date Range",
                value=[datetime.now(), datetime.now()],
                help="Select start and end dates for the report"
            )
            
            if st.button("Generate Tournament Report"):
                with st.spinner("Generating report..."):
                    # Simulate processing time
                    import time
                    time.sleep(1)
                    
                    st.success(f"Report generated successfully! Click below to download.")
                    
                    # In a real application, we would generate the actual report file
                    # Here we'll just show the download button
                    
                    st.download_button(
                        label=f"📥 Download {report_format} Report",
                        data=b"This would be the actual report file",
                        file_name=f"CHRISPO25_Tournament_Report.{report_format.lower()}",
                        mime="application/octet-stream"
                    )
    
    # Footer with improved styling
    st.markdown("""
    <div class="footer">
        <div style="margin-bottom:10px;">© 2025 CHRISPO '25 Analytics Team</div>
        <div style="display:flex; justify-content:center; gap:20px; font-size:0.7rem;">
            <span>Privacy Policy</span>
            <span>Terms of Service</span>
            <span>Contact Us</span>
        </div>
    </div>
    """, unsafe_allow_html=True)