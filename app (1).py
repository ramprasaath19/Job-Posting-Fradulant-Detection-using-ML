import os
import re
import uuid
import base64
from datetime import datetime
from collections import Counter

import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Set page config
st.set_page_config(
    page_title="Job Posting Fraud Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Function to load the trained model
@st.cache_resource
def load_model():
    try:
        # Update paths to match the trained model files
        model_path = os.path.join(os.path.dirname(__file__), 'tuned_fraudulent_job_detector.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'onehot_encoder.pkl')

        # Use joblib to load the model instead of pickle
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        encoder = joblib.load(encoder_path)
        return model, vectorizer, encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Provide more helpful error message
        st.info("Make sure the model files exist in the correct location.")
        return None, None, None

# Initialize NLTK components
@st.cache_resource
def initialize_nltk():
    try:
        # Download necessary NLTK data
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"Error initializing NLTK: {e}")
        return None

# Suspicious keywords that might indicate fraudulent job postings
SUSPICIOUS_KEYWORDS = [
    "no experience necessary", "unlimited earning", "earn money fast", "work from home",
    "be your own boss", "no experience needed", "immediate start", "quick cash",
    "earn thousands", "money back guarantee", "processing fee", "credit card", "bank details",
    "ssn", "social security", "immediate openings", "urgent hiring", "huge income",
    "investment opportunity", "easy money", "free money", "paid daily", "no skills needed"
]

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Generate word cloud from text
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          max_words=100, contour_width=3, contour_color='steelblue').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# Function to download DataFrame as CSV
def download_csv(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Function to get sample job data
def get_sample_job_data(job_type):
    if job_type == "legitimate":
        return {
            "job_title": "Senior Software Engineer",
            "company_name": "Tech Innovations Inc.",
            "job_location": "San Francisco, CA",
            "employment_type": "Full-time",
            "required_experience": "Senior level",
            "required_education": "Bachelor's",
            "salary_range": "$120,000-$150,000",
            "industry": "Information Technology",
            "job_description": """We are seeking a talented Senior Software Engineer to join our development team.
            You'll work on challenging projects using the latest technologies to create innovative solutions.
            Required skills: Python, JavaScript, Docker, Kubernetes, and cloud platforms (AWS/Azure/GCP).
            Responsibilities include system design, code implementation, testing, and mentoring junior developers.""",
            "company_profile": "Tech Innovations Inc. is a leading software company with 500+ employees worldwide.",
            "benefits": "Health insurance, 401(k), flexible work hours, professional development budget",
            "has_questions": True,
            "telecommuting": True
        }
    else:  # fraudulent
        return {
            "job_title": "Earn the Income You Deserve",
            "company_name": "Not Specified",
            "job_location": "Anywhere",
            "employment_type": "Part-time",
            "required_experience": "Not Specified",
            "required_education": "Not Specified",
            "salary_range": "$10,000 weekly guaranteed!!!",
            "industry": "Business",
            "job_description": """IMMEDIATE START! Make $$$ from home doing simple tasks!!!
            NO EXPERIENCE NEEDED!! We will provide EVERYTHING you need to start making money TODAY.
            CONTACT US NOW to get started and receive your first payment within 24 HOURS!!
            We just need a small processing fee of $99 to get you started with our PROVEN SYSTEM!!""",
            "company_profile": "",
            "benefits": "BE YOUR OWN BOSS! Unlimited earning potential!",
            "has_questions": False,
            "telecommuting": False
        }

# Function to make prediction
def make_prediction(features, model, vectorizer, encoder=None):
    try:
        # Combine all text fields
        combined_text = features['combined_text']
        processed_text = preprocess_text(combined_text)

        # Get categorical and numerical features
        cat_features = {
            'location': features.get('job_location', 'Unknown'),
            'department': features.get('department', 'Unknown'),
            'employment_type': features.get('employment_type', 'Not Specified'),
            'required_experience': features.get('required_experience', 'Not Specified'),
            'required_education': features.get('required_education', 'Not Specified'),
            'industry': features.get('industry', 'Unknown'),
            'function': features.get('function', 'Unknown')
        }

        bin_features = {
            'telecommuting': features.get('telecommuting', 0),
            'has_company_logo': 0,
            'has_questions': features.get('has_questions', 0)
        }

        # Vectorize the text
        text_features = vectorizer.transform([processed_text])

        if encoder:
            # Prepare DataFrame for categorical features
            cat_df = pd.DataFrame([cat_features])

            # One-hot encode categorical features
            cat_encoded = encoder.transform(cat_df)

            # Prepare numerical features
            num_features = np.array([[
                bin_features['telecommuting'],
                bin_features['has_company_logo'],
                bin_features['has_questions']
            ]])
            num_features_sparse = csr_matrix(num_features)

            # Combine all features
            X = hstack([text_features, cat_encoded, num_features_sparse])
        else:
            # If encoder not available, just use text features
            X = text_features

        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0][1]

        # Get feature importance - basic approach for text features only
        words = vectorizer.get_feature_names_out()
        feature_vector = text_features.toarray()[0]

        # Get top words by TF-IDF score
        word_scores = [(word, feature_vector[i]) for i, word in enumerate(words) if feature_vector[i] > 0]
        top_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:10]

        return {
            'prediction': prediction,
            'probability': prediction_proba,
            'top_words': top_words
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to analyze text readability and sentiment
def analyze_text(text):
    if not text:
        return {}

    sia = initialize_nltk()
    results = {}

    # Readability scores
    results['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    results['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
    results['gunning_fog'] = textstat.gunning_fog(text)

    # Text statistics
    results['word_count'] = textstat.lexicon_count(text)
    results['sentence_count'] = textstat.sentence_count(text)

    # Detect suspicious keywords
    text_lower = text.lower()
    suspicious_terms = [term for term in SUSPICIOUS_KEYWORDS if term in text_lower]
    results['suspicious_terms'] = suspicious_terms
    results['suspicious_count'] = len(suspicious_terms)

    # Sentiment analysis
    if sia:
        sentiment = sia.polarity_scores(text)
        results['sentiment'] = sentiment

    # Word frequency
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words).most_common(20)
    results['word_frequency'] = word_freq

    return results

# Function to highlight suspicious terms in text
def highlight_suspicious_terms(text):
    if not text:
        return ""

    highlighted_text = text
    for term in SUSPICIOUS_KEYWORDS:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span style="background-color: #ffcccc; font-weight: bold;">{term}</span>', highlighted_text)

    return highlighted_text

# Function to clear all input fields
def clear_input_fields():
    # Clear text inputs
    for key in ["input_job_title", "input_company_name", "input_job_location",
                "input_salary_range", "input_industry", "input_job_description",
                "input_company_profile", "input_benefits"]:
        st.session_state[key] = ""

    # Reset select boxes to first option
    st.session_state["input_employment_type"] = "Full-time"
    st.session_state["input_required_experience"] = "Entry level"
    st.session_state["input_required_education"] = "High School"

    # Uncheck checkboxes
    st.session_state["has_questions"] = False
    st.session_state["telecommuting"] = False

# Main function
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        st.divider()

        # Load sample data
        st.subheader("Sample Data")
        sample_type = st.radio("Load sample job posting:",
                              ["None", "Legitimate Job", "Fraudulent Job"])

        if sample_type != "None" and st.button("Load Sample"):
            job_type = "legitimate" if sample_type == "Legitimate Job" else "fraudulent"
            sample_data = get_sample_job_data(job_type)

            # Update session state with sample data
            for key, value in sample_data.items():
                if key in ["has_questions", "telecommuting"]:
                    st.session_state[key] = value
                else:
                    st.session_state[f"input_{key}"] = value

            st.success(f"Loaded {job_type} job posting sample")
            st.rerun()

        st.divider()

        # Model info
        st.subheader("Model Information")
        try:
            model, _, _ = load_model()
            if model:
                st.success("✅ Model loaded successfully")
                st.info(f"Model type: {type(model).__name__}")
            else:
                st.warning("❌ Model not loaded")
        except:
            st.warning("❌ Error loading model")

        st.divider()

        # Session info
        st.subheader("Session Info")
        st.info(f"Session ID: {st.session_state.session_id[:8]}...")
        st.info(f"Predictions made: {len(st.session_state.history)}")

        if st.session_state.history and st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared!")
            st.rerun()

    # Main content
    st.title("📋 Job Posting Fraud Detector")
    st.write("""
    ## Is that job posting legitimate or fraudulent?
    Enter the details of the job posting to check if it might be fraudulent.
    """)

    # Create tabs for different functionalities with new "Advanced Tools" tab
    tabs = st.tabs(["Single Prediction", "Batch Upload", "Advanced Tool", "History", "Analytics"])

    # Single Prediction Tab
    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            # Job details input
            job_title = st.text_input("Job Title", key="input_job_title", value="")
            company_name = st.text_input("Company Name", key="input_company_name", value="")
            job_location = st.text_input("Location", key="input_job_location", value="")
            employment_type = st.selectbox(
                "Employment Type",
                ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"],
                key="input_employment_type"
            )

            required_experience = st.selectbox(
                "Required Experience",
                ["Entry level", "Mid level", "Senior level", "Executive", "Not Specified"],
                key="input_required_experience"
            )

            required_education = st.selectbox(
                "Required Education",
                ["High School", "Bachelor's", "Master's", "PhD", "Not Specified"],
                key="input_required_education"
            )

        with col2:
            # More job details
            salary_range = st.text_input("Salary Range (e.g., $50,000-$70,000)", key="input_salary_range", value="")
            industry = st.text_input("Industry", key="input_industry", value="")
            job_description = st.text_area("Job Description", key="input_job_description", height=150, value="")
            company_profile = st.text_area("Company Profile", key="input_company_profile", height=100, value="")
            benefits = st.text_input("Benefits", key="input_benefits", value="")
            has_questions = st.checkbox("Has Screening Questions", key="has_questions")
            telecommuting = st.checkbox("Allows Telecommuting", key="telecommuting")

        # Create two columns for the buttons
        button_col1, button_col2 = st.columns([1, 1])

        # Predict button
        with button_col1:
            predict_button = st.button("Check if Fraudulent", type="primary")

        # Clear fields button
        with button_col2:
            clear_button = st.button("Clear Fields", on_click=clear_input_fields)

        if predict_button:
            if job_title and job_description:
                try:
                    # Load model
                    model, vectorizer, encoder = load_model()

                    if model and vectorizer:
                        # Combine all text fields
                        combined_text = f"{job_title} {company_name} {job_location} {employment_type} {industry} {job_description} {company_profile} {benefits}"

                        # Create features
                        features = {
                            'combined_text': combined_text,
                            'has_company': 1 if company_name.strip() else 0,
                            'has_salary': 1 if salary_range.strip() else 0,
                            'has_questions': 1 if has_questions else 0,
                            'telecommuting': 1 if telecommuting else 0,
                            'employment_type': employment_type,
                            'required_experience': required_experience,
                            'required_education': required_education
                        }

                        # Make prediction
                        result = make_prediction(features, model, vectorizer, encoder)

                        if result:
                            prediction = result['prediction']
                            prediction_proba = result['probability']
                            top_words = result['top_words']

                            # Add to history
                            history_entry = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'job_title': job_title,
                                'company': company_name,
                                'prediction': "Fraudulent" if prediction == 1 else "Legitimate",
                                'confidence': prediction_proba if prediction == 1 else 1-prediction_proba,
                                'description': job_description[:100] + "..." if len(job_description) > 100 else job_description
                            }
                            st.session_state.history.append(history_entry)

                            # Display result
                            st.subheader("Prediction Result")

                            # Create two columns for results
                            result_col1, result_col2 = st.columns([3, 2])

                            with result_col1:
                                if prediction == 1:
                                    st.error(f"⚠️ This job posting appears to be FRAUDULENT (Confidence: {prediction_proba:.2%})")
                                    st.write("""
                                    ### Warning Signs:
                                    - Job seems too good to be true
                                    - Vague job description
                                    - Poor grammar or spelling
                                    - Requests for personal information or payment
                                    """)
                                else:
                                    st.success(f"✅ This job posting appears to be LEGITIMATE (Confidence: {(1-prediction_proba):.2%})")
                                    st.write("The posting shows characteristics consistent with legitimate job advertisements.")

                                # Display feature importance
                                st.subheader("Key Terms Detected")
                                if top_words:
                                    terms_df = pd.DataFrame(top_words, columns=["Term", "Importance"])
                                    terms_df["Importance"] = terms_df["Importance"].round(4)
                                    st.dataframe(terms_df)

                            with result_col2:
                                # Generate and display word cloud
                                st.subheader("Word Cloud")
                                cloud_fig = generate_word_cloud(preprocess_text(combined_text))
                                st.pyplot(cloud_fig)

                                # Confidence meter
                                st.subheader("Confidence Meter")
                                confidence = prediction_proba if prediction == 1 else (1-prediction_proba)
                                st.progress(float(confidence))

                                if confidence > 0.9:
                                    st.write("Very high confidence in this prediction")
                                elif confidence > 0.7:
                                    st.write("High confidence in this prediction")
                                elif confidence > 0.5:
                                    st.write("Moderate confidence in this prediction")
                                else:
                                    st.write("Low confidence - consider additional verification")
                    else:
                        st.warning("Model not available. Using placeholder prediction.")
                        st.info("This is a demo version. In a real application, the prediction would be made using a trained model.")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.info("This is a demo version. In a real application, the prediction would be made using a trained model.")
            else:
                st.warning("Please provide at least the job title and job description.")

    # Batch Upload Tab
    with tabs[1]:
        st.subheader("Batch Processing")
        st.write("Upload a CSV file with multiple job postings to analyze them all at once.")

        st.info("CSV file should contain columns for job_title, company_name, job_description, etc.")

        upload_col1, upload_col2 = st.columns(2)

        with upload_col1:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"File uploaded successfully! Found {len(df)} job postings.")
                    st.dataframe(df.head(3))

                    # Allow user to map their columns to the expected columns
                    st.subheader("Map Your Columns")
                    job_title_col = st.selectbox("Select column for Job Title", df.columns, index=0)
                    job_description_col = st.selectbox("Select column for Job Description", df.columns, index=1)
                    company_name_col = st.selectbox("Select column for Company Name", df.columns, index=2)
                    employment_type_col = st.selectbox("Select column for Employment Type", df.columns, index=3)
                    industry_col = st.selectbox("Select column for Industry", df.columns, index=4)
                    company_profile_col = st.selectbox("Select column for Company Profile", df.columns, index=5)
                    benefits_col = st.selectbox("Select column for Benefits", df.columns, index=6)

                    if st.button("Process Batch", type="primary"):
                        # Load model
                        model, vectorizer, encoder = load_model()

                        if model and vectorizer:
                            # Process each job posting
                            results = []
                            progress_bar = st.progress(0)

                            for i, row in df.iterrows():
                                # Extract data (handling missing values)
                                job_title = str(row.get(job_title_col, ''))
                                job_description = str(row.get(job_description_col, ''))
                                company_name = str(row.get(company_name_col, ''))
                                employment_type = str(row.get(employment_type_col, ''))
                                industry = str(row.get(industry_col, ''))
                                company_profile = str(row.get(company_profile_col, ''))
                                benefits = str(row.get(benefits_col, ''))

                                # Combine text fields
                                combined_text = f"{job_title} {company_name} {employment_type} {industry} {job_description} {company_profile} {benefits}"

                                # Create features
                                features = {
                                    'combined_text': combined_text,
                                    'has_company': 1 if pd.notna(row.get(company_name_col, '')) and row.get(company_name_col, '').strip() else 0,
                                    'has_salary': 1 if pd.notna(row.get('salary_range', '')) and row.get('salary_range', '').strip() else 0,
                                    'has_questions': 1 if row.get('has_questions', False) else 0,
                                    'telecommuting': 1 if row.get('telecommuting', False) else 0,
                                    'employment_type': str(row.get(employment_type_col, 'Not Specified')),
                                    'required_experience': str(row.get('required_experience', 'Not Specified')),
                                    'required_education': str(row.get('required_education', 'Not Specified'))
                                }

                                # Make prediction
                                result = make_prediction(features, model, vectorizer, encoder)

                                if result:
                                    # Create result dictionary for results table
                                    results.append({
                                        'job_title': job_title,
                                        'company_name': company_name,
                                        'prediction': "Fraudulent" if result['prediction'] == 1 else "Legitimate",
                                        'confidence': result['probability'] if result['prediction'] == 1 else 1-result['probability']
                                    })

                                    # Add to history for History and Analytics tabs
                                    history_entry = {
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'job_title': job_title,
                                        'company': company_name,
                                        'prediction': "Fraudulent" if result['prediction'] == 1 else "Legitimate",
                                        'confidence': result['probability'] if result['prediction'] == 1 else 1-result['probability'],
                                        'description': job_description[:100] + "..." if len(job_description) > 100 else job_description
                                    }
                                    st.session_state.history.append(history_entry)

                                # Update progress bar
                                progress_bar.progress((i + 1) / len(df))

                            # Display results
                            results_df = pd.DataFrame(results)
                            st.subheader("Batch Results")
                            st.dataframe(results_df)

                            # Display download link
                            st.markdown(download_csv(results_df, "fraud_detection_results.csv"), unsafe_allow_html=True)

                            # Show summary statistics
                            st.subheader("Summary")
                            fraudulent_count = sum(1 for r in results if r['prediction'] == "Fraudulent")
                            legitimate_count = sum(1 for r in results if r['prediction'] == "Legitimate")

                            summary_col1, summary_col2 = st.columns(2)

                            with summary_col1:
                                st.metric("Total Job Postings", len(results))
                                st.metric("Fraudulent Postings", fraudulent_count)
                                st.metric("Legitimate Postings", legitimate_count)

                            with summary_col2:
                                # Create pie chart
                                fig, ax = plt.subplots()
                                ax.pie([legitimate_count, fraudulent_count],
                                       labels=['Legitimate', 'Fraudulent'],
                                       autopct='%1.1f%%',
                                       colors=['#28a745', '#dc3545'])
                                ax.set_title('Distribution of Job Posting Types')
                                st.pyplot(fig)
                        else:
                            st.error("Failed to load model for batch processing")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

        with upload_col2:
            st.subheader("Sample Format")
            st.write("Your CSV file should follow this format:")
            sample_data = {
                "job_title": ["Software Developer", "DATA ENTRY WORK FROM HOME"],
                "job_description": ["We are seeking a skilled software developer...", "MAKE $5000/WEEK TYPING FROM HOME!!!"],
                "company_name": ["Tech Solutions Inc.", ""],
                "employment_type": ["Full-time", "Other"]
            }
            st.dataframe(pd.DataFrame(sample_data))

            st.download_button(
                label="Download Sample CSV",
                data=pd.DataFrame(sample_data).to_csv(index=False),
                file_name="sample_job_postings.csv",
                mime="text/csv"
            )

    # Advanced Tools Tab
    with tabs[2]:
        st.subheader("Advanced Analysis Tool")

        # Changed from multiple subtabs to just Text Analysis
        st.write("Analyze job descriptions for readability, sentiment, and suspicious content")
        text_to_analyze = st.text_area("Enter job description text", height=200)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analyze Text"):
                if text_to_analyze:
                    with st.spinner("Analyzing text..."):
                        analysis_results = analyze_text(text_to_analyze)

                        # Display readability metrics
                        st.subheader("Readability Metrics")
                        metrics_col1, metrics_col2 = st.columns(2)

                        with metrics_col1:
                            st.metric("Flesch Reading Ease", f"{analysis_results['flesch_reading_ease']:.1f}")
                            st.caption("Higher score = easier to read (90-100: Very Easy, 0-30: Very Difficult)")

                            st.metric("Word Count", analysis_results['word_count'])

                        with metrics_col2:
                            st.metric("Flesch-Kincaid Grade", f"{analysis_results['flesch_kincaid_grade']:.1f}")
                            st.caption("Indicates the US grade level needed to understand the text")

                            st.metric("Sentence Count", analysis_results['sentence_count'])

                        # Sentiment analysis
                        if 'sentiment' in analysis_results:
                            st.subheader("Sentiment Analysis")
                            sentiment = analysis_results['sentiment']

                            sent_col1, sent_col2, sent_col3 = st.columns(3)
                            with sent_col1:
                                st.metric("Positive", f"{sentiment['pos']:.2f}")
                            with sent_col2:
                                st.metric("Neutral", f"{sentiment['neu']:.2f}")
                            with sent_col3:
                                st.metric("Negative", f"{sentiment['neg']:.2f}")

                            # Compound score interpretation
                            compound = sentiment['compound']
                            if compound >= 0.05:
                                sentiment_text = "Positive"
                                sentiment_color = "green"
                            elif compound <= -0.05:
                                sentiment_text = "Negative"
                                sentiment_color = "red"
                            else:
                                sentiment_text = "Neutral"
                                sentiment_color = "gray"

                            st.markdown(f"Overall sentiment: <span style='color:{sentiment_color};font-weight:bold'>{sentiment_text}</span> ({compound:.2f})", unsafe_allow_html=True)

                        # Suspicious terms
                        st.subheader("Suspicious Terms Detection")
                        if analysis_results['suspicious_count'] > 0:
                            st.warning(f"Found {analysis_results['suspicious_count']} suspicious terms that may indicate a fraudulent posting")
                            for term in analysis_results['suspicious_terms']:
                                st.markdown(f"- {term}")
                        else:
                            st.success("No suspicious terms detected")

                        # Word frequency
                        st.subheader("Most Common Words")
                        word_freq_df = pd.DataFrame(analysis_results['word_frequency'], columns=['Word', 'Count'])
                        st.dataframe(word_freq_df)
                else:
                    st.warning("Please enter some text to analyze")

        with col2:
            st.subheader("Highlighted Suspicious Terms")
            if text_to_analyze:
                highlighted_text = highlight_suspicious_terms(text_to_analyze)
                st.markdown(highlighted_text, unsafe_allow_html=True)

    # History Tab with User Feedback
    with tabs[3]:
        st.subheader("Prediction History")

        if not st.session_state.history:
            st.info("No predictions have been made yet. Make some predictions to see them here.")
        else:
            history_df = pd.DataFrame(st.session_state.history)

            # Add filters
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                prediction_filter = st.multiselect(
                    "Filter by prediction:",
                    options=["Fraudulent", "Legitimate"],
                    default=["Fraudulent", "Legitimate"]
                )

            # Apply filters
            filtered_df = history_df[history_df['prediction'].isin(prediction_filter)]

            # Show history table
            st.dataframe(filtered_df, use_container_width=True)

            # Download history
            st.markdown(download_csv(history_df, "prediction_history.csv"), unsafe_allow_html=True)

            # User feedback system
            st.subheader("Model Feedback")
            st.write("Help improve the model by providing feedback on predictions")

            feedback_col1, feedback_col2 = st.columns(2)

            with feedback_col1:
                selected_job_index = st.selectbox(
                    "Select a job prediction to provide feedback on:",
                    range(len(st.session_state.history)),
                    format_func=lambda i: f"{st.session_state.history[i]['job_title']} ({st.session_state.history[i]['prediction']})"
                )

            with feedback_col2:
                feedback_options = ["Correct Prediction", "Incorrect - Actually Legitimate", "Incorrect - Actually Fraudulent"]
                user_feedback = st.radio("Your feedback:", feedback_options)

                if st.button("Submit Feedback"):
                    # In a real app, this would be stored in a database
                    st.session_state.history[selected_job_index]['user_feedback'] = user_feedback
                    st.success("Thank you for your feedback! This will help improve the model.")

                    # Show updated history
                    updated_history_df = pd.DataFrame(st.session_state.history)
                    st.dataframe(updated_history_df, use_container_width=True)

    # Analytics Tab
    with tabs[4]:
        st.subheader("Analytics Dashboard")

        if not st.session_state.history:
            st.info("No data available yet. Make some predictions to see analytics.")
        else:
            history_df = pd.DataFrame(st.session_state.history)

            # Create dashboard layout
            dash_col1, dash_col2 = st.columns(2)

            with dash_col1:
                # Prediction distribution
                st.subheader("Prediction Distribution")
                pred_counts = history_df['prediction'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%',
                       colors=['#28a745', '#dc3545'] if 'Legitimate' in pred_counts.index else ['#dc3545'])
                st.pyplot(fig)

            with dash_col2:
                # Confidence distribution
                st.subheader("Confidence Distribution")
                fig, ax = plt.subplots()
                sns.histplot(data=history_df, x='confidence', hue='prediction', bins=10, ax=ax)
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Count')
                st.pyplot(fig)

            # Trends over time (if enough data)
            if len(history_df) > 5:
                st.subheader("Prediction Trends")
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                history_df['date'] = history_df['timestamp'].dt.date

                daily_counts = history_df.groupby(['date', 'prediction']).size().unstack(fill_value=0)

                fig, ax = plt.subplots(figsize=(10, 5))
                daily_counts.plot(kind='bar', stacked=True, ax=ax)
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Predictions')
                ax.legend(title='Prediction')
                plt.tight_layout()
                st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
