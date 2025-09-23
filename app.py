import os
import streamlit as st
import pandas as pd
from main import get_summary_and_sentiment, to_csv

st.set_page_config(
    page_title= '📞 Call Transcript Analyzer', 
    layout= 'wide'
)

# --- Sidebar ---
with st.sidebar:
    st.markdown('## 📞 Call Transcript Analyzer')
    st.markdown('---')

    # API Key input
    groq_api_key = st.text_input(
        'Enter your GROQ API key',
        key= 'groq_api_key',
        type= 'password',
        placeholder= 'gsk-...'
    )

    if groq_api_key:
        os.environ['GROQ_API_KEY'] = groq_api_key
        st.toast('✅ API Key set successfully')

    st.markdown('---')
    st.caption('[🔑 Get a Groq API key](https://console.groq.com/keys)')
    st.caption('[📂 Source Code](https://github.com/Harshit1234G/sush-assignment)')


# --- Title & Description ---
st.title('📞 Call Transcript Analyzer')
st.caption('Summarize transcripts and extract sentiment instantly using Groq LLM.')

st.markdown(
    """
    Paste a transcript of a customer call, and the AI will:  
    1. Generate a **2-3 sentence summary**  
    2. Detect the **customer sentiment** (Positive, Neutral, Negative)  

    Results are saved in `call_analysis.csv`.  
    """
)


# --- Transcript Input ---
transcript = st.text_area("📝 Enter Call Transcript", placeholder= "Paste the call transcript here...", height= 200)

if st.button("🔍 Analyze"):
    if not groq_api_key:
        st.info("⚠️ Please add your GROQ API key in the sidebar to continue.")
        st.stop()

    if not transcript.strip():
        st.warning("⚠️ Transcript cannot be empty.")
        st.stop()

    with st.spinner("Analyzing transcript..."):
        result = get_summary_and_sentiment(transcript)

    if result:
        st.success("✅ Analysis Complete")

        st.subheader("📄 Summary")
        st.write(result["summary"])

        st.subheader("😊 Sentiment")
        st.write(result["sentiment"])

        # Save to CSV
        to_csv(transcript, result)
        st.success("💾 Saved to `call_analysis.csv`")

        # Show CSV download
        df = pd.read_csv("call_analysis.csv")
        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="call_analysis.csv",
            mime="text/csv"
        )
