import streamlit as st
import re
import fun
from fun import *


def process_candidate(transcript_input, jd_text):
    cleaned = fun.clean_transcript(transcript_input)
    final_cleaned_text=fun.final_clean_for_summary(cleaned)
    summary = fun.llm_recruiter_summary(final_cleaned_text)
    clean_summ=fun.clean_summary(summary)
    can_skills = fun.extract_skills(clean_summ)
    key_skills=fun.extract_skills(jd_text)
    match=fun.check_skills_in_summary(can_skills,key_skills)
    red_flag=fun.detect_red_flags(clean_summ)
    
    
    return clean_summ, can_skills,key_skills, match, red_flag


st.title("🧠 Interview Transcript Analyzer")
transcript_input = st.text_area("📝 Enter Transcript", height=200)
jd_input = st.text_area("📌 Enter Job Description Skills (comma-separated)", height=100)

if st.button("🔍 Analyze"):
    if transcript_input and jd_input:

        summary, candiate_skills,key_skills,match_status, red_flag_result = process_candidate(transcript_input, jd_input)
        
        st.subheader("📄 Summary")
        st.write(summary)
        
        st.subheader("🛠 Skills Identified")
        st.write(", ".join(candiate_skills) if candiate_skills else "No skills detected")
        
        st.subheader("📊 JD Match Result")
    
        if match_status == "Positive":
            st.success(f"✅ {match_status}")
        elif match_status == "Neutral":
            st.warning(f"⚠️ {match_status}")
        else:
            st.error(f"❌ {match_status}")

        st.subheader("🎯 Candidate vs JD Skills")
        st.write(f"**Candidate Skills:** {', '.join(candiate_skills)}")
        st.write(f"**JD Skills:** {', '.join(key_skills)}")

        st.subheader("🚩 Red Flags")
        st.write(red_flag_result)

    else:

        st.error("Please enter both Transcript and JD!")
