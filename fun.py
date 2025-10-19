import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PreTrainedTokenizerFast
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings('ignore')

model_name = "philschmid/bart-large-cnn-samsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def clean_transcript(text):
    text = str(text) if pd.notna(text) else ""
    if not text:
        return ""

    
    text = re.sub(r"\b[A-Za-z ]{2,15}:\s*", "", text)

    
    fillers = [
        "Sure, well,", "for sharing that", "Good to know", "That's helpful",
        "Could you elaborate on that", "How did your team respond to your decision",
        "Let's start with some questions", "To begin", "Great question", "Tell me about",
        "for this opportunity", "Alright", "Got it", "for joining", "let's begin",
        "Do you have", "anything to ask us", "questions for us"
    ]
    for f in fillers:
        text = re.sub(rf"{re.escape(f)}[., ]*", "", text, flags=re.IGNORECASE)

    
    greetings = [
        "hello", "good morning", "good afternoon", "good evening",
        "thank you", "thanks", "nice to meet you", "hope you are doing well",
        "pleasure to be here", "thank you for having me", "you're welcome"
    ]
    for g in greetings:
        text = re.sub(rf"\b{g}\b", "", text, flags=re.IGNORECASE)

    
    text = re.sub(r"\b(What|Why|How|Tell me|Can you|Could you|Do you)\b.*?\?", "", text, flags=re.IGNORECASE)

    
    text = re.sub(r"[!?,.]{2,}", ". ", text)
    text = re.sub(r"\.{2,}", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def final_clean_for_summary(text):

    text = re.sub(r"\b(\w+)-\1\b", r"\1", text)
    text = re.sub(r"(you know|uh|sure|well|great,|let's move|tell me about|let's begin|! for joining).*?,,!",
                  "", text, flags=re.IGNORECASE)
    text = re.sub(r"(tell me|describe your|which tools|any \?|you know|for this opportunity\.?)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\?\!]", "", text).strip()

    sentences = []
    for s in text.split("."):
        s = s.strip()
        if s and s not in sentences:
            sentences.append(s)
    text = ". ".join(sentences)

    text = re.sub(r"\s+", " ", text).strip()
    return text

def llm_recruiter_summary(text):
    prompt = """
You are an HR Recruiter. Summarize the candidate’s profile from the interview transcript.
Focus only on:
- Key technical skills
- Soft skills
- Experience & roles
- Career goals (if any)
Write 4-5 clean sentences.
Transcript:
"""
    input_text = prompt + text[:3000]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)

    summary_ids = model.generate(
        **inputs,
        max_length=200,
        min_length=80,
        num_beams=6,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def clean_summary(summary):

    summary = re.sub(r"[^.]*\?", "", summary)

    summary = re.sub(r"(He would like to know.*?|He also wants to know.*?|He is looking for.*)", "", summary)

    summary = re.sub(r"\s+", " ", summary).strip()

    return summary

global_skills = [
    "python", "java", "c++", "sql", "excel", "power bi", "tableau",
    "machine learning", "deep learning", "nlp","django","pandas",
    "communication", "leadership", "teamwork",
    "git", "docker", "api", "cloud",
    "Jenkins for CI/CD",
    "algorithms", "data structures",
    "campaigns", "brand awareness", "digital marketing","pipeline",
    "seo", "branding", "digital marketing", "analytics",
    "recruitment", "employee engagement", "policy implementation",
    "agile", "scrum", "project management","detail-oriented",
    "communication", "leadership", "teamwork", "problem solving", "employee engagement",
    "collaboration", "stakeholder management","cross-functional coordination"

]

def extract_skills(summary):
    detected = []
    for i in global_skills:
        if i in summary.lower():
          detected.append(i)
    return list(set(detected))

def check_skills_in_summary(summary, required_skills_input):
    
    if isinstance(required_skills_input, str):
        required_skills = [skill.strip().lower() for skill in required_skills_input.split(',')]
    elif isinstance(required_skills_input, list):
        required_skills = [skill.strip().lower() for skill in required_skills_input]
    else:
        return "Invalid required skills input"
    
    
    if isinstance(summary, list):
        summary_lower = " ".join(summary).lower()
    else:
        summary_lower = str(summary).lower()

    matched_skills = [skill for skill in required_skills if skill in summary_lower]
    
    total = len(required_skills)
    matched = len(matched_skills)

    if matched > total / 2:
        return "Positive"
    elif matched == total / 2:
        return "Neutral"
    else:
        return "Negative"


    
red_flag_phrases = [
    "i don't know", "not sure", "um", "uh", "you know",
    "blame", "issue with manager", "problem with team",
    "i didn't like", "i hate", "no idea", "confused"
]


def detect_red_flags(text):
    if not isinstance(text, str):
        text = str(text)
    flags = []
    lower_text = text.lower()

    for phrase in red_flag_phrases:
        if phrase in lower_text:
            flags.append(phrase)

    if flags:
        return f"Red Flag  - {', '.join(flags)}"
    else:
        return "No Red Flag"
