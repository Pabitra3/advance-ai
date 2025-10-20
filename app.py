import streamlit as st
import os, requests, json, tempfile
from datetime import datetime
from gtts import gTTS
import speech_recognition as sr
import matplotlib.pyplot as plt
import pandas as pd

# =============================
# Intellexa AI Tutor Settings
# =============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = "gpt-4o-mini"

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="🧠 Intellexa AI Tutor",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# Custom CSS for buttons and layout
# -------------------------
st.markdown("""
<style>
.main .block-container{padding-top:1rem;}
.stButton>button{background-color:#4CAF50;color:white;font-weight:bold;height:3em;width:100%;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Intellexa — AI Learning & Interview Coach")
st.caption("Personalized AI Tutor + Voice-Based AI Interview Coach")

# -------------------------
# Sidebar Inputs
# -------------------------
with st.sidebar:
    st.header("🎯 User Settings")
    name = st.text_input("Enter your name", "")
    level = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
    hours = st.slider("Study Time (hours/day)", 1, 6, 2)
    days = st.slider("Number of Days for Completion", 5, 30, 10)

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["📚 Learning Plan", "🤖 AI Tutor", "🎤 AI Interview Coach", "📊 Progress Dashboard"])

# -------------------------
# Tab 1: Learning Plan (AI-Generated)
# -------------------------
with tabs[0]:
    st.subheader("📅 AI-Powered Personalized Learning Plan")

    goal = st.selectbox(
        "Choose your Goal",
        [
            "Data Analytics",
            "Web Development",
            "Machine Learning",
            "Data Science",
            "MERN Stack",
            "Java Development",
            "Android Development",
        ],
    )

    if name:
        if st.button("✨ Generate AI Learning Plan"):
            with st.spinner("AI is generating your personalized study plan..."):
                # Construct dynamic prompt for AI
                prompt = f"""
                You are an expert AI tutor.
                Create a personalized {days}-day learning plan for a student named {name}
                who wants to master {goal}.
                The student's skill level is {level}, and they can study {hours} hours per day.

                The plan must be returned as a numbered list in this format:
                Day 01: [Topic Name] — [Key Concepts/Activities]
                Day 02: [Topic Name] — [Key Concepts/Activities]
                ...
                Up to Day {days}.
                Be detailed and tailored to their skill level and available time.
                """

                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": prompt}],
                }

                try:
                    r = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=40,
                    )
                    if r.status_code == 200:
                        ai_plan = r.json()["choices"][0]["message"]["content"]
                        st.session_state["ai_plan_text"] = ai_plan
                        st.success("✅ AI learning plan generated successfully!")
                    else:
                        st.error(f"❌ API Error: {r.status_code}")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")

    # -------------------------
    # Display the AI Plan as Clickable Templates
    # -------------------------
    if "ai_plan_text" in st.session_state:
        ai_plan = st.session_state["ai_plan_text"]

        # Parse plan into list of (Day, Content)
        import re
        day_blocks = re.findall(r"(Day\s*\d{1,2}[:\-]?\s*)([^\n]+)", ai_plan)
        if day_blocks:
            st.markdown("### 🗓️ Click a Day to View Details")
            for i, (day, content) in enumerate(day_blocks):
                with st.expander(f"{day.strip().replace(':','')}"):
                    st.markdown(f"**{day.strip()}** — {content.strip()}")
        else:
            st.warning("Could not parse AI response. Try regenerating.")


# -------------------------
# Tab 2: AI Tutor
# -------------------------
with tabs[1]:
    if name:
        st.subheader("🤖 AI Tutor Assistant")
        chosen_topic = st.selectbox("Select a Topic", topic_list)
        action = st.radio(
            "Choose AI Action",
            ["Explain Topic Clearly","Generate Practice Questions",
             "Suggest Next Topic","Give Study Improvement Tips"]
        )

        if st.button("Ask AI ✨", key="ai_tutor"):
            with st.spinner("AI is generating..."):
                prompt = f"You are an AI tutor for {goal}. {action} about {chosen_topic} for a {level} learner."
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
                try:
                    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
                    if r.status_code == 200:
                        result = r.json()["choices"][0]["message"]["content"]
                        st.success(result)
                    else:
                        st.error(f"❌ API Error: {r.status_code} — {r.text}")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")

# -------------------------
# Tab 3: AI Interview Coach
# -------------------------
with tabs[2]:
    st.subheader("🎤 AI Interview Coach")
    domain = st.selectbox("Choose your Interview Domain", ["Machine Learning", "Web Development", "Data Analytics", "Data Science"])
    mode = st.radio("Answer Mode", ["Text", "Voice"])
    user_answer = ""

    if mode == "Text":
        user_answer = st.text_area("Type your answer here")
    else:
        audio_file = st.file_uploader("Upload your answer as audio (.wav or .mp3)", type=["wav","mp3"])
        if audio_file:
            r = sr.Recognizer()
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            with sr.AudioFile(tmp_path) as source:
                audio_data = r.record(source)
                try:
                    user_answer = r.recognize_google(audio_data)
                    st.info(f"Transcribed Answer: {user_answer}")
                except:
                    st.error("Could not recognize audio. Try typing instead.")

    if user_answer:
        if st.button("Evaluate Answer", key="ai_interview"):
            with st.spinner("AI evaluating..."):
                prompt = f"Act as an interview evaluator for {domain}. Evaluate this answer: '{user_answer}'. Provide a score (0-10) for confidence, tone, technical accuracy and give improvement suggestions."
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
                try:
                    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
                    if r.status_code == 200:
                        result = r.json()["choices"][0]["message"]["content"]
                        st.markdown(f"**AI Feedback:**\n{result}")

                        # Audio Feedback
                        tts = gTTS(text=result, lang="en")
                        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tts.save(tmp_audio.name)
                        audio_bytes = open(tmp_audio.name, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")

                        # Save Feedback
                        if "interview_history" not in st.session_state:
                            st.session_state["interview_history"] = []
                        st.session_state["interview_history"].append({
                            "domain": domain, "answer": user_answer, "feedback": result, "date": str(datetime.now())
                        })
                    else:
                        st.error(f"❌ API Error: {r.status_code}")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")

# -------------------------
# Tab 4: Progress Dashboard
# -------------------------
with tabs[3]:
    st.subheader("📊 Interview Progress Dashboard")
    history = st.session_state.get("interview_history", [])
    if history:
        df = pd.DataFrame(history)
        st.dataframe(df[["date","domain","answer"]])

        # Naive score extraction
        def extract_score(text, metric):
            import re
            match = re.search(f"{metric}: *(\\d+)", text)
            return int(match.group(1)) if match else None

        df["Confidence"] = df["feedback"].apply(lambda x: extract_score(x, "confidence"))
        df["Tone"] = df["feedback"].apply(lambda x: extract_score(x, "tone"))
        df["Technical"] = df["feedback"].apply(lambda x: extract_score(x, "technical"))

        # Line chart
        st.markdown("### 📈 Score Trends")
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["Confidence"], label="Confidence", marker='o', color="#4CAF50")
        ax.plot(df["date"], df["Tone"], label="Tone", marker='o', color="#FF9800")
        ax.plot(df["date"], df["Technical"], label="Technical", marker='o', color="#2196F3")
        ax.set_xlabel("Date")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 10)
        ax.legend()
        st.pyplot(fig)

        # Badges
        st.markdown("### 🏅 Achievements")
        if len(df) >= 1:
            st.success("🥇 First Interview Completed!")
        if df["Confidence"].iloc[-1] and df["Confidence"].iloc[-1] >= 8:
            st.success("💡 Confidence Master Badge!")
        if df["Technical"].iloc[-1] and df["Technical"].iloc[-1] >= 8:
            st.success("🧠 Technical Genius Badge!")
    else:
        st.info("No interviews yet. Try the AI Interview Coach tab first.")