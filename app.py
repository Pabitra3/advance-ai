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
        # Extract topics dynamically from AI plan
        if "ai_plan_text" in st.session_state:
            import re
            plan_text = st.session_state["ai_plan_text"]
            topic_list = re.findall(r"Day\s*\d{1,2}[:\-]?\s*(.+?)(?:—|-|\n|$)", plan_text)
            topic_list = [t.strip() for t in topic_list if len(t.strip()) > 2]
        else:
            topic_list = ["General Concepts", "Exercises", "Mini Project"]
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
# Tab 3: AI Interview Coach (Enhanced)
# -------------------------
with tabs[2]:
    st.subheader("🎤 AI Interview Coach")

    # 1️⃣ Auto-fetch domain from goal
    domain = goal if goal else st.selectbox(
        "Choose your Interview Domain", 
        ["Machine Learning", "Web Development", "Data Analytics", "Data Science"]
    )
    st.info(f"📘 Interview Domain: **{domain}**")

    # 2️⃣ Choose number of questions
    num_questions = st.radio(
        "Select number of interview questions",
        [50, 100, 200],
        horizontal=True
    )

    # 3️⃣ Generate Questions
    if st.button("⚙️ Generate Interview Questions"):
        with st.spinner("AI is generating interview questions..."):
            prompt = f"""
            You are an expert technical interviewer for {domain}.
            Generate {num_questions} high-quality interview questions covering beginner to advanced concepts.
            Each question should be clear, concise, and relevant.
            Number the questions as:
            1. Question one
            2. Question two
            ...
            """
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}

            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if r.status_code == 200:
                    ai_questions = r.json()["choices"][0]["message"]["content"]
                    st.session_state["interview_questions"] = ai_questions
                    st.success("✅ Interview questions generated successfully!")
                else:
                    st.error(f"❌ API Error: {r.status_code}")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")

    # 4️⃣ Show the questions if available
    if "interview_questions" in st.session_state:
        import re
        questions_text = st.session_state["interview_questions"]

        # Extract numbered questions
        question_list = re.findall(r"\d+\.\s*(.+)", questions_text)
        if question_list:
            st.markdown("### 🧩 Your Interview Questions")

            # Initialize answer storage
            if "interview_answers" not in st.session_state:
                st.session_state["interview_answers"] = {}

            for idx, q in enumerate(question_list, 1):
                with st.expander(f"Q{idx}: {q}"):
                    mode = st.radio(
                        f"Answer mode for Q{idx}",
                        ["Text", "Voice"],
                        key=f"mode_{idx}"
                    )

                    user_answer = ""

                    if mode == "Text":
                        user_answer = st.text_area(f"✍️ Your answer for Q{idx}", key=f"ans_{idx}")
                    else:
                        audio_file = st.file_uploader(
                            f"🎙️ Upload your voice answer for Q{idx} (.mp3)", 
                            type=["mp3"], 
                            key=f"audio_{idx}"
                        )
                        if audio_file:
                            import tempfile
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
                                    st.error("❌ Could not recognize audio. Try again or type manually.")

                    # 5️⃣ Evaluate each answer individually
                    if user_answer and st.button(f"🧠 Evaluate Answer Q{idx}", key=f"eval_{idx}"):
                        with st.spinner("AI is evaluating your answer..."):
                            eval_prompt = f"""
                            You are a senior interviewer in {domain}.
                            Evaluate the following answer for question: '{q}'.
                            Answer: '{user_answer}'.
                            Provide:
                            - A score (0-10) for confidence, technical depth, and clarity.
                            - 2-3 lines of improvement suggestions.
                            """
                            headers = {
                                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                                "Content-Type": "application/json",
                            }
                            payload = {
                                "model": MODEL_ID,
                                "messages": [{"role": "user", "content": eval_prompt}],
                            }

                            try:
                                r = requests.post(
                                    "https://openrouter.ai/api/v1/chat/completions",
                                    headers=headers,
                                    json=payload,
                                    timeout=60,
                                )
                                if r.status_code == 200:
                                    feedback = r.json()["choices"][0]["message"]["content"]
                                    st.markdown(f"**🧩 AI Feedback:**\n{feedback}")
                                    st.session_state["interview_answers"][idx] = {
                                        "question": q,
                                        "answer": user_answer,
                                        "feedback": feedback,
                                    }

                                    # Optional: Convert feedback to voice
                                    try:
                                        tts = gTTS(text=feedback, lang="en")
                                        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                                        tts.save(tmp_audio.name)
                                        st.audio(tmp_audio.name)
                                    except:
                                        pass
                                else:
                                    st.error(f"❌ API Error: {r.status_code}")
                            except Exception as e:
                                st.error(f"❌ Request failed: {e}")
        else:
            st.warning("⚠️ Could not extract questions from AI response. Try regenerating.")


# -------------------------
# Tab 4: Progress Dashboard (AI Interview Integration)
# -------------------------
with tabs[3]:
    st.subheader("📊 Interview Progress Dashboard")

    # Fetch saved answers from interview tab
    history = st.session_state.get("interview_answers", {})

    if history:
        # Convert dict to DataFrame
        import pandas as pd
        df = pd.DataFrame(history).T  # transpose to get each idx as a row
        st.dataframe(df[["question", "answer", "feedback"]])

        # -------------------------
        # Extract scores from AI feedback
        # -------------------------
        import re

        def extract_score(text, metric):
            # Example: 'confidence: 8'
            match = re.search(f"{metric}: *(\\d+)", text, re.IGNORECASE)
            return int(match.group(1)) if match else None

        df["Confidence"] = df["feedback"].apply(lambda x: extract_score(x, "confidence"))
        df["Technical"] = df["feedback"].apply(lambda x: extract_score(x, "technical"))
        df["Clarity"] = df["feedback"].apply(lambda x: extract_score(x, "clarity"))

        # -------------------------
        # Show charts
        # -------------------------
        import matplotlib.pyplot as plt
        st.markdown("### 📈 Score Trends")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Confidence"], label="Confidence", marker="o", color="#4CAF50")
        ax.plot(df.index, df["Technical"], label="Technical", marker="o", color="#2196F3")
        ax.plot(df.index, df["Clarity"], label="Clarity", marker="o", color="#FF9800")
        ax.set_xlabel("Question Number")
        ax.set_ylabel("Score (0-10)")
        ax.set_ylim(0, 10)
        ax.legend()
        st.pyplot(fig)

        # -------------------------
        # Badges / Achievements
        # -------------------------
        st.markdown("### 🏅 Achievements")
        if len(df) >= 1:
            st.success("🥇 First Interview Completed!")
        if df["Confidence"].dropna().max() >= 8:
            st.success("💡 Confidence Master Badge!")
        if df["Technical"].dropna().max() >= 8:
            st.success("🧠 Technical Genius Badge!")
        if df["Clarity"].dropna().max() >= 8:
            st.success("🎯 Clear Communicator Badge!")
    else:
        st.info("No interview answers yet. Use the AI Interview Coach tab first.")
