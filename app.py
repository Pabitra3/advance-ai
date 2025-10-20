# app.py — Intellexa (cleaned & fixed)
import streamlit as st
import os
import requests
import json
import tempfile
from datetime import datetime
from gtts import gTTS
import speech_recognition as sr
import matplotlib.pyplot as plt
import pandas as pd
import re
import networkx as nx
import imageio
from fpdf import FPDF
from dotenv import load_dotenv

# =============================
# Intellexa AI Tutor Settings
# =============================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = "gpt-4o-mini"
IMAGEGEN_API_KEY = os.getenv("IMAGEGEN_API_KEY")


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="🧠 Intellexa AI Tutor",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# Custom CSS
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
tabs = st.tabs([
    "📚 Learning Plan",
    "🤖 AI Tutor",
    "🎤 AI Interview Coach",
    "📊 Progress Dashboard",
    "⚡ AI Doubt Visualizer"
])

# =========================
# Helper functions
# =========================
def call_openrouter(prompt, timeout=40):
    """Call OpenRouter chat completions and return text or raise Exception."""
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error {r.status_code}: {r.text}")

def create_pdf(plan_text, student_name, goal, filename="Learning_Plan.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Learning Plan for {student_name}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Goal: {goal}", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(6)
    for line in plan_text.split("\n"):
        pdf.multi_cell(0, 7, line)
    pdf.output(filename)
    return filename

def draw_step_graph(steps, highlight_index, outpath):
    G = nx.DiGraph()
    for i in range(len(steps) - 1):
        G.add_edge(i, i+1)
    labels = {i: f"{i+1}. {steps[i]['title']}" for i in range(len(steps))}
    pos = {i: (i, 0) for i in range(len(steps))}
    node_colors = ["#4CAF50" if i == highlight_index else "#B0BEC5" for i in range(len(steps))]
    node_sizes = [1400 if i == highlight_index else 900 for i in range(len(steps))]
    plt.figure(figsize=(8, 3))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")
    # show detail below graph
    plt.text(0, -0.8, steps[highlight_index]["detail"], fontsize=10, wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# =========================
# Tab 1: Learning Plan
# =========================
with tabs[0]:
    st.subheader("📅 AI-Powered Personalized Learning Plan")
    goal = st.selectbox(
        "Choose your Goal",
        ["Data Analytics","Web Development","Machine Learning","Data Science",
         "MERN Stack","Java Development","Android Development"]
    )

    if "ai_plan_text" not in st.session_state:
        st.session_state["ai_plan_text"] = None

    if st.button("✨ Generate AI Learning Plan"):
        if not name:
            st.warning("Please enter your name in the sidebar first.")
        else:
            with st.spinner("AI is generating your personalized study plan..."):
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
                try:
                    ai_plan = call_openrouter(prompt, timeout=60)
                    st.session_state["ai_plan_text"] = ai_plan
                    st.success("✅ AI learning plan generated successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to generate plan: {e}")

    if st.session_state.get("ai_plan_text"):
        plan_text = st.session_state["ai_plan_text"]
        st.markdown("#### Your AI-generated plan:")
        st.code(plan_text, language="")

        # parse days robustly
        day_blocks = re.findall(r"(Day\s*\d{1,2}[:\-]?\s*)([^\n]+)", plan_text)
        if day_blocks:
            st.markdown("### 🗓️ Click a Day to View Details")
            if "plan_progress" not in st.session_state:
                st.session_state["plan_progress"] = {}
            for i, (day, content) in enumerate(day_blocks):
                label = day.strip().replace(":", "")
                with st.expander(f"{label}"):
                    st.markdown(f"**{label}** — {content.strip()}")
                    done = st.checkbox(f"✅ Mark {label} as Done", key=f"done_{i}")
                    st.session_state["plan_progress"][label] = bool(done)

            # Download PDF
            if st.button("📄 Download Learning Plan as PDF"):
                try:
                    pdf_path = create_pdf(plan_text, name or "Student", goal)
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(pdf_path))
                except Exception as e:
                    st.error(f"Failed to create PDF: {e}")
        else:
            st.warning("Could not parse AI response into days. Try regenerating.")

# =========================
# Tab 2: AI Tutor (Adaptive)
# =========================
with tabs[1]:
    st.subheader("🤖 AI Tutor Assistant")
    if "ai_plan_text" in st.session_state and st.session_state["ai_plan_text"]:
        topic_list = re.findall(r"Day\s*\d{1,2}[:\-]?\s*(.+?)(?:—|-|\n|$)", st.session_state["ai_plan_text"])
        topic_list = [t.strip() for t in topic_list if t.strip()]
    else:
        topic_list = ["General Concepts", "Exercises", "Mini Project"]

    chosen_topic = st.selectbox("Select a Topic", topic_list)
    action = st.radio(
        "Choose AI Action",
        ["Explain Topic Clearly", "Generate Practice Questions", "Suggest Next Topic", "Give Study Improvement Tips"]
    )

    if st.button("Ask AI ✨", key="ai_tutor"):
        if not name:
            st.warning("Please enter your name in the sidebar first.")
        else:
            with st.spinner("AI is generating..."):
                style = {
                    "Beginner": "Use simple analogies, step-by-step examples, small hands-on exercises.",
                    "Intermediate": "Provide conceptual depth, practical tips, and intermediate exercises.",
                    "Advanced": "Give in-depth technical explanation, advanced examples, and references."
                }[level]
                prompt = f"You are an AI tutor for {goal}. {action} about {chosen_topic} for a {level} learner. {style}"
                try:
                    result = call_openrouter(prompt, timeout=40)
                    # Show result safely
                    st.markdown("### AI Response")
                    st.write(result)
                except Exception as e:
                    st.error(f"AI request failed: {e}")

# =========================
# Tab 3: AI Interview Coach
# =========================
with tabs[2]:
    st.subheader("🎤 AI Interview Coach")
    domain = goal or st.selectbox("Choose your Interview Domain", ["Machine Learning","Web Development","Data Analytics","Data Science"])
    st.info(f"📘 Interview Domain: **{domain}**")
    num_questions = st.radio("Select number of interview questions", [50, 100, 200], horizontal=True)
    if st.button("⚙️ Generate Interview Questions"):
        with st.spinner("AI is generating interview questions..."):
            prompt = f"You are an expert technical interviewer for {domain}. Generate {num_questions} high-quality interview questions numbered 1..{num_questions}."
            try:
                qs_text = call_openrouter(prompt, timeout=90)
                st.session_state["interview_questions"] = qs_text
                st.success("✅ Interview questions generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate questions: {e}")

    if "interview_questions" in st.session_state and st.session_state["interview_questions"]:
        questions_text = st.session_state["interview_questions"]
        question_list = re.findall(r"\d+\.\s*(.+)", questions_text)
        if question_list:
            st.markdown("### 🧩 Your Interview Questions")
            if "interview_answers" not in st.session_state:
                st.session_state["interview_answers"] = {}
            # limit displayed questions to avoid huge UI freeze; show first 200 only
            for idx, q in enumerate(question_list, 1):
                with st.expander(f"Q{idx}: {q}"):
                    mode = st.radio(f"Answer mode for Q{idx}", ["Text", "Voice"], key=f"mode_{idx}")
                    user_answer = ""
                    if mode == "Text":
                        user_answer = st.text_area(f"✍️ Your answer for Q{idx}", key=f"ans_{idx}")
                    else:
                        audio_file = st.file_uploader(f"🎙️ Upload voice answer for Q{idx} (.mp3)", type=["mp3"], key=f"audio_{idx}")
                        if audio_file:
                            recognizer = sr.Recognizer()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                tmp_file.write(audio_file.read())
                                tmp_path = tmp_file.name
                            try:
                                with sr.AudioFile(tmp_path) as source:
                                    audio_data = recognizer.record(source)
                                    user_answer = recognizer.recognize_google(audio_data)
                                    st.info(f"Transcribed Answer: {user_answer}")
                            except Exception as e:
                                st.error("Could not transcribe audio. Try typing your answer.")
                    if user_answer and st.button(f"🧠 Evaluate Answer Q{idx}", key=f"eval_{idx}"):
                        with st.spinner("AI is evaluating your answer..."):
                            eval_prompt = (
                                f"You are a senior interviewer in {domain}. "
                                f"Evaluate the following answer for question: '{q}'.\nAnswer: '{user_answer}'.\n"
                                "Provide: a score (0-10) for confidence, technical depth, and clarity. "
                                "Then 2-3 concise improvement suggestions."
                            )
                            try:
                                feedback = call_openrouter(eval_prompt, timeout=60)
                                st.markdown("**🧩 AI Feedback:**")
                                st.write(feedback)
                                st.session_state["interview_answers"][idx] = {"question": q, "answer": user_answer, "feedback": feedback}
                                # TTS playback
                                try:
                                    tts = gTTS(text=feedback, lang="en")
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as ta:
                                        tts.save(ta.name)
                                        st.audio(ta.name)
                                except Exception:
                                    pass
                            except Exception as e:
                                st.error(f"Evaluation failed: {e}")
        else:
            st.warning("Could not extract questions properly. Try regenerating with fewer questions.")

# =========================
# Tab 4: Progress Dashboard
# =========================
with tabs[3]:
    st.subheader("📊 Progress Dashboard")
    # Show completed days
    completed = st.session_state.get("plan_progress", {})
    if completed:
        completed_days = [d for d, v in completed.items() if v]
        if completed_days:
            st.markdown(f"**✅ Completed Days:** {', '.join(completed_days)}")
        else:
            st.info("No days marked completed yet.")
    else:
        st.info("No learning plan yet. Generate one in the Learning Plan tab.")

    # Interview history
    history = st.session_state.get("interview_answers", {})
    if history:
        df = pd.DataFrame(history).T
        st.markdown("### 📝 Interview Answers & Feedback")
        st.dataframe(df[["question", "answer", "feedback"]])

        # Try extracting numeric scores
        def extract_score(text, metric):
            match = re.search(f"{metric}: *(\\d+)", text, re.IGNORECASE)
            return int(match.group(1)) if match else None

        df["Confidence"] = df["feedback"].apply(lambda x: extract_score(x, "confidence"))
        df["Technical"] = df["feedback"].apply(lambda x: extract_score(x, "technical"))
        df["Clarity"] = df["feedback"].apply(lambda x: extract_score(x, "clarity"))

        st.markdown("### 📈 Score Trends")
        fig, ax = plt.subplots(figsize=(10, 4))
        if df["Confidence"].notnull().any():
            ax.plot(df.index, df["Confidence"], marker="o", label="Confidence")
        if df["Technical"].notnull().any():
            ax.plot(df.index, df["Technical"], marker="o", label="Technical")
        if df["Clarity"].notnull().any():
            ax.plot(df.index, df["Clarity"], marker="o", label="Clarity")
        ax.set_ylim(0, 10)
        ax.set_xlabel("Question Number (index)")
        ax.set_ylabel("Score (0-10)")
        ax.legend()
        st.pyplot(fig)

        # Achievements
        st.markdown("### 🏅 Achievements")
        if len(df) >= 1:
            st.success("🥇 First Interview Completed!")
        if df["Confidence"].dropna().max() >= 8 if df["Confidence"].dropna().size else False:
            st.success("💡 Confidence Master Badge!")
        if df["Technical"].dropna().max() >= 8 if df["Technical"].dropna().size else False:
            st.success("🧠 Technical Genius Badge!")
        if df["Clarity"].dropna().max() >= 8 if df["Clarity"].dropna().size else False:
            st.success("🎯 Clear Communicator Badge!")
    else:
        st.info("No interview answers yet. Use the AI Interview Coach tab first.")

# =========================
# Tab 5: AI Doubt Visualizer (FIXED)
# =========================
with tabs[4]:
    st.header("🎨 AI Doubt Visualizer")

    doubt = st.text_area("💭 Enter your doubt or concept to visualize")
    
    # Option 1: Use AI to generate a textual diagram/explanation
    if st.button("🎨 Generate Visual Explanation"):
        with st.spinner("Generating visual explanation..."):
            try:
                # Since OpenRouter's image generation endpoint might not be available,
                # we'll use the text API to generate a detailed visual description
                # and create a diagram using matplotlib or mermaid
                
                visual_prompt = f"""
                Create a detailed step-by-step visual explanation for this concept: {doubt}
                
                Format your response as:
                1. Main Concept: [brief description]
                2. Key Components: [list 3-5 main parts]
                3. How It Works: [step-by-step process]
                4. Visual Representation: [describe a simple diagram structure]
                5. Simple Analogy: [relate to everyday concept]
                
                Make it clear, beginner-friendly, and easy to visualize.
                """
                
                # Get AI explanation
                explanation = call_openrouter(visual_prompt, timeout=60)
                
                # Display the explanation
                st.markdown("### 🧠 AI Visual Explanation")
                st.info(explanation)
                
                # Generate a simple flowchart/diagram using matplotlib
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.axis('off')
                    
                    # Parse the explanation to create a visual
                    lines = explanation.split('\n')
                    y_pos = 0.9
                    
                    for line in lines[:15]:  # Show first 15 lines
                        if line.strip():
                            # Color code different sections
                            if 'Main Concept' in line or 'Key Components' in line:
                                bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#4CAF50', alpha=0.8)
                            elif 'How It Works' in line:
                                bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#2196F3', alpha=0.8)
                            elif 'Visual Representation' in line or 'Analogy' in line:
                                bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#FF9800', alpha=0.8)
                            else:
                                bbox_props = dict(boxstyle='round,pad=0.3', facecolor='#E0E0E0', alpha=0.6)
                            
                            ax.text(0.5, y_pos, line.strip()[:80], 
                                   ha='center', va='top', wrap=True,
                                   fontsize=10, bbox=bbox_props)
                            y_pos -= 0.12
                    
                    plt.title(f"Visual Explanation: {doubt[:50]}", fontsize=14, fontweight='bold', pad=20)
                    
                    # Save and display
                    diagram_path = "visual_explanation.png"
                    plt.tight_layout()
                    plt.savefig(diagram_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    st.image(diagram_path, caption="🎨 Concept Visualization", use_column_width=True)
                
                except Exception as viz_error:
                    st.warning(f"Could not generate diagram: {viz_error}")
                
                # Add audio explanation
                try:
                    tts_text = f"Let me explain {doubt}. {explanation[:500]}"  # First 500 chars for audio
                    tts = gTTS(text=tts_text, lang="en")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                        tts.save(tmpfile.name)
                        st.audio(tmpfile.name, format="audio/mp3")
                except Exception as tts_error:
                    st.warning(f"Could not generate audio: {tts_error}")
                
            except Exception as e:
                st.error(f"❌ Error generating explanation: {e}")
    
    # Option 2: Generate a Mermaid diagram
    st.markdown("---")
    st.subheader("🔷 Generate Concept Diagram")
    
    if st.button("📊 Create Mermaid Diagram"):
        with st.spinner("Creating diagram..."):
            try:
                diagram_prompt = f"""
                Create a Mermaid flowchart diagram to explain: {doubt}
                
                Return ONLY the mermaid code (starting with ```mermaid and ending with ```).
                Use flowchart syntax (graph TD or graph LR).
                Keep it simple with 5-10 nodes maximum.
                Use descriptive labels and clear arrows.
                
                Example format:
                ```mermaid
                graph TD
                    A[Start] --> B[Process]
                    B --> C[Result]
                ```
                """
                
                mermaid_code = call_openrouter(diagram_prompt, timeout=60)
                
                # Extract mermaid code from markdown
                mermaid_match = re.search(r'```mermaid\n(.*?)\n```', mermaid_code, re.DOTALL)
                
                if mermaid_match:
                    mermaid_diagram = mermaid_match.group(1)
                    st.code(mermaid_diagram, language="mermaid")
                    st.info("💡 Copy this code and paste it into https://mermaid.live to view the diagram")
                else:
                    st.markdown("### Generated Diagram Code:")
                    st.code(mermaid_code)
                
            except Exception as e:
                st.error(f"❌ Error creating diagram: {e}")
    
    # Option 3: Simple concept map using NetworkX
    st.markdown("---")
    st.subheader("🕸️ Generate Concept Map")
    
    if st.button("🗺️ Create Concept Map"):
        with st.spinner("Building concept map..."):
            try:
                map_prompt = f"""
                For the concept '{doubt}', provide:
                1. Central concept name (1-3 words)
                2. 5-7 related sub-concepts (each 1-3 words)
                3. Brief relationship between central and each sub-concept
                
                Format as:
                CENTRAL: [main concept]
                SUB1: [sub-concept] - [relationship]
                SUB2: [sub-concept] - [relationship]
                ...
                """
                
                map_data = call_openrouter(map_prompt, timeout=60)
                
                # Parse the response
                lines = map_data.split('\n')
                central = "Main Concept"
                nodes = []
                
                for line in lines:
                    if 'CENTRAL:' in line:
                        central = line.split('CENTRAL:')[1].split('-')[0].strip()
                    elif 'SUB' in line and ':' in line:
                        parts = line.split(':', 1)[1].split('-')
                        if parts:
                            nodes.append(parts[0].strip())
                
                # Create network graph
                G = nx.Graph()
                G.add_node(central)
                for node in nodes[:7]:  # Limit to 7 sub-concepts
                    G.add_edge(central, node)
                
                # Draw
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_color='#4CAF50', 
                                      node_size=3000, alpha=0.9)
                nx.draw_networkx_nodes(G, pos, nodelist=[central], 
                                      node_color='#FF5722', node_size=5000, alpha=0.9)
                
                # Draw edges and labels
                nx.draw_networkx_edges(G, pos, width=2, alpha=0.6)
                nx.draw_networkx_labels(G, pos, font_size=10, 
                                       font_weight='bold', font_color='white')
                
                plt.title(f"Concept Map: {doubt}", fontsize=16, fontweight='bold')
                plt.axis('off')
                plt.tight_layout()
                
                map_path = "concept_map.png"
                plt.savefig(map_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                st.image(map_path, caption="🗺️ Concept Map", use_column_width=True)
                st.info(map_data)
                
            except Exception as e:
                st.error(f"❌ Error creating concept map: {e}")