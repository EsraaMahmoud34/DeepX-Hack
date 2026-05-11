import streamlit as st
import os
import random
from inference import ABSAPipeline
from config import OUTPUT_DIR
from preprocessing import preprocess_text

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Arabic ABSA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INIT SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current_review" not in st.session_state:
    st.session_state.current_review = ""
if "language" not in st.session_state:
    st.session_state.language = "English"

# --- TRANSLATIONS ---
TRANSLATIONS = {
    "English": {
        "title": "🧠 Arabic ABSA",
        "subtitle": "Aspect-Based Sentiment Analysis AI",
        "example_reviews": "💡 Example Reviews",
        "click_example": "Click any example to load it:",
        "example1_btn": "Example 1 (Mixed)",
        "example2_btn": "Example 2 (Positive)",
        "example3_btn": "Example 3 (Negative)",
        "session_history": "🕒 Session History",
        "no_predictions": "No predictions made yet.",
        "dashboard_title": "✨ AI Review Analyzer Dashboard",
        "dashboard_desc": "Extract insights from customer feedback in real-time using our advanced Aspect-Based Sentiment Analysis models.",
        "input_review": "📝 Input Review",
        "settings": "**Settings**",
        "use_random_id": "Use Random Review ID",
        "custom_review_id": "Custom Review ID",
        "enter_review": "Enter Arabic Review Text:",
        "placeholder": "...اكتب التقييم هنا",
        "analyze_btn": "Analyze Review 🚀",
        "warning_empty": "⚠️ Please enter a review text.",
        "analyzing": "🤖 AI is analyzing your review...",
        "analysis_complete": "✅ Analysis Complete!",
        "analysis_results": "📊 Analysis Results",
        "no_aspects": "No specific aspects detected in this review.",
        "detected_aspects": "#### Detected Aspects & Sentiments",
        "view_preprocessed": "🛠️ View Preprocessed Text",
        "view_json": "📦 View Raw JSON Output",
        "error_prediction": "❌ An error occurred during prediction:",
        "aspects_count": "aspects",
        "positive": "POSITIVE",
        "negative": "NEGATIVE",
        "neutral": "NEUTRAL",
        "aspect_map": lambda x: x.capitalize()
    },
    "العربية": {
        "title": "🧠 تحليل التقييمات",
        "subtitle": "الذكاء الاصطناعي لتحليل المشاعر المبني على الجوانب",
        "example_reviews": "💡 أمثلة للتقييمات",
        "click_example": "انقر على أي مثال لتحميله:",
        "example1_btn": "مثال 1 (مختلط)",
        "example2_btn": "مثال 2 (إيجابي)",
        "example3_btn": "مثال 3 (سلبي)",
        "session_history": "🕒 سجل الجلسة",
        "no_predictions": "لم يتم إجراء أي تحليلات بعد.",
        "dashboard_title": "✨ لوحة تحكم تحليل التقييمات بالذكاء الاصطناعي",
        "dashboard_desc": "استخرج رؤى من تعليقات العملاء في الوقت الفعلي باستخدام نماذج تحليل المشاعر المتقدمة.",
        "input_review": "📝 أدخل التقييم",
        "settings": "**الإعدادات**",
        "use_random_id": "معرّف تقييم عشوائي",
        "custom_review_id": "معرّف تقييم مخصص",
        "enter_review": "أدخل نص التقييم بالعربية:",
        "placeholder": "...اكتب التقييم هنا",
        "analyze_btn": "تحليل التقييم 🚀",
        "warning_empty": "⚠️ يرجى إدخال نص التقييم.",
        "analyzing": "🤖 الذكاء الاصطناعي يحلل التقييم...",
        "analysis_complete": "✅ اكتمل التحليل!",
        "analysis_results": "📊 نتائج التحليل",
        "no_aspects": "لم يتم اكتشاف جوانب محددة في هذا التقييم.",
        "detected_aspects": "#### الجوانب والمشاعر المكتشفة",
        "view_preprocessed": "🛠️ عرض النص المعالج",
        "view_json": "📦 عرض مخرجات JSON الأولية",
        "error_prediction": "❌ حدث خطأ أثناء التوقع:",
        "aspects_count": "جوانب",
        "positive": "إيجابي",
        "negative": "سلبي",
        "neutral": "محايد",
        "aspect_map": lambda x: {
            "food": "الطعام", "service": "الخدمة", "price": "السعر",
            "cleanliness": "النظافة", "delivery": "التوصيل", "ambiance": "الأجواء",
            "app_experience": "تجربة التطبيق", "general": "عام", "none": "لا يوجد"
        }.get(x, x)
    }
}

# --- SIDEBAR: Language Toggle First ---
language = st.sidebar.radio("🌐 Language / اللغة", ["English", "العربية"], horizontal=True)
st.session_state.language = language
t = TRANSLATIONS[language]

# --- CUSTOM CSS ---
rtl_css = """
    <style>
    .stMarkdown, .stButton>button, .stCheckbox, .stNumberInput>div>label, .stTextArea>label {
        direction: rtl;
        text-align: right;
    }
    .stAlert {
        direction: rtl;
        text-align: right;
    }
    </style>
""" if language == "العربية" else ""

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Cairo', sans-serif;
    }}

    /* Target the text area specifically for RTL support */
    textarea {{
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Cairo', sans-serif !important;
        font-size: 1.1rem !important;
    }}

    .rtl-text {{
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
        font-size: 1.1rem;
    }}

    .sentiment-badge {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 10px;
        width: 100%;
        text-align: center;
    }}

    .badge-positive {{
        color: #155724;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }}
    .badge-negative {{
        color: #721c24;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }}
    .badge-neutral {{
        color: #383d41;
        background-color: #e2e3e5;
        border: 1px solid #d6d8db;
    }}
    </style>
    {rtl_css}
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource(show_spinner=False)
def load_pipeline():
    aspect_model_dir = os.path.join(OUTPUT_DIR, "aspect_model")
    sentiment_model_dir = os.path.join(OUTPUT_DIR, "sentiment_model")

    if not os.path.exists(aspect_model_dir) or not os.path.exists(sentiment_model_dir):
        st.error("Models not found. Please ensure train.py was run and models are in the outputs folder.")
        st.stop()

    return ABSAPipeline(aspect_model_dir, sentiment_model_dir)

with st.spinner("Loading AI Models..." if language == "English" else "جاري تحميل النماذج..."):
    pipeline = load_pipeline()

# --- SIDEBAR CONTINUED ---
with st.sidebar:
    st.markdown("---")
    st.title(t["title"])
    st.markdown(t["subtitle"])
    st.markdown("---")

    st.subheader(t["example_reviews"])
    st.markdown(t["click_example"])

    example1 = "الأكل ممتاز لكن الخدمة بطيئة والأسعار غالية"
    example2 = "المطعم نظيف جداً وتجربة التطبيق رائعة وسهلة"
    example3 = "التوصيل تأخر جداً والطعام كان بارد، تجربة سيئة"

    if st.button(t["example1_btn"], use_container_width=True):
        st.session_state.current_review = example1
    if st.button(t["example2_btn"], use_container_width=True):
        st.session_state.current_review = example2
    if st.button(t["example3_btn"], use_container_width=True):
        st.session_state.current_review = example3

    st.markdown("---")
    st.subheader(t["session_history"])
    if not st.session_state.history:
        st.info(t["no_predictions"])
    else:
        for idx, item in enumerate(reversed(st.session_state.history[-5:])): # Show last 5
            with st.expander(f"ID #{item['id']} ({item['aspect_count']} {t['aspects_count']})"):
                st.markdown(f"<div class='rtl-text' style='font-size:0.95rem;'>{item['text']}</div>", unsafe_allow_html=True)

# --- MAIN PANEL ---
st.title(t["dashboard_title"])
st.markdown(t["dashboard_desc"])

# --- INPUT CONTAINER ---
with st.container(border=True):
    st.subheader(t["input_review"])

    col1, col2 = st.columns([1, 4] if language == "English" else [4, 1])

    # We will adjust layout for RTL
    settings_col = col1 if language == "English" else col2
    text_col = col2 if language == "English" else col1

    with settings_col:
        st.markdown(t["settings"])
        use_random_id = st.checkbox(t["use_random_id"], value=True)
        review_id = st.number_input(t["custom_review_id"], min_value=1, value=10018, step=1, disabled=use_random_id)
        if use_random_id:
            review_id = random.randint(10000, 99999)

    with text_col:
        review_text = st.text_area(
            t["enter_review"],
            value=st.session_state.current_review,
            height=130,
            placeholder=t["placeholder"]
        )

analyze_btn = st.button(t["analyze_btn"], type="primary", use_container_width=True)

# --- INFERENCE & OUTPUT ---
if analyze_btn:
    if not review_text.strip():
        st.warning(t["warning_empty"])
    else:
        with st.spinner(t["analyzing"]):
            try:
                # Backend logic
                cleaned_text = preprocess_text(review_text)
                result = pipeline.predict(review_text, review_id=review_id)

                aspects = result.get("aspects", [])
                sentiments = result.get("aspect_sentiments", {})

                # Update history
                st.session_state.history.append({
                    "id": review_id,
                    "text": review_text,
                    "aspect_count": len(aspects) if aspects != ["none"] else 0
                })

                st.success(t["analysis_complete"])

                st.markdown("---")
                st.subheader(t["analysis_results"])

                # Aspect & Sentiment Cards
                if aspects == ["none"] or not aspects:
                    st.info(t["no_aspects"])
                else:
                    st.markdown(t["detected_aspects"])

                    # RTL support for columns
                    cols = st.columns(min(len(aspects), 4))
                    if language == "العربية":
                        cols = list(reversed(cols))

                    for i, aspect in enumerate(aspects):
                        sentiment = sentiments.get(aspect, "unknown")

                        badge_class = "badge-neutral"
                        if sentiment == "positive": badge_class = "badge-positive"
                        elif sentiment == "negative": badge_class = "badge-negative"

                        emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐", "unknown": "❓"}
                        emoji = emoji_map.get(sentiment, "❓")

                        disp_sentiment = t.get(sentiment, sentiment.upper())
                        disp_aspect = t["aspect_map"](aspect)

                        with cols[i % len(cols)]:
                            with st.container(border=True):
                                st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>{disp_aspect}</h3>", unsafe_allow_html=True)
                                st.markdown(f"<div class='sentiment-badge {badge_class}'>{emoji} {disp_sentiment}</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Expanders for detailed views
                col_exp1, col_exp2 = st.columns(2)
                if language == "العربية":
                    col_exp1, col_exp2 = col_exp2, col_exp1

                with col_exp1:
                    with st.expander(t["view_preprocessed"]):
                        st.markdown(f"<div class='rtl-text'>{cleaned_text}</div>", unsafe_allow_html=True)

                with col_exp2:
                    with st.expander(t["view_json"]):
                        st.json(result)

            except Exception as e:
                st.error(f"{t['error_prediction']} {str(e)}")
