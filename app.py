"""Streamlit app for a hybrid college helpdesk chatbot.

Flow:
1. Try local FAQ match using TF-IDF + cosine similarity.
2. If local match confidence is low, fall back to a generative AI API.
"""

from __future__ import annotations

import time
from urllib.parse import quote

import streamlit as st

from utils.api_handler import get_fallback_ai_response
from utils.nlp_engine import get_best_match


QUICK_ACTIONS = [
	{
		"label_en": "Admissions",
		"label_hi": "प्रवेश",
		"prompt_en": "What is the admission process?",
		"prompt_hi": "प्रवेश प्रक्रिया क्या है?",
	},
	{
		"label_en": "Fees",
		"label_hi": "फीस",
		"prompt_en": "What is the fee structure and payment schedule?",
		"prompt_hi": "फीस संरचना और भुगतान समय-सारणी क्या है?",
	},
	{
		"label_en": "Hostel",
		"label_hi": "हॉस्टल",
		"prompt_en": "What are the hostel facilities and charges?",
		"prompt_hi": "हॉस्टल सुविधाएं और शुल्क क्या हैं?",
	},
	{
		"label_en": "Scholarship",
		"label_hi": "छात्रवृत्ति",
		"prompt_en": "How can I apply for scholarships?",
		"prompt_hi": "मैं छात्रवृत्ति के लिए कैसे आवेदन कर सकता/सकती हूं?",
	},
	{
		"label_en": "Exam Form",
		"label_hi": "परीक्षा फॉर्म",
		"prompt_en": "When does exam form submission start?",
		"prompt_hi": "परीक्षा फॉर्म जमा करना कब शुरू होता है?",
	},
]


FOLLOW_UP_MAP = {
	"admission": {
		"en": [
			"What documents are required for admission?",
			"What is the admission deadline?",
			"Is there any entrance test?",
		],
		"hi": [
			"प्रवेश के लिए कौन-कौन से दस्तावेज चाहिए?",
			"प्रवेश की अंतिम तिथि क्या है?",
			"क्या कोई प्रवेश परीक्षा है?",
		],
	},
	"fee": {
		"en": [
			"Are there installment options for fees?",
			"How can I pay fees online?",
			"Is there any late fee penalty?",
		],
		"hi": [
			"क्या फीस किस्तों में जमा कर सकते हैं?",
			"मैं ऑनलाइन फीस कैसे जमा करूं?",
			"लेट फीस पर कोई पेनल्टी है क्या?",
		],
	},
	"hostel": {
		"en": [
			"What are hostel room types?",
			"Are mess charges included?",
			"How do I apply for hostel allocation?",
		],
		"hi": [
			"हॉस्टल के कमरे कितने प्रकार के हैं?",
			"क्या मेस चार्ज शामिल है?",
			"हॉस्टल अलॉटमेंट के लिए कैसे आवेदन करें?",
		],
	},
	"scholarship": {
		"en": [
			"Who is eligible for scholarships?",
			"What is the scholarship application process?",
			"When are scholarship results announced?",
		],
		"hi": [
			"छात्रवृत्ति के लिए कौन पात्र है?",
			"छात्रवृत्ति आवेदन प्रक्रिया क्या है?",
			"छात्रवृत्ति परिणाम कब आते हैं?",
		],
	},
	"exam": {
		"en": [
			"What is the exam form fee?",
			"How can I download my admit card?",
			"What if I miss exam form submission?",
		],
		"hi": [
			"परीक्षा फॉर्म फीस कितनी है?",
			"एडमिट कार्ड कैसे डाउनलोड करें?",
			"यदि परीक्षा फॉर्म छूट जाए तो क्या करें?",
		],
	},
}


TOPIC_KEYWORDS = {
	"admission": ["admission", "apply", "enroll", "प्रवेश", "दाखिला"],
	"fee": ["fee", "fees", "payment", "tuition", "फीस", "भुगतान"],
	"hostel": ["hostel", "mess", "accommodation", "हॉस्टल", "मेस"],
	"scholarship": ["scholarship", "financial aid", "छात्रवृत्ति"],
	"exam": ["exam", "admit card", "backlog", "परीक्षा", "एडमिट कार्ड"],
}


UI_TEXT = {
	"English": {
		"title": "GLA University Helpdesk",
		"caption": "Hybrid chatbot: local FAQ matching + AI fallback",
		"quick_actions": "Quick Actions",
		"followups": "Suggested follow-up questions",
		"ask_placeholder": "Ask your college-related question...",
		"thinking": "Thinking...",
		"need_help": "Need help from office staff?",
		"help_caption": "Submit details to prepare an email draft for the helpdesk office.",
		"name": "Name",
		"roll": "Roll Number",
		"issue": "Issue Summary",
		"issue_placeholder": "Describe your unresolved query here...",
		"submit_btn": "Create Helpdesk Email",
		"warning_form": "Please enter at least your name and issue summary.",
		"success_mail": "Email draft prepared. Open your mail app using the link below.",
		"open_mail": "Open Email Draft",
		"source_local": "Local FAQ Match",
		"source_ai": "AI Response",
		"conf_high": "High",
		"conf_check": "Needs Verification",
		"sidebar_title": "Interaction Insights",
		"sidebar_total": "Total Queries",
		"sidebar_local": "Local Answer Ratio",
		"sidebar_ai": "AI Answer Ratio",
		"sidebar_topics": "Top Asked Topics",
		"sidebar_empty_topics": "Ask a few questions to populate analytics.",
		"language": "Language",
		"typing_cursor": "▌",
		"default_followups": [
			"What are the office timings for student helpdesk?",
			"Who should I contact for urgent academic issues?",
			"Can you share official email or phone details?",
		],
	},
	"Hindi": {
		"title": "GLA यूनिवर्सिटी हेल्पडेस्क",
		"caption": "हाइब्रिड चैटबॉट: लोकल FAQ मैचिंग + AI फॉलबैक",
		"quick_actions": "त्वरित विकल्प",
		"followups": "सुझाए गए अगले प्रश्न",
		"ask_placeholder": "कॉलेज से जुड़ा अपना प्रश्न पूछें...",
		"thinking": "सोच रहा हूं...",
		"need_help": "ऑफिस स्टाफ से मदद चाहिए?",
		"help_caption": "हेल्पडेस्क ऑफिस के लिए ईमेल ड्राफ्ट बनाने हेतु विवरण भरें।",
		"name": "नाम",
		"roll": "रोल नंबर",
		"issue": "समस्या सारांश",
		"issue_placeholder": "अपनी अनसुलझी समस्या यहां लिखें...",
		"submit_btn": "हेल्पडेस्क ईमेल बनाएं",
		"warning_form": "कृपया कम से कम नाम और समस्या सारांश भरें।",
		"success_mail": "ईमेल ड्राफ्ट तैयार है। नीचे लिंक से मेल ऐप खोलें।",
		"open_mail": "ईमेल ड्राफ्ट खोलें",
		"source_local": "लोकल FAQ मैच",
		"source_ai": "AI उत्तर",
		"conf_high": "उच्च",
		"conf_check": "सत्यापन आवश्यक",
		"sidebar_title": "इंटरैक्शन एनालिटिक्स",
		"sidebar_total": "कुल प्रश्न",
		"sidebar_local": "लोकल उत्तर अनुपात",
		"sidebar_ai": "AI उत्तर अनुपात",
		"sidebar_topics": "सबसे अधिक पूछे गए विषय",
		"sidebar_empty_topics": "एनालिटिक्स देखने के लिए कुछ प्रश्न पूछें।",
		"language": "भाषा",
		"typing_cursor": "▌",
		"default_followups": [
			"स्टूडेंट हेल्पडेस्क का समय क्या है?",
			"तुरंत अकादमिक सहायता के लिए किससे संपर्क करें?",
			"क्या आप आधिकारिक ईमेल या फोन साझा कर सकते हैं?",
		],
	},
}


def _inject_custom_styles() -> None:
	"""Inject custom CSS for better visual hierarchy and interaction."""
	st.markdown(
		"""
		<style>
		.badge-row {
			display: flex;
			gap: 0.5rem;
			margin: 0.35rem 0 0.55rem;
		}
		.badge {
			padding: 0.2rem 0.6rem;
			border-radius: 999px;
			font-size: 0.78rem;
			font-weight: 600;
			border: 1px solid rgba(255, 255, 255, 0.2);
			display: inline-block;
		}
		.badge-local {
			background: rgba(0, 153, 102, 0.2);
			color: #9af5d0;
		}
		.badge-ai {
			background: rgba(255, 174, 0, 0.15);
			color: #ffd87a;
		}
		.info-card {
			padding: 0.75rem 0.9rem;
			border: 1px solid rgba(255, 255, 255, 0.14);
			border-radius: 12px;
			background: linear-gradient(
				120deg,
				rgba(28, 33, 49, 0.9),
				rgba(13, 19, 36, 0.95)
			);
			margin-top: 0.5rem;
		}
		.info-card h4 {
			margin: 0 0 0.25rem;
			font-size: 0.95rem;
		}
		.info-card p {
			margin: 0;
			font-size: 0.87rem;
			opacity: 0.95;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def _detect_topic(text: str) -> str:
	"""Infer broad topic from query text for analytics and suggestions."""
	lower_text = text.lower()
	for topic, keywords in TOPIC_KEYWORDS.items():
		if any(keyword in lower_text for keyword in keywords):
			return topic
	return "general"


def _animate_response(text: str, cursor_symbol: str) -> None:
	"""Render a lightweight typing animation for assistant messages."""
	words = text.split()
	if len(words) > 120:
		st.markdown(text)
		return

	placeholder = st.empty()
	progressive = []
	for word in words:
		progressive.append(word)
		placeholder.markdown(f"{' '.join(progressive)} {cursor_symbol}")
		time.sleep(0.015)

	placeholder.markdown(text)


def _build_info_cards(user_query: str, language: str) -> list[dict[str, str]]:
	"""Return lightweight topic cards relevant to the current query."""
	query = user_query.lower()
	if any(keyword in query for keyword in ["admission", "apply", "enroll", "प्रवेश"]):
		if language == "Hindi":
			return [
				{
					"title": "प्रवेश चेकलिस्ट",
					"body": "फॉर्म जमा करने से पहले मार्कशीट, पहचान पत्र, फोटो और "
					"जरूरी प्रमाणपत्र तैयार रखें।",
				},
				{
					"title": "हेल्पडेस्क सुझाव",
					"body": "अंतिम तिथि और प्रक्रिया को हमेशा आधिकारिक यूनिवर्सिटी "
					"पोर्टल पर सत्यापित करें।",
				},
			]
		return [
			{
				"title": "Admissions Checklist",
				"body": "Keep mark sheets, ID proof, photographs, and category "
				"documents ready before form submission.",
			},
			{
				"title": "Helpdesk Tip",
				"body": "Always verify deadlines on the official university portal "
				"before final submission.",
			},
		]

	if any(keyword in query for keyword in ["fee", "payment", "tuition", "फीस"]):
		if language == "Hindi":
			return [
				{
					"title": "फीस भुगतान रिमाइंडर",
					"body": "फीस केवल आधिकारिक चैनल से जमा करें और रसीद सुरक्षित रखें।",
				},
				{
					"title": "लेट फीस अलर्ट",
					"body": "अंतिम तिथि छूटने पर तुरंत अकाउंट्स ऑफिस से संपर्क करें।",
				},
			]
		return [
			{
				"title": "Fee Payment Reminder",
				"body": "Use official payment channels and keep transaction receipts "
				"for future verification.",
			},
			{
				"title": "Late Fee Alert",
				"body": "If you miss the deadline, contact accounts office quickly "
				"to avoid penalties.",
			},
		]

	if any(keyword in query for keyword in ["hostel", "mess", "accommodation", "हॉस्टल"]):
		if language == "Hindi":
			return [
				{
					"title": "हॉस्टल आवंटन",
					"body": "रूम आवंटन सीट उपलब्धता और प्रशासनिक स्वीकृति पर निर्भर करता है।",
				},
				{
					"title": "मेस और सुविधाएं",
					"body": "मेस चार्ज, वाई-फाई, लॉन्ड्री और सिक्योरिटी डिपॉजिट की जानकारी पहले लें।",
				},
			]
		return [
			{
				"title": "Hostel Allocation",
				"body": "Room allocation usually depends on availability, semester, "
				"and approval from hostel administration.",
			},
			{
				"title": "Mess and Facilities",
				"body": "Confirm whether mess charges, Wi-Fi, laundry, and security "
				"deposit are included in your package.",
			},
		]

	if any(keyword in query for keyword in ["scholarship", "financial aid", "छात्रवृत्ति"]):
		if language == "Hindi":
			return [
				{
					"title": "छात्रवृत्ति दस्तावेज",
					"body": "इनकम सर्टिफिकेट, अकादमिक रिकॉर्ड और बैंक विवरण पहले से तैयार रखें।",
				},
				{
					"title": "पात्रता जांच",
					"body": "पात्रता अंक, श्रेणी और पारिवारिक आय के आधार पर तय हो सकती है।",
				},
			]
		return [
			{
				"title": "Scholarship Documents",
				"body": "Prepare income certificate, academic records, and bank "
				"details for faster scholarship processing.",
			},
			{
				"title": "Eligibility Check",
				"body": "Eligibility may depend on marks, category, and annual "
				"family income.",
			},
		]

	if any(keyword in query for keyword in ["exam", "admit card", "backlog", "परीक्षा"]):
		if language == "Hindi":
			return [
				{
					"title": "परीक्षा तैयारी",
					"body": "फॉर्म, विषय और एडमिट कार्ड की स्थिति परीक्षा से पहले जांच लें।",
				},
				{
					"title": "सहायता संपर्क",
					"body": "विषय या हॉल टिकट में त्रुटि हो तो तुरंत परीक्षा सेल से संपर्क करें।",
				},
			]
		return [
			{
				"title": "Exam Preparation",
				"body": "Complete exam form, verify subjects, and check admit card "
				"availability before the exam week.",
			},
			{
				"title": "Support Contact",
				"body": "For discrepancies in subjects or hall ticket details, "
				"contact exam cell immediately.",
			},
		]

	return []


def _suggest_followups(user_query: str, bot_response: str, language: str) -> list[str]:
	"""Suggest follow-up questions based on user context."""
	combined = f"{user_query} {bot_response}".lower()
	language_key = "hi" if language == "Hindi" else "en"
	for topic, suggestions in FOLLOW_UP_MAP.items():
		if topic in combined:
			return suggestions[language_key]

	return UI_TEXT[language]["default_followups"]


def _render_badges(source: str, confidence: str, language: str) -> None:
	"""Render source and confidence badges for assistant responses."""
	source_class = "badge-local" if "Local" in source else "badge-ai"
	if language == "Hindi":
		source_class = "badge-local" if "लोकल" in source else "badge-ai"
	st.markdown(
		f"""
		<div class="badge-row">
			<span class="badge {source_class}">{source}</span>
			<span class="badge">{'Confidence' if language == 'English' else 'विश्वसनीयता'}: {confidence}</span>
		</div>
		""",
		unsafe_allow_html=True,
	)


def _render_cards(cards: list[dict[str, str]]) -> None:
	"""Render contextual information cards under assistant responses."""
	for card in cards:
		st.markdown(
			f"""
			<div class="info-card">
				<h4>{card['title']}</h4>
				<p>{card['body']}</p>
			</div>
			""",
			unsafe_allow_html=True,
		)


def _render_escalation_form(ui: dict[str, object]) -> None:
	"""Render one-click escalation form for unresolved queries."""
	with st.expander(str(ui["need_help"])):
		st.caption(str(ui["help_caption"]))
		with st.form("escalation_form", clear_on_submit=True):
			name = st.text_input(str(ui["name"]))
			roll_no = st.text_input(str(ui["roll"]))
			issue = st.text_area(
				str(ui["issue"]),
				placeholder=str(ui["issue_placeholder"]),
			)
			submitted = st.form_submit_button(str(ui["submit_btn"]))

		if submitted:
			if not name.strip() or not issue.strip():
				st.warning(str(ui["warning_form"]))
				return

			subject = f"Helpdesk Query - {name.strip()}"
			body = (
				f"Name: {name.strip()}\n"
				f"Roll Number: {roll_no.strip() or 'N/A'}\n"
				f"Issue:\n{issue.strip()}\n\n"
				"Requested via GLA University Helpdesk chatbot."
			)

			mailto_link = (
				"mailto:helpdesk@gla.ac.in?subject="
				f"{quote(subject)}&body={quote(body)}"
			)
			st.success(str(ui["success_mail"]))
			st.markdown(f"[{str(ui['open_mail'])}]({mailto_link})")


def _render_sidebar_analytics(ui: dict[str, object]) -> None:
	"""Display language selection and query analytics in the sidebar."""
	st.sidebar.subheader(str(ui["sidebar_title"]))
	total_queries = st.session_state.analytics["total"]
	local_queries = st.session_state.analytics["local"]
	ai_queries = st.session_state.analytics["ai"]

	local_ratio = (local_queries / total_queries) if total_queries else 0.0
	ai_ratio = (ai_queries / total_queries) if total_queries else 0.0

	st.sidebar.metric(str(ui["sidebar_total"]), str(total_queries))
	st.sidebar.metric(str(ui["sidebar_local"]), f"{local_ratio:.0%}")
	st.sidebar.metric(str(ui["sidebar_ai"]), f"{ai_ratio:.0%}")

	st.sidebar.caption(str(ui["sidebar_topics"]))
	topic_items = sorted(
		st.session_state.analytics["topics"].items(),
		key=lambda item: item[1],
		reverse=True,
	)
	if topic_items:
		for topic, count in topic_items[:5]:
			display_topic = topic.title() if topic != "general" else "General"
			if st.session_state.language == "Hindi":
				mapping = {
					"Admission": "प्रवेश",
					"Fee": "फीस",
					"Hostel": "हॉस्टल",
					"Scholarship": "छात्रवृत्ति",
					"Exam": "परीक्षा",
					"General": "सामान्य",
				}
				display_topic = mapping.get(display_topic, display_topic)
			st.sidebar.write(f"- {display_topic}: {count}")
	else:
		st.sidebar.caption(str(ui["sidebar_empty_topics"]))


# Configure Streamlit page as requested.
st.set_page_config(page_title="GLA University Helpdesk", layout="centered")
_inject_custom_styles()


# Session state keeps chat history across reruns in the same browser session.
if "messages" not in st.session_state:
	st.session_state.messages = []
if "pending_query" not in st.session_state:
	st.session_state.pending_query = ""
if "followups" not in st.session_state:
	st.session_state.followups = []
if "analytics" not in st.session_state:
	st.session_state.analytics = {
		"total": 0,
		"local": 0,
		"ai": 0,
		"topics": {},
	}
if "language" not in st.session_state:
	st.session_state.language = "English"


selected_language = st.sidebar.selectbox(
	"Language",
	options=["English", "Hindi"],
	index=0 if st.session_state.language == "English" else 1,
)
st.session_state.language = selected_language
ui = UI_TEXT[st.session_state.language]

_render_sidebar_analytics(ui)

st.title(str(ui["title"]))
st.caption(str(ui["caption"]))


st.subheader(str(ui["quick_actions"]))
quick_action_columns = st.columns(len(QUICK_ACTIONS))
for index, action in enumerate(QUICK_ACTIONS):
	label = action["label_hi"] if st.session_state.language == "Hindi" else action["label_en"]
	prompt = action["prompt_hi"] if st.session_state.language == "Hindi" else action["prompt_en"]
	if quick_action_columns[index].button(label, use_container_width=True):
		st.session_state.pending_query = prompt
		st.rerun()


# Render the existing conversation.
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])
		if message["role"] == "assistant":
			source = message.get("source", "")
			confidence = message.get("confidence", "")
			cards = message.get("cards", [])
			if source and confidence:
				_render_badges(
					source=source,
					confidence=confidence,
					language=st.session_state.language,
				)
			if cards:
				_render_cards(cards)


if st.session_state.followups:
	st.caption(str(ui["followups"]))
	followup_columns = st.columns(len(st.session_state.followups))
	for index, followup in enumerate(st.session_state.followups):
		if followup_columns[index].button(followup, key=f"followup_{index}"):
			st.session_state.pending_query = followup
			st.rerun()


_render_escalation_form(ui)


typed_query = st.chat_input(str(ui["ask_placeholder"]))
user_query = typed_query or st.session_state.pending_query
if user_query == st.session_state.pending_query:
	st.session_state.pending_query = ""

if user_query:
	# Store and render the user message.
	st.session_state.messages.append({"role": "user", "content": user_query})
	with st.chat_message("user"):
		st.markdown(user_query)

	with st.chat_message("assistant"):
		try:
			local_answer, match_found = get_best_match(user_query, threshold=0.6)
		except Exception:
			# If local engine fails (e.g., CSV issue), continue with API fallback.
			local_answer, match_found = None, False

		if match_found and local_answer:
			final_response = local_answer
			source = str(ui["source_local"])
			confidence = str(ui["conf_high"])
		else:
			with st.spinner(str(ui["thinking"])):
				final_response = get_fallback_ai_response(
					user_query,
					response_language=st.session_state.language,
				)
			source = str(ui["source_ai"])
			confidence = str(ui["conf_check"])

		cards = _build_info_cards(user_query, st.session_state.language)
		_render_badges(
			source=source,
			confidence=confidence,
			language=st.session_state.language,
		)

		_animate_response(final_response, str(ui["typing_cursor"]))
		if cards:
			_render_cards(cards)

	topic = _detect_topic(user_query)
	st.session_state.analytics["total"] += 1
	if "Local" in source or "लोकल" in source:
		st.session_state.analytics["local"] += 1
	else:
		st.session_state.analytics["ai"] += 1

	topic_counts = st.session_state.analytics["topics"]
	topic_counts[topic] = topic_counts.get(topic, 0) + 1

	st.session_state.followups = _suggest_followups(
		user_query,
		final_response,
		st.session_state.language,
	)

	# Persist assistant response in session history.
	st.session_state.messages.append(
		{
			"role": "assistant",
			"content": final_response,
			"source": source,
			"confidence": confidence,
			"cards": cards,
		}
	)
