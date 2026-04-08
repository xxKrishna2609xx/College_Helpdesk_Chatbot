"""Streamlit app for a hybrid college helpdesk chatbot.

Flow:
1. Try local FAQ match using TF-IDF + cosine similarity.
2. If local match confidence is low, fall back to a generative AI API.
"""

from __future__ import annotations

from urllib.parse import quote

import streamlit as st

from utils.api_handler import get_fallback_ai_response
from utils.nlp_engine import get_best_match


QUICK_ACTIONS = [
	("Admissions", "What is the admission process?"),
	("Fees", "What is the fee structure and payment schedule?"),
	("Hostel", "What are the hostel facilities and charges?"),
	("Scholarship", "How can I apply for scholarships?"),
	("Exam Form", "When does exam form submission start?"),
]


FOLLOW_UP_MAP = {
	"admission": [
		"What documents are required for admission?",
		"What is the admission deadline?",
		"Is there any entrance test?",
	],
	"fee": [
		"Are there installment options for fees?",
		"How can I pay fees online?",
		"Is there any late fee penalty?",
	],
	"hostel": [
		"What are hostel room types?",
		"Are mess charges included?",
		"How do I apply for hostel allocation?",
	],
	"scholarship": [
		"Who is eligible for scholarships?",
		"What is the scholarship application process?",
		"When are scholarship results announced?",
	],
	"exam": [
		"What is the exam form fee?",
		"How can I download my admit card?",
		"What if I miss exam form submission?",
	],
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


def _build_info_cards(user_query: str) -> list[dict[str, str]]:
	"""Return lightweight topic cards relevant to the current query."""
	query = user_query.lower()
	if any(keyword in query for keyword in ["admission", "apply", "enroll"]):
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

	if any(keyword in query for keyword in ["fee", "payment", "tuition"]):
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

	if any(keyword in query for keyword in ["hostel", "mess", "accommodation"]):
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

	if any(keyword in query for keyword in ["scholarship", "financial aid"]):
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

	if any(keyword in query for keyword in ["exam", "admit card", "backlog"]):
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


def _suggest_followups(user_query: str, bot_response: str) -> list[str]:
	"""Suggest follow-up questions based on user context."""
	combined = f"{user_query} {bot_response}".lower()
	for topic, suggestions in FOLLOW_UP_MAP.items():
		if topic in combined:
			return suggestions

	return [
		"What are the office timings for student helpdesk?",
		"Who should I contact for urgent academic issues?",
		"Can you share official email or phone details?",
	]


def _render_badges(source: str, confidence: str) -> None:
	"""Render source and confidence badges for assistant responses."""
	source_class = "badge-local" if "Local" in source else "badge-ai"
	st.markdown(
		f"""
		<div class="badge-row">
			<span class="badge {source_class}">{source}</span>
			<span class="badge">Confidence: {confidence}</span>
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


def _render_escalation_form() -> None:
	"""Render one-click escalation form for unresolved queries."""
	with st.expander("Need help from office staff?"):
		st.caption("Submit details to prepare an email draft for the helpdesk office.")
		with st.form("escalation_form", clear_on_submit=True):
			name = st.text_input("Name")
			roll_no = st.text_input("Roll Number")
			issue = st.text_area(
				"Issue Summary",
				placeholder="Describe your unresolved query here...",
			)
			submitted = st.form_submit_button("Create Helpdesk Email")

		if submitted:
			if not name.strip() or not issue.strip():
				st.warning("Please enter at least your name and issue summary.")
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
			st.success("Email draft prepared. Open your mail app using the link below.")
			st.markdown(f"[Open Email Draft]({mailto_link})")


# Configure Streamlit page as requested.
st.set_page_config(page_title="GLA University Helpdesk", layout="centered")
_inject_custom_styles()
st.title("GLA University Helpdesk")
st.caption("Hybrid chatbot: local FAQ matching + AI fallback")


# Session state keeps chat history across reruns in the same browser session.
if "messages" not in st.session_state:
	st.session_state.messages = []
if "pending_query" not in st.session_state:
	st.session_state.pending_query = ""
if "followups" not in st.session_state:
	st.session_state.followups = []


st.subheader("Quick Actions")
quick_action_columns = st.columns(len(QUICK_ACTIONS))
for index, (label, prompt) in enumerate(QUICK_ACTIONS):
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
				_render_badges(source=source, confidence=confidence)
			if cards:
				_render_cards(cards)


if st.session_state.followups:
	st.caption("Suggested follow-up questions")
	followup_columns = st.columns(len(st.session_state.followups))
	for index, followup in enumerate(st.session_state.followups):
		if followup_columns[index].button(followup, key=f"followup_{index}"):
			st.session_state.pending_query = followup
			st.rerun()


_render_escalation_form()


typed_query = st.chat_input("Ask your college-related question...")
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
			source = "Local FAQ Match"
			confidence = "High"
		else:
			with st.spinner("Thinking..."):
				final_response = get_fallback_ai_response(user_query)
			source = "AI Response"
			confidence = "Needs Verification"

		cards = _build_info_cards(user_query)
		_render_badges(source=source, confidence=confidence)

		st.markdown(final_response)
		if cards:
			_render_cards(cards)

	st.session_state.followups = _suggest_followups(user_query, final_response)

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
