"""API handler for fallback generative AI responses."""

from __future__ import annotations

import os

from dotenv import load_dotenv

try:
	import google.generativeai as genai
except ImportError:  # pragma: no cover - depends on environment package state
	genai = None


SYSTEM_PROMPT = (
	"You are a helpful and polite college administration assistant. "
	"Provide concise, accurate guidance for student queries related to "
	"admissions, fees, academics, exams, scholarships, and campus services. "
	"If information is uncertain, clearly mention that the student should "
	"confirm with the official college office."
)


PREFERRED_MODELS = [
	"models/gemini-flash-latest",
	"models/gemini-flash-lite-latest",
	"models/gemini-2.0-flash",
	"models/gemini-2.5-flash",
	"models/gemini-pro-latest",
]


def _resolve_model_candidates() -> list[str]:
	"""Resolve ordered Gemini model candidates.

	Priority:
	1. GEMINI_MODEL environment variable (if set)
	2. PREFERRED_MODELS filtered by API-accessible models
	3. PREFERRED_MODELS defaults
	"""
	override_model = os.getenv("GEMINI_MODEL", "").strip()
	if override_model:
		return [override_model]

	try:
		available_models = {
			model.name
			for model in genai.list_models()
			if "generateContent" in getattr(model, "supported_generation_methods", [])
		}
		candidates = [
			candidate for candidate in PREFERRED_MODELS if candidate in available_models
		]
		if candidates:
			return candidates
	except Exception:
		# If listing models fails, continue with preferred defaults.
		pass

	return PREFERRED_MODELS


def get_fallback_ai_response(user_query: str) -> str:
	"""Generate a fallback answer from a generative AI model.

	The API key is read from a .env file using python-dotenv.
	Supported env variables:
	- GEMINI_API_KEY
	- GOOGLE_API_KEY

	Args:
		user_query: The user's input query.

	Returns:
		AI-generated response text, or a user-friendly error message.
	"""
	if not user_query or not user_query.strip():
		return "Please enter a valid query so I can help you."

	load_dotenv()

	api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
	if not api_key:
		return (
			"AI fallback is not configured. Please add GEMINI_API_KEY "
			"to your .env file."
		)

	if genai is None:
		return (
			"The google-generativeai package is not installed. "
			"Please install dependencies from requirements.txt."
		)

	try:
		genai.configure(api_key=api_key)
		last_error_text = ""
		for model_name in _resolve_model_candidates():
			try:
				model = genai.GenerativeModel(
					model_name=model_name,
					system_instruction=SYSTEM_PROMPT,
				)
				response = model.generate_content(user_query)

				text = (response.text or "").strip() if response else ""
				if text:
					return text
			except Exception as exc:
				last_error_text = str(exc).lower()
				if "api key" in last_error_text or "permission" in last_error_text:
					return (
						"Gemini API authentication failed. Please verify "
						"GEMINI_API_KEY in your .env file."
					)

		if "quota" in last_error_text or "rate limit" in last_error_text:
			return (
				"Gemini API quota is currently exceeded. Please wait and retry, "
				"or switch GEMINI_MODEL in .env to a model with available quota."
			)
		if "not found" in last_error_text or "model" in last_error_text:
			return (
				"No compatible Gemini model is available right now. Set "
				"GEMINI_MODEL in .env, for example: models/gemini-flash-latest"
			)
		return (
			"I could not generate a response at the moment. "
			"Please try again shortly."
		)

	except Exception as exc:
		error_text = str(exc).lower()
		if "api key" in error_text or "permission" in error_text:
			return (
				"Gemini API authentication failed. Please verify GEMINI_API_KEY "
				"in your .env file."
			)
		if "not found" in error_text or "model" in error_text:
			return (
				"The configured Gemini model is unavailable. Set GEMINI_MODEL "
				"in .env to a valid model, such as models/gemini-2.0-flash."
			)
		return (
			"The AI service is currently unavailable. "
			"Please try again later or contact the helpdesk office."
		)
 