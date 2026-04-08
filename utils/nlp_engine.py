"""NLP engine for local FAQ matching using TF-IDF and cosine similarity."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FAQMatcher:
	"""Encapsulates the local FAQ retrieval pipeline.

	The class is responsible for:
	1. Loading FAQ data from CSV.
	2. Preprocessing questions with NLTK stopword removal.
	3. Building a TF-IDF representation of FAQ questions.
	4. Returning the best answer match for a user query.
	"""

	def __init__(self, csv_path: Optional[str] = None) -> None:
		"""Initialize the matcher and train vector representations.

		Args:
			csv_path: Optional path to the FAQ CSV file. If not provided,
				defaults to data/college_faq.csv relative to project root.
		"""
		default_path = Path(__file__).resolve().parents[1] / "data" / "college_faq.csv"
		self.csv_path = Path(csv_path) if csv_path else default_path

		self._ensure_nltk_resources()
		self.faq_df = self._load_dataset()
		self.vectorizer = TfidfVectorizer()
		self.question_vectors = self.vectorizer.fit_transform(
			self.faq_df["processed_question"]
		)

	@staticmethod
	def _ensure_nltk_resources() -> None:
		"""Ensure required NLTK corpora are available.

		Downloads stopwords quietly if not already present.
		"""
		try:
			stopwords.words("english")
		except LookupError:
			nltk.download("stopwords", quiet=True)

	def _load_dataset(self) -> pd.DataFrame:
		"""Load and validate the FAQ dataset from CSV.

		Returns:
			A cleaned DataFrame with standardized columns and preprocessed
			question text.

		Raises:
			FileNotFoundError: If the CSV file is missing.
			ValueError: If required columns are not present or data is invalid.
		"""
		if not self.csv_path.exists():
			raise FileNotFoundError(
				f"FAQ dataset not found at: {self.csv_path}. "
				"Please ensure data/college_faq.csv exists."
			)

		faq_df = pd.read_csv(self.csv_path)

		# Normalize column names for resilient matching.
		normalized = {col.strip().lower(): col for col in faq_df.columns}
		if "question" not in normalized or "answer" not in normalized:
			raise ValueError(
				"The FAQ CSV must contain 'Question' and 'Answer' columns."
			)

		question_col = normalized["question"]
		answer_col = normalized["answer"]

		faq_df = faq_df[[question_col, answer_col]].rename(
			columns={question_col: "Question", answer_col: "Answer"}
		)
		faq_df = faq_df.dropna(subset=["Question", "Answer"]).copy()
		faq_df["Question"] = faq_df["Question"].astype(str)
		faq_df["Answer"] = faq_df["Answer"].astype(str)

		faq_df["processed_question"] = faq_df["Question"].apply(self.preprocess_text)
		faq_df = faq_df[faq_df["processed_question"].str.strip().astype(bool)]

		if faq_df.empty:
			raise ValueError("No valid FAQ records were found after preprocessing.")

		return faq_df.reset_index(drop=True)

	@staticmethod
	def preprocess_text(text: str) -> str:
		"""Preprocess text: lowercase, remove punctuation, remove stopwords.

		Args:
			text: Raw text string.

		Returns:
			A cleaned text string suitable for vectorization.
		"""
		stop_words = set(stopwords.words("english"))

		# Lowercase and strip punctuation/special symbols.
		cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
		tokens = cleaned.split()

		# Keep informative tokens only.
		filtered_tokens = [token for token in tokens if token not in stop_words]
		return " ".join(filtered_tokens)

	def get_best_match(
		self, user_query: str, threshold: float = 0.6
	) -> Tuple[Optional[str], bool]:
		"""Find the best FAQ answer for a user query.

		Args:
			user_query: Raw query typed by the user.
			threshold: Minimum cosine similarity score to accept a local match.

		Returns:
			A tuple of (answer, match_found).
			- If matched: (answer_text, True)
			- If not matched: (None, False)
		"""
		if not user_query or not user_query.strip():
			return None, False

		processed_query = self.preprocess_text(user_query)
		if not processed_query:
			return None, False

		query_vector = self.vectorizer.transform([processed_query])
		similarities = cosine_similarity(query_vector, self.question_vectors).flatten()

		if similarities.size == 0:
			return None, False

		best_index = int(similarities.argmax())
		best_score = float(similarities[best_index])

		if best_score >= threshold:
			return self.faq_df.iloc[best_index]["Answer"], True
		return None, False


_MATCHER: Optional[FAQMatcher] = None


def _get_matcher() -> FAQMatcher:
	"""Create and cache the FAQ matcher instance lazily."""
	global _MATCHER
	if _MATCHER is None:
		_MATCHER = FAQMatcher()
	return _MATCHER


def get_best_match(user_query: str, threshold: float = 0.6) -> Tuple[Optional[str], bool]:
	"""Module-level helper to fetch the best local FAQ answer."""
	return _get_matcher().get_best_match(user_query=user_query, threshold=threshold)
