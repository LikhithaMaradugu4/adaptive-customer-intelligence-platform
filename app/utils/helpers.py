from typing import Iterable, Optional
import re

DOMAIN_KEYWORDS = [
	"order",
	"orders",
	"refund",
	"return",
	"replacement",
	"cancel",
	"cancellation",
	"delivery",
	"shipping",
	"shipment",
	"tracking",
	"payment",
	"charged",
	"charge",
	"billing",
	"invoice",
	"upi",
	"card",
	"account",
	"login",
	"password",
	"security",
	"fraud",
	"dispute",
	"warranty",
	"policy",
	"support",
	"ticket",
	"customer service",
	"shopsphere",
]

RAG_KEYWORDS = [
	"refund",
	"return",
	"replacement",
	"cancellation",
	"policy",
	"warranty",
	"shipping",
	"delivery",
	"tracking",
	"payment",
	"billing",
	"invoice",
	"dispute",
	"account",
	"login",
	"password",
	"security",
	"fraud",
	"support",
	"customer service",
	"ticket",
]

NON_SUPPORT_KEYWORDS = [
	"hello",
	"hi",
	"hey",
	"good morning",
	"good evening",
	"how are you",
	"what's up",
	"thanks",
	"thank you",
	"tell me a joke",
	"joke",
	"cook",
	"recipe",
	"pasta",
	"math",
	"weather",
	"news",
	"sports",
]

SUPPORT_INTENTS = {
	"PAYMENT_ISSUE",
	"REFUND_ISSUE",
	"RETURN_ISSUE",
	"DELIVERY_ISSUE",
	"PRODUCT_ISSUE",
	"ACCOUNT_ISSUE",
	"COMPLAINT",
	"MULTI_INTENT",
}

NON_SUPPORT_INTENTS = {
	"GENERAL_QUERY",
	"NON_SUPPORT",
	"UNKNOWN_INTENT",
}

def _normalize_query(query: str) -> str:
	return (query or "").strip().lower()

def _matches_keywords(query: str, keywords: Iterable[str]) -> bool:
	return any(keyword in query for keyword in keywords)

def is_domain_relevant(
	query: str,
	intents: Optional[Iterable[str]] = None
) -> bool:
	normalized = _normalize_query(query)

	if intents and any(intent in SUPPORT_INTENTS for intent in intents):
		return True

	return _matches_keywords(normalized, DOMAIN_KEYWORDS)

def is_non_support_query(query: str) -> bool:
	normalized = _normalize_query(query)

	for keyword in NON_SUPPORT_KEYWORDS:
		if " " in keyword:
			if keyword in normalized:
				return True
		else:
			if re.search(rf"\b{re.escape(keyword)}\b", normalized):
				return True

	return False

def should_use_rag(
	query: str,
	intent: Optional[Iterable[str]],
	decision: Optional[str] = None,
	confidence: Optional[float] = None
) -> bool:
	normalized = _normalize_query(query)
	intents = list(intent or [])

	if not normalized:
		return False

	if decision in [
		"OUT_OF_SCOPE",
		"CLARIFY"
	]:
		return False

	if any(intent in NON_SUPPORT_INTENTS for intent in intents):
		return False

	if (
		confidence is not None
		and confidence < 0.5
		and not is_domain_relevant(normalized, intents)
	):
		return False

	if (
		is_non_support_query(normalized)
		and not is_domain_relevant(normalized, intents)
	):
		return False

	if not is_domain_relevant(normalized, intents):
		return False

	if any(intent in SUPPORT_INTENTS for intent in intents):
		return True

	if _matches_keywords(normalized, RAG_KEYWORDS):
		return True

	return False

def get_rag_reason(
	query: str,
	intent: Optional[Iterable[str]],
	decision: Optional[str],
	confidence: Optional[float],
	requires_rag: bool
) -> str:
	normalized = _normalize_query(query)
	intents = list(intent or [])

	if not normalized:
		return "empty_query"

	if decision in [
		"OUT_OF_SCOPE",
		"CLARIFY"
	]:
		return "decision_blocks_rag"

	if any(intent in NON_SUPPORT_INTENTS for intent in intents):
		return "non_support_intent"

	if (
		confidence is not None
		and confidence < 0.5
		and not is_domain_relevant(normalized, intents)
	):
		return "low_confidence_non_domain"

	if (
		is_non_support_query(normalized)
		and not is_domain_relevant(normalized, intents)
	):
		return "non_support_query"

	if not is_domain_relevant(normalized, intents):
		return "no_domain_relevance"

	if requires_rag:
		if any(intent in SUPPORT_INTENTS for intent in intents):
			return "support_intent"
		if _matches_keywords(normalized, RAG_KEYWORDS):
			return "policy_or_support_keywords"

	return "rag_not_required"
