"""
query_processor.py
Lightweight intent classification using keyword patterns.
"""
import re
from typing import Tuple

# ── Intent patterns (checked in priority order) ────────────────────────────────
INTENT_PATTERNS = {
    "ESCALATE": [
        r"\blegal\s*action\b", r"\bsue\b", r"\bsuing\b", r"\blawsuit\b",
        r"\bfraud\b", r"\bpolice\b", r"\bscam\b", r"\btheft\b",
        r"\bchargeback\b", r"\bfile\s+a\s+complaint\b", r"\battorney\b",
        r"\brefund\s+denied\b", r"\bblacklist\b",
    ],
    "COMPLAINT": [
        r"\bterrible\b", r"\bawful\b", r"\bworst\b", r"\bunacceptable\b",
        r"\bdisgust\w*\b", r"\bfurious\b", r"\bangry\b", r"\boutrages?\b",
        r"\bhorrible\b", r"\bnever\s+buying\b", r"\bnever\s+shopping\b",
        r"\bvery\s+unhappy\b", r"\bextremely\s+dissatisfied\b",
    ],
    "RETURN_REQUEST": [
        r"\breturn\b", r"\brefund\b", r"\bexchange\b", r"\bsend\s+back\b",
        r"\bmoney\s+back\b", r"\bget\s+my\s+money\b", r"\bcancel\s+order\b",
    ],
    "ORDER_STATUS": [
        r"\border\s+status\b", r"\btrack\b", r"\btracking\b",
        r"\bwhere\s+is\s+my\b", r"\bshipment\b", r"\bdelivered\b",
        r"\bdispatch\b", r"\bshipping\s+update\b", r"\bpackage\b",
    ],
    "PAYMENT": [
        r"\bpayment\b", r"\bbilling\b", r"\bcharge\b", r"\bcharged\b",
        r"\binvoice\b", r"\breceipt\b", r"\bdiscount\s+code\b",
        r"\bpromo\s+code\b", r"\bcoupon\b", r"\bprice\s+match\b",
    ],
    "ACCOUNT": [
        r"\baccount\b", r"\bpassword\b", r"\blogin\b", r"\bsign\s*in\b",
        r"\brewards?\b", r"\bpoints?\b", r"\bprofile\b",
    ],
    "GENERAL_FAQ": [],   # catch-all
}

ESCALATION_KEYWORDS = INTENT_PATTERNS["ESCALATE"] + INTENT_PATTERNS["COMPLAINT"]


def classify_intent(query: str) -> str:
    """Return the most specific intent label for a query."""
    q = query.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return "GENERAL_FAQ"


def has_hard_escalation(query: str) -> bool:
    """True if query contains keywords that require immediate human escalation."""
    q = query.lower()
    for pat in INTENT_PATTERNS["ESCALATE"]:
        if re.search(pat, q):
            return True
    return False


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    Valid queries are non-empty, <= 2000 chars.
    """
    if not query or not query.strip():
        return False, "Query cannot be empty."
    if len(query.strip()) < 3:
        return False, "Query is too short. Please enter a complete question."
    if len(query) > 2000:
        return False, "Query is too long (max 2000 characters)."
    return True, ""
