"""spaCy-based entity extraction with normalization for dedup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.language import Language

from .schema import SPACY_LABEL_TO_NODE_TYPE


_NOISE_TOKENS = {
    "the", "a", "an", "this", "that", "these", "those",
    "company", "companies", "group", "inc", "ltd", "llc", "corp",
}

_LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|gmbh|s\.?a\.?|plc|co|sa|nv|ag|bv)\b\.?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Entity:
    canonical: str
    node_type: str
    mention: str


@lru_cache(maxsize=1)
def load_model(model_name: str = "en_core_web_sm") -> Language:
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model_name}' not installed. Run: "
            f"python -m spacy download {model_name}"
        ) from e


def normalize(name: str, node_type: str) -> str:
    """Collapse variants of the same entity to one canonical form."""
    s = name.strip().strip(".,;:!?\"'`()[]")
    s = re.sub(r"\s+", " ", s)

    if node_type in {"Organization"}:
        s = _LEGAL_SUFFIX_RE.sub("", s).strip().rstrip(",")
    if node_type == "Location":
        s = s.title()

    return s


def is_noise(text: str) -> bool:
    if len(text) < 2:
        return True
    if text.lower() in _NOISE_TOKENS:
        return True
    if not any(c.isalpha() for c in text):
        return True
    return False


def extract(text: str, nlp: Language | None = None) -> list[Entity]:
    """Extract deduplicated entities from a text."""
    if not text or not text.strip():
        return []
    nlp = nlp or load_model()
    doc = nlp(text)

    seen: dict[tuple[str, str], Entity] = {}
    for ent in doc.ents:
        node_type = SPACY_LABEL_TO_NODE_TYPE.get(ent.label_)
        if not node_type:
            continue
        if is_noise(ent.text):
            continue
        canonical = normalize(ent.text, node_type)
        if not canonical or is_noise(canonical):
            continue
        key = (canonical.lower(), node_type)
        if key not in seen:
            seen[key] = Entity(canonical=canonical, node_type=node_type, mention=ent.text)
    return list(seen.values())
