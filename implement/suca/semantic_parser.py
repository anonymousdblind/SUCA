"""
Semantic Unit Parser: Decomposes text prompts into verifiable semantic units.

Each semantic unit represents an atomic, verifiable semantic constraint:
  - Entity existence: "a cat exists"
  - Attribute binding: "the cat is black"
  - Count: "there are three apples"
  - Spatial relation: "the cat is inside the box"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class UnitType(str, Enum):
    ENTITY = "entity"
    ATTRIBUTE = "attribute"
    COUNT = "count"
    RELATION = "relation"


# Unit difficulty for hierarchical reward: higher = harder = more important to evaluate
UNIT_DIFFICULTY = {
    UnitType.ENTITY: 0.3,       # easy — models rarely miss objects entirely
    UnitType.ATTRIBUTE: 0.6,    # medium — color/material binding is error-prone
    UnitType.COUNT: 0.9,        # hard — counting is a known weakness
    UnitType.RELATION: 0.8,     # hard — spatial relations are challenging
}


@dataclass
class SemanticUnit:
    """A single verifiable semantic constraint extracted from a prompt."""

    unit_type: UnitType
    description: str  # human-readable description
    vqa_question: str  # yes/no question for VLM verification
    token_keywords: List[str]  # keywords to locate in tokenized prompt
    token_indices: List[int] = field(default_factory=list)  # filled after tokenization

    @property
    def difficulty(self) -> float:
        return UNIT_DIFFICULTY.get(self.unit_type, 0.5)

    @property
    def is_hard(self) -> bool:
        """Hard units: count, relation, attribute binding — worth evaluating."""
        return self.unit_type in (UnitType.COUNT, UnitType.RELATION, UnitType.ATTRIBUTE)


def filter_units_for_online_reward(
    units: List[SemanticUnit],
    max_units: int = 8,
    min_difficulty: float = 0.5,
) -> List[SemanticUnit]:
    """
    Lightweight unit filtering for online RL training.

    Strategy:
    - Always keep hard units (count, relation, attribute)
    - Keep at most 2 entity units (easy, just for baseline)
    - Cap total at max_units to limit VLM calls
    """
    hard = [u for u in units if u.difficulty >= min_difficulty]
    easy = [u for u in units if u.difficulty < min_difficulty]

    # Keep all hard units, up to max_units
    selected = hard[:max_units]

    # Fill remaining slots with easy units (up to 2)
    remaining = max_units - len(selected)
    if remaining > 0:
        selected.extend(easy[:min(remaining, 2)])

    return selected


def validate_units(units: List[SemanticUnit]) -> List[SemanticUnit]:
    """
    Rule-based validator: filter out malformed or redundant units.

    Checks:
    - Remove empty descriptions/questions
    - Remove duplicate units (same question)
    - Validate relation units have valid entity references
    - Remove units with no token keywords
    - Filter out VQA questions with non-noun objects (verbs, prepositions)
    """
    # Common verbs/prepositions/function words that should never be the "object" in a VQA question
    _bad_objects = {
        "creating", "adding", "showing", "featuring", "casting", "filtering",
        "capturing", "emphasizing", "highlighting", "glowing", "hanging",
        "standing", "sitting", "wearing", "holding", "resting", "leaning",
        "walking", "running", "reading", "playing", "looking", "facing",
        "lined", "scattered", "surrounded", "positioned", "placed",
        "from", "into", "onto", "over", "through", "during", "against",
        "bright", "dark", "light", "almost", "slightly", "partially",
    }

    seen_questions = set()
    valid = []

    for u in units:
        # Skip empty
        if not u.description.strip() or not u.vqa_question.strip():
            continue
        # Skip no keywords
        if not u.token_keywords:
            continue
        # Skip duplicates
        q_key = u.vqa_question.lower().strip()
        if q_key in seen_questions:
            continue

        # Validate: VQA question should not contain bad object words as the target
        # e.g., "Is the from above the creating?" — both "from" and "creating" are bad
        skip = False
        if u.unit_type == UnitType.ATTRIBUTE:
            # Pattern: "Is the {obj} {adj}?" — check obj
            m = re.match(r"Is the (\w+)", u.vqa_question, re.IGNORECASE)
            if m and m.group(1).lower() in _bad_objects:
                skip = True
        elif u.unit_type == UnitType.RELATION:
            # Pattern: "Is the {subj} {rel} the {obj}?" — check both
            for kw in u.token_keywords:
                if kw.lower() in _bad_objects:
                    skip = True
                    break
        elif u.unit_type == UnitType.ENTITY:
            # Filter entities with commas or very long phrases (likely parse errors)
            desc = u.description.replace(" exists", "")
            if "," in desc or len(desc.split()) > 4:
                skip = True

        if skip:
            continue

        seen_questions.add(q_key)
        valid.append(u)

    return valid


PARSER_SYSTEM_PROMPT = """\
You are a semantic parser for text-to-image prompts. Given a prompt, extract all verifiable semantic units.

Each unit is ONE of:
- entity: an object that should exist in the image
- attribute: a property (color, shape, material, size, style) bound to a specific object
- count: a specific number of an object
- relation: a spatial or logical relation between two objects

Output a JSON array. Each element has:
  {"type": "<entity|attribute|count|relation>",
   "description": "<short description>",
   "vqa_question": "<yes/no question to verify this unit in the generated image>",
   "token_keywords": ["<keyword1>", "<keyword2>"]}

Rules:
- vqa_question must be answerable with Yes/No by a VLM looking at the image.
- token_keywords should be the most relevant 1-3 words from the original prompt.
- Be exhaustive: extract ALL units, not just the obvious ones.
- Output ONLY the JSON array, no other text.
"""

PARSER_EXAMPLE = """Prompt: "three red apples and two green pears on a wooden table"

[
  {"type": "entity", "description": "apples exist", "vqa_question": "Are there apples in this image?", "token_keywords": ["apples"]},
  {"type": "entity", "description": "pears exist", "vqa_question": "Are there pears in this image?", "token_keywords": ["pears"]},
  {"type": "attribute", "description": "apples are red", "vqa_question": "Are the apples red?", "token_keywords": ["red", "apples"]},
  {"type": "attribute", "description": "pears are green", "vqa_question": "Are the pears green?", "token_keywords": ["green", "pears"]},
  {"type": "count", "description": "three apples", "vqa_question": "Are there exactly three apples?", "token_keywords": ["three", "apples"]},
  {"type": "count", "description": "two pears", "vqa_question": "Are there exactly two pears?", "token_keywords": ["two", "pears"]},
  {"type": "entity", "description": "wooden table exists", "vqa_question": "Is there a wooden table in this image?", "token_keywords": ["wooden", "table"]},
  {"type": "relation", "description": "fruits on table", "vqa_question": "Are the fruits placed on top of the table?", "token_keywords": ["on", "table"]}
]"""


class SemanticUnitParser:
    """Parse text prompts into semantic units using an LLM or rule-based fallback."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
        use_rules_only: bool = True,
    ):
        self.use_rules_only = use_rules_only
        self.device = device
        self.model = None
        self.tokenizer = None
        self._cache = {}  # prompt → parsed units cache

        if model_name and not use_rules_only:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            self.model.eval()

    def parse(self, prompt: str, online_mode: bool = False) -> List[SemanticUnit]:
        """Parse a single prompt into semantic units.

        Args:
            prompt: Text prompt to parse.
            online_mode: If True, validate + filter to hard units only (for training).
                         If False, return all units (for evaluation).
        """
        # Check cache first
        cache_key = prompt.strip()
        if cache_key in self._cache:
            units = self._cache[cache_key]
        else:
            if self.use_rules_only or self.model is None:
                units = self._rule_based_parse(prompt)
            else:
                units = self._llm_parse(prompt)
            # Validate
            units = validate_units(units)
            # Cache
            self._cache[cache_key] = units

        if online_mode:
            # Lightweight: only hard units, capped at 8
            return filter_units_for_online_reward(units, max_units=8, min_difficulty=0.5)
        return units

    def parse_batch(self, prompts: List[str], online_mode: bool = False) -> List[List[SemanticUnit]]:
        """Parse a batch of prompts."""
        return [self.parse(p, online_mode=online_mode) for p in prompts]

    # ------------------------------------------------------------------
    # LLM-based parsing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _llm_parse(self, prompt: str) -> List[SemanticUnit]:
        messages = [
            {"role": "system", "content": PARSER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Example:\n{PARSER_EXAMPLE}"},
            {"role": "user", "content": f'Prompt: "{prompt}"'},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
        )
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return self._parse_json_output(generated, prompt)

    def _parse_json_output(
        self, text: str, prompt: str
    ) -> List[SemanticUnit]:
        # Extract JSON array from LLM output
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return self._rule_based_parse(prompt)
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return self._rule_based_parse(prompt)

        units: List[SemanticUnit] = []
        for item in items:
            try:
                unit = SemanticUnit(
                    unit_type=UnitType(item["type"]),
                    description=item["description"],
                    vqa_question=item["vqa_question"],
                    token_keywords=item.get("token_keywords", []),
                )
                units.append(unit)
            except (KeyError, ValueError):
                continue
        return units if units else self._rule_based_parse(prompt)

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
    _COLORS = {
        "red", "blue", "green", "yellow", "black", "white", "orange",
        "purple", "pink", "brown", "gray", "grey", "golden", "silver",
        "beige", "tan", "maroon", "navy", "teal", "turquoise", "crimson",
        "ivory", "coral", "magenta", "indigo", "violet", "scarlet",
    }
    _MATERIALS = {
        "wooden", "wood", "metal", "metallic", "glass", "plastic", "stone",
        "paper", "ceramic", "crystal", "leather", "silk", "cotton", "wool",
        "rubber", "concrete", "brick", "marble", "granite", "steel", "iron",
        "copper", "bronze", "brass", "chrome", "aluminum", "clay", "porcelain",
        "fabric", "velvet", "satin", "denim", "linen", "wicker", "bamboo",
    }
    _TEXTURES = {
        "spotted", "striped", "checkered", "plaid", "polka-dot", "floral",
        "sparkling", "glowing", "shiny", "matte", "glossy", "rusty",
        "weathered", "worn", "tattered", "frayed", "smooth", "rough",
        "furry", "fluffy", "fuzzy", "hairy", "feathered", "scaly",
        "translucent", "transparent", "opaque", "reflective", "luminous",
        "frosted", "ornate", "carved", "embroidered", "knitted", "woven",
        "curly", "curved", "straight", "wavy", "pointed", "rounded",
        "colorful", "vibrant", "vivid", "bright", "dark", "pale", "light",
        "rugged", "rocky", "sandy", "muddy", "dusty", "icy", "snowy",
    }
    _SIZES = {
        "tiny", "small", "little", "large", "big", "huge", "enormous",
        "giant", "miniature", "tall", "short", "long", "wide", "narrow",
        "thick", "thin", "slim", "massive", "vast", "compact",
    }
    # Union of all adjective sets for count skip and attribute detection
    _ALL_ADJECTIVES = _COLORS | _MATERIALS | _TEXTURES | _SIZES
    _NUMBERS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    # Multi-word relations must come before single-word ones for correct matching
    _RELATIONS_MULTI = [
        "on top of", "in front of", "to the left of", "to the right of",
        "next to",
    ]
    _RELATIONS_SINGLE = [
        "above", "below", "under", "behind", "beside", "between",
        "near", "inside", "on", "in",
    ]
    # Words that should never be extracted as entity nouns
    _ENTITY_STOPWORDS = {
        "view", "scene", "shot", "image", "photograph", "photo", "picture",
        "illustration", "painting", "drawing", "rendering", "depiction",
        "close-up", "closeup", "backdrop", "background", "foreground",
        "setting", "atmosphere", "mood", "feel", "style", "angle",
        "shades", "hues", "tones", "row", "pair", "group", "set",
        "way", "sense", "mix", "hint", "touch", "burst",
    }

    def _rule_based_parse(self, prompt: str) -> List[SemanticUnit]:
        units: List[SemanticUnit] = []
        prompt_lower = prompt.lower()
        words = prompt_lower.split()

        # --- Entity extraction ---
        nouns = self._extract_noun_phrases(prompt_lower)
        for noun in nouns:
            # Filter out stopword entities ("view", "scene", "shot", etc.)
            noun_words = set(noun.split())
            if noun_words & self._ENTITY_STOPWORDS:
                continue
            # Skip pure adjective phrases (no noun content)
            if all(w.rstrip("s.,;") in self._ALL_ADJECTIVES for w in noun.split()):
                continue
            # Correct grammar: "a X" for singular, "Are there X" for plural
            if noun.endswith("s") and not noun.endswith("ss"):
                q = f"Are there {noun} in this image?"
            else:
                q = f"Is there a {noun} in this image?"
            units.append(SemanticUnit(
                unit_type=UnitType.ENTITY,
                description=f"{noun} exists",
                vqa_question=q,
                token_keywords=[noun.split()[-1]],
            ))

        # --- Attribute detection (colors + materials + textures + sizes) ---
        _skip_words = {"and", "or", "with", "the", "a", "an", "is", "are", "in",
                       "on", "of", "for", "by", "at", "to", "from", "against", ""}
        for i, w in enumerate(words):
            w_clean = w.rstrip("s.,;:)")
            if w_clean in self._ALL_ADJECTIVES and i + 1 < len(words):
                # Find the noun: skip subsequent adjectives
                j = i + 1
                while j < len(words) and words[j].rstrip("s.,;:)") in self._ALL_ADJECTIVES:
                    j += 1
                if j < len(words):
                    obj = words[j].rstrip(".,;:)")
                    # Skip if "obj" is a stopword, conjunction, or preposition
                    if obj in _skip_words:
                        continue
                    if obj in self._ENTITY_STOPWORDS:
                        continue
                    # Skip if obj is itself an adjective (no noun found)
                    if obj in self._ALL_ADJECTIVES:
                        continue
                    units.append(SemanticUnit(
                        unit_type=UnitType.ATTRIBUTE,
                        description=f"{obj} is {w_clean}",
                        vqa_question=f"Is the {obj} {w_clean}?",
                        token_keywords=[w_clean, obj],
                    ))

        # --- Count detection ---
        for i, w in enumerate(words):
            num = self._NUMBERS.get(w)
            if num is None and w.isascii() and w.isdecimal():
                num = int(w)
            if num is not None:
                # Skip adjectives to find the actual noun
                j = i + 1
                while j < len(words) and words[j].rstrip("s.,;:)") in self._ALL_ADJECTIVES:
                    j += 1
                if j < len(words):
                    obj = words[j].rstrip(".,;:)")
                    if obj in {"and", "or", "with", "the", "a", "an", ""}:
                        continue
                    units.append(SemanticUnit(
                        unit_type=UnitType.COUNT,
                        description=f"{num} {obj}",
                        vqa_question=f"Are there exactly {num} {obj}?",
                        token_keywords=[w, obj],
                    ))

        # --- Relation detection with entity-aware VQA questions ---
        # Collect entity nouns for relation question generation
        entity_nouns = [u.token_keywords[-1] for u in units if u.unit_type == UnitType.ENTITY]

        # Multi-word relations first (e.g., "on top of", "next to")
        matched_spans = []  # track matched positions to avoid overlap
        for rel in self._RELATIONS_MULTI:
            pattern = r'\b' + re.escape(rel) + r'\b'
            for m in re.finditer(pattern, prompt_lower):
                subj, obj = self._find_relation_entities(prompt_lower, m.start(), m.end(), rel)
                if subj and obj:
                    units.append(SemanticUnit(
                        unit_type=UnitType.RELATION,
                        description=f"{subj} {rel} {obj}",
                        vqa_question=f"Is the {subj} {rel} the {obj}?",
                        token_keywords=[subj.split()[-1], obj.split()[-1]],
                    ))
                    matched_spans.append((m.start(), m.end()))

        # Single-word relations (with word boundary check)
        for rel in self._RELATIONS_SINGLE:
            pattern = r'\b' + re.escape(rel) + r'\b'
            for m in re.finditer(pattern, prompt_lower):
                # Skip if already covered by a multi-word relation
                if any(s <= m.start() < e for s, e in matched_spans):
                    continue
                subj, obj = self._find_relation_entities(prompt_lower, m.start(), m.end(), rel)
                if subj and obj:
                    units.append(SemanticUnit(
                        unit_type=UnitType.RELATION,
                        description=f"{subj} {rel} {obj}",
                        vqa_question=f"Is the {subj} {rel} the {obj}?",
                        token_keywords=[subj.split()[-1], obj.split()[-1]],
                    ))

        # Deduplicate
        seen = set()
        deduped = []
        for u in units:
            key = u.vqa_question.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(u)

        return deduped if deduped else [SemanticUnit(
            unit_type=UnitType.ENTITY,
            description="overall prompt",
            vqa_question=f"Does this image match the description: {prompt}?",
            token_keywords=prompt.split()[:3],
        )]

    def _find_relation_entities(self, text: str, rel_start: int, rel_end: int, rel: str):
        """Extract subject (before relation) and object (after relation) noun phrases."""
        before = text[:rel_start].strip().rstrip(",;: ")
        after = text[rel_end:].strip().lstrip(",;: ")

        # Extract last noun phrase from 'before' as subject
        subj = self._last_noun_from_text(before)
        # Extract first noun phrase from 'after' as object
        obj = self._first_noun_from_text(after)

        return subj, obj

    # Words that cannot be a relation subject/object
    _RELATION_SKIP = {
        "the", "a", "an", "and", "or", "with", "its", "their", "",
        "up", "down", "out", "off", "away", "back", "over", "around",
        "it", "this", "that", "them", "here", "there", "where", "which",
        "not", "no", "very", "also", "just", "then", "so", "yet",
    }

    def _last_noun_from_text(self, text: str) -> Optional[str]:
        """Extract the last noun (with optional preceding adjective) from text."""
        words = text.split()
        if not words:
            return None
        # Walk backwards: skip stopwords, function words, adjectives-only
        i = len(words) - 1
        while i >= 0 and words[i].rstrip("s.,;:)") in self._RELATION_SKIP:
            i -= 1
        if i < 0:
            return None
        noun = words[i].rstrip("s.,;:)")
        if noun in self._ENTITY_STOPWORDS or not noun:
            return None
        # Skip if noun is purely an adjective (not a real noun)
        if noun in self._ALL_ADJECTIVES:
            # Try one more word back
            i -= 1
            while i >= 0 and words[i].rstrip("s.,;:)") in self._RELATION_SKIP:
                i -= 1
            if i < 0:
                return None
            adj = noun
            noun = words[i].rstrip("s.,;:)")
            if noun in self._ENTITY_STOPWORDS or noun in self._RELATION_SKIP or not noun:
                return None
            return f"{adj} {noun}"
        # Optionally include one preceding adjective
        if i > 0 and words[i-1].rstrip("s.,;:)") in self._ALL_ADJECTIVES:
            return f"{words[i-1].rstrip('s.,;:)')} {noun}"
        return noun

    def _first_noun_from_text(self, text: str) -> Optional[str]:
        """Extract the first noun (with optional preceding adjective) from text."""
        words = text.split()
        if not words:
            return None
        # Skip articles
        i = 0
        while i < len(words) and words[i].rstrip("s.,;:)") in {"the", "a", "an", ""}:
            i += 1
        if i >= len(words):
            return None
        # Skip all adjectives to reach the actual noun
        last_adj = None
        while i < len(words) and words[i].rstrip("s.,;:)") in self._ALL_ADJECTIVES:
            last_adj = words[i].rstrip("s.,;:)")
            i += 1
        if i >= len(words):
            return None
        noun = words[i].rstrip("s.,;:)")
        if noun in self._ENTITY_STOPWORDS or not noun:
            return None
        if last_adj:
            return f"{last_adj} {noun}"
        return noun

    @staticmethod
    def _extract_noun_phrases(text: str) -> List[str]:
        """Extract noun phrases by splitting on conjunctions, prepositions, and relations."""
        # Split on articles, conjunctions, prepositions, and common relation words
        split_pattern = (
            r"\b(?:and|or|but|with|on|on top of|in|in front of|inside|of|the|a|an"
            r"|next to|to the left of|to the right of|above|below|under|behind"
            r"|beside|between|near|featuring|creating|adding|showing|including"
            r"|where|while|that|which|is|are|was|were|has|have|from|for|by|at"
            r"|into|onto|over|through|during|against)\b"
        )
        parts = re.split(split_pattern, text)
        nouns = []
        for part in parts:
            part = part.strip().strip(".,;:()\"'")
            if not part or len(part) <= 1:
                continue
            ws = [w for w in part.split() if w.strip(".,;:()\"'")]
            if not ws:
                continue
            # Take last 1-2 words as noun phrase, skip leading adjectives for the phrase
            if len(ws) >= 2:
                nouns.append(" ".join(ws[-2:]))
            else:
                nouns.append(ws[-1])
        return nouns

    def resolve_token_indices(
        self, units: List[SemanticUnit], prompt: str, tokenizer
    ) -> List[SemanticUnit]:
        """Map each unit's token_keywords to actual token indices in the prompt."""
        tokens = tokenizer.tokenize(prompt)
        token_texts = [t.replace("</w>", "").replace("Ġ", "").lower() for t in tokens]

        for unit in units:
            indices = []
            for kw in unit.token_keywords:
                kw_lower = kw.lower()
                for idx, tok_text in enumerate(token_texts):
                    if kw_lower in tok_text or tok_text in kw_lower:
                        indices.append(idx)
            unit.token_indices = sorted(set(indices))
            # Fallback: if no indices found, use all tokens
            if not unit.token_indices:
                unit.token_indices = list(range(len(tokens)))
        return units
