"""Prototype implementation of the 3-layer grounding safeguard."""

import re

def sanitize_grounded_caption(caption: str, is_portrait_context: bool = False) -> str:
    """Sanitize out-of-frame and proprietary franchise hallucinations while preserving visual details."""
    if not caption or not isinstance(caption, str):
        return ""
    
    text = caption.strip().strip('"').strip("'")
    
    # 1. Clean repeated token loops
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    
    # 2. Check if context is a portrait / headshot / face
    is_portrait = is_portrait_context or bool(
        re.search(r'\b(face|portrait|headshot|head|bust|profile|close[\s-]up)\b', text, re.IGNORECASE)
    )
    
    # 3. If portrait / upper-torso framing, prune unobservable lower-body clothing
    if is_portrait:
        # e.g. "wearing blue shirt and jeans" -> "wearing blue shirt"
        text = re.sub(
            r'(\b(?:wearing|in)\s+[^,]+?)\s+and\s+(?:(?:blue|black|dark|light|faded)\s+)?(?:jeans|pants|trousers|shorts|slacks|sweatpants)\b',
            r'\1',
            text,
            flags=re.IGNORECASE
        )
        # e.g. ", wearing jeans" or " and jeans" at end of clause
        text = re.sub(
            r'[,;\s]+(?:and\s+)?(?:wearing\s+)?(?:(?:blue|black|dark|light|faded)\s+)?(?:jeans|pants|trousers|shorts|slacks|shoes|sneakers|boots)\b',
            '',
            text,
            flags=re.IGNORECASE
        )
        
    # 4. Prune speculative proprietary franchises for generic illustrations / avatars
    text = re.sub(r'\b(?:in|from)\s+the\s+sims(?:\s+game)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthe\s+sims\s+game\b', 'an animated character', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthe\s+sims\s+family\b', 'a family', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfor\s+the\s+disney\s+movie\b', 'in digital illustration style', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdisney\s+movie\b', 'animated style', text, flags=re.IGNORECASE)
    
    # Clean whitespace and trailing punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\s]+|[,\s]+$', '', text)
    return text


# Test cases
test_cases = [
    ("the face of a woman with long hair, wearing blue shirt and jeans", True),
    ("a screenshot of a woman in a blue shirt and jeans", True),
    ("a cartoon character in the sims game", False),
    ("the face of a young girl with long black hair", True),
    ("a man standing on the street in a blue shirt and jeans", False),  # Full body street scene: jeans PRESERVED!
    ("a dog running on grass with a red frisbee", False),             # Benchmark image: PRESERVED!
]

print("TESTING SANITIZATION FUNCTION:")
for orig, is_port in test_cases:
    clean = sanitize_grounded_caption(orig, is_portrait_context=is_port)
    print(f"\nOriginal: {orig}")
    print(f"Cleaned : {clean}")
