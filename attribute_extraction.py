import re
from typing import Optional


HEBREW_ENGLISH_ALIASES = {
    "סמסונג": "samsung",
    "גלקסי": "galaxy",
    "אייפון": "iphone",
    "אפל": "apple",
    "איירפודס": "airpods",
    "שיאומי": "xiaomi",
    "שיואמי": "xiaomi",
    "רדמי": "redmi",
    "נוט": "note",
    "לנובו": "lenovo",
    "אידיאפד": "ideapad",
    "סוני": "sony",
    "אולטרה": "ultra",
    "פרו": "pro",
    "פלוס": "plus",
    "מחודש": "refurbished",
    "מחודשת": "refurbished",
    "משומש": "used",
    "חדש": "new",
    "גיגה": "gb",
    "דור": "gen",
    "ווטש": "watch",
    "סוויץ": "switch",
    "לבן": "white",
    "שחור": "black",
    "מנטה": "mint",
    "טבעי": "natural",
    "טיטניום": "titanium",
    "סלולר": "cellular",
    "מידנייט": "midnight",
    "כסוף": "silver",
}

BRAND_PATTERNS = {
    "samsung": [r"\bsamsung\b"],
    "apple": [r"\bapple\b", r"\biphone\b", r"\bairpods\b", r"\bwatch\b"],
    "xiaomi": [r"\bxiaomi\b", r"\bredmi\b"],
    "lenovo": [r"\blenovo\b", r"\bideapad\b"],
    "sony": [r"\bsony\b"],
    "nintendo": [r"\bnintendo\b", r"\bswitch\b"],
}

FAMILY_PATTERNS = {
    "galaxy": [r"\bgalaxy\b"],
    "iphone": [r"\biphone\b", r"\biphone\d{2}\b"],
    "airpods": [r"\bairpods\b"],
    "redmi_note": [r"\bredmi\s*note\b", r"\bredmi\s*note\d+\b"],
    "ideapad": [r"\bideapad\b"],
    "watch": [r"\bwatch\b"],
    "switch": [r"\bswitch\b"],
    "wh1000x": [r"\bwh1000xm[345]\b", r"\bwh\s*1000\s*xm[345]\b"],
}

TIER_VARIANT_PATTERNS = {
    "ultra": [r"\bultra\b"],
    "pro": [r"\bpro\b"],
    "plus": [r"\bplus\b"],
    "fe": [r"\bfe\b"],
}

CONNECTIVITY_PATTERNS = {
    "gps_cellular": [
        r"\bgps\s*(and|&|\+)?\s*cellular\b",
        r"\bcellular\s*(and|&|\+)?\s*gps\b",
    ],
    "cellular": [r"\bcellular\b"],
    "gps": [r"\bgps\b"],
}

CONDITION_PATTERNS = {
    "refurbished": [r"\brefurbished\b", r"\brenewed\b"],
    "used": [r"\bused\b"],
    "new": [r"\bnew\b"],
}

COLOR_PATTERNS = {
    "black": [r"\bblack\b"],
    "white": [r"\bwhite\b"],
    "mint": [r"\bmint\b"],
    "midnight": [r"\bmidnight\b"],
    "natural": [r"\bnatural\b"],
    "titanium": [r"\btitanium\b"],
    "silver": [r"\bsilver\b"],
}


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()

    for he, en in HEBREW_ENGLISH_ALIASES.items():
        text = re.sub(rf"\b{re.escape(he)}\b", en, text)

    text = re.sub(r"(\d+)\s*(gb|g|gig|gigabyte)\b", r"\1gb", text)
    text = re.sub(r"(\d+)\s*tb\b", lambda m: f"{int(m.group(1)) * 1024}gb", text)
    text = re.sub(r"(\d+)\s*ssd\b", r"\1ssd", text)
    text = re.sub(r"(\d+)\s*ram\b", r"\1gb ram", text)
    text = re.sub(r"(\d+)\s*ממ\b", r"\1mm", text)
    text = re.sub(r"(\d+)\s*inch\b", r"\1 inch", text)

    text = re.sub(r"\b1st\s*gen\b", "gen1", text)
    text = re.sub(r"\b2nd\s*gen\b", "gen2", text)
    text = re.sub(r"\b3rd\s*gen\b", "gen3", text)
    text = re.sub(r"\bgen\s+(\d)\b", r"gen\1", text)

    text = re.sub(r"\biphone\s+(11|12|13|14|15|16)\b", r"iphone \1", text)
    text = re.sub(r"\bnote\s+(\d{1,2})\b", r"note \1", text)
    text = re.sub(r"\bseries\s+(\d{1,2})\b", r"series \1", text)
    text = re.sub(r"\bwh[\-\s]?1000[\-\s]?xm([345])\b", r"wh1000xm\1", text)

    text = re.sub(r"[\-\+\(\),/|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_by_patterns(text: str, patterns_map: dict[str, list[str]]) -> Optional[str]:
    for value, patterns in patterns_map.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return value
    return None


def extract_brand(text: str) -> Optional[str]:
    return _extract_by_patterns(text, BRAND_PATTERNS)


def extract_model_family(text: str) -> Optional[str]:
    return _extract_by_patterns(text, FAMILY_PATTERNS)


def extract_tier_variant(text: str) -> Optional[str]:
    for value in ["ultra", "pro", "plus", "fe"]:
        for pattern in TIER_VARIANT_PATTERNS[value]:
            if re.search(pattern, text):
                return value
    return None


def extract_display_variant(text: str) -> Optional[str]:
    if re.search(r"\boled\b", text):
        return "oled"
    if re.search(r"\bswitch\b", text):
        return "standard"
    return None


def extract_connectivity(text: str) -> Optional[str]:
    if re.search(r"\bgps\b", text) and re.search(r"\bcellular\b", text):
        return "gps_cellular"
    return _extract_by_patterns(text, CONNECTIVITY_PATTERNS)


def extract_condition(text: str) -> Optional[str]:
    if re.search(r"\brefurbished\b", text):
        return "refurbished"
    if re.search(r"\bused\b", text):
        return "used"
    if re.search(r"\bnew\b", text):
        return "new"
    return None


def extract_storage(text: str) -> Optional[str]:
    matches = re.findall(r"\b(32|64|128|256|512|1024|2048)gb\b", text)
    ssd_matches = re.findall(r"\b(128|256|512|1024|2048)ssd\b", text)
    matches.extend(ssd_matches)

    if not matches:
        return None

    nums = sorted(int(x) for x in matches)
    return f"{nums[-1]}gb"


def extract_ram(text: str) -> Optional[str]:
    match = re.search(r"\b(4|8|16|24|32|64)gb\s+ram\b", text)
    if match:
        return f"{match.group(1)}gb"

    match = re.search(
        r"\b(4|8|16|24|32|64)gb\b(?=.*\b(?:128ssd|256ssd|512ssd|1024ssd|128gb|256gb|512gb|1024gb)\b)",
        text
    )
    if match:
        return f"{match.group(1)}gb"

    return None


def extract_screen_size(text: str) -> Optional[str]:
    match = re.search(r"\b(13|14|15|16|17)(?:\.3)?\s*inch\b", text)
    if match:
        return match.group(1)

    match = re.search(r"\b(13|14|15|16|17)\b(?=.*\b(?:8gb|16gb|512ssd|512gb)\b)", text)
    if match:
        return match.group(1)

    match = re.search(r"\b(41|45)mm\b", text)
    if match:
        return match.group(1)

    return None


def extract_model_number(text: str, model_family: Optional[str]) -> Optional[str]:
    if model_family == "galaxy":
        match = re.search(r"\b(s\d{2})\b", text)
        if match:
            return match.group(1)
        match = re.search(r"\b(a\d{2})\b", text)
        if match:
            return match.group(1)

    if model_family == "iphone":
        match = re.search(r"\biphone\s(11|12|13|14|15|16)\b", text)
        if match:
            return match.group(1)
        match = re.search(r"\biphone(11|12|13|14|15|16)\b", text)
        if match:
            return match.group(1)

    if model_family == "airpods":
        match = re.search(r"\bgen([123])\b", text)
        if match:
            return f"gen{match.group(1)}"

    if model_family == "redmi_note":
        match = re.search(r"\bnote\s(\d{1,2})\b", text)
        if match:
            return match.group(1)
        match = re.search(r"\bnote(\d{1,2})\b", text)
        if match:
            return match.group(1)

    if model_family == "ideapad":
        match = re.search(r"\bideapad\s+(\d)\b", text)
        if match:
            return match.group(1)

    if model_family == "watch":
        match = re.search(r"\bseries\s(\d{1,2})\b", text)
        if match:
            return match.group(1)
        match = re.search(r"\bwatch\s(7|8|9|10)\b", text)
        if match:
            return match.group(1)

    if model_family == "wh1000x":
        match = re.search(r"\bwh1000xm([345])\b", text)
        if match:
            return f"xm{match.group(1)}"

    return None


def extract_color(text: str) -> Optional[str]:
    return _extract_by_patterns(text, COLOR_PATTERNS)


def infer_category(model_family: Optional[str]) -> Optional[str]:
    mapping = {
        "galaxy": "phone",
        "iphone": "phone",
        "redmi_note": "phone",
        "airpods": "earbuds",
        "ideapad": "laptop",
        "watch": "watch",
        "switch": "console",
        "wh1000x": "headphones",
    }
    return mapping.get(model_family)


def confidence(value: Optional[str]) -> str:
    return "high" if value is not None else "missing"


def extract_attributes(text: str) -> dict:
    normalized = normalize_text(text)

    brand = extract_brand(normalized)
    model_family = extract_model_family(normalized)
    model_number = extract_model_number(normalized, model_family)
    tier_variant = extract_tier_variant(normalized)
    display_variant = extract_display_variant(normalized)
    connectivity = extract_connectivity(normalized)
    condition = extract_condition(normalized)
    storage = extract_storage(normalized)
    ram = extract_ram(normalized)
    screen_size = extract_screen_size(normalized)
    color = extract_color(normalized)
    category = infer_category(model_family)

    return {
        "normalized_title": normalized,
        "brand": brand,
        "model_family": model_family,
        "model_number": model_number,
        "tier_variant": tier_variant,
        "display_variant": display_variant,
        "connectivity": connectivity,
        "condition": condition,
        "storage": storage,
        "ram": ram,
        "screen_size": screen_size,
        "color": color,
        "category": category,
        "brand_confidence": confidence(brand),
        "model_family_confidence": confidence(model_family),
        "model_number_confidence": confidence(model_number),
        "tier_variant_confidence": confidence(tier_variant),
        "display_variant_confidence": confidence(display_variant),
        "connectivity_confidence": confidence(connectivity),
        "condition_confidence": confidence(condition),
        "storage_confidence": confidence(storage),
        "ram_confidence": confidence(ram),
        "screen_size_confidence": confidence(screen_size),
        "color_confidence": confidence(color),
    }