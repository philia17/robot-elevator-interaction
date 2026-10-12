"""Deterministic Chinese/English parser for the controlled intent set."""

from __future__ import annotations

import re

from voice_control.intents import Intent, ParseResult

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_FLOOR_TOKEN = re.compile(
    r"(?:(?P<basement>[Bb])(?:(?P<bdigit>\d+)|"
    r"(?P<bchinese>[零〇一二两三四五六七八九十]+))(?:楼|层|F)?|"
    r"(?:(?P<arabic>-?\d+)|"
    r"(?P<chinese>(?:地下|负)?[零〇一二两三四五六七八九十]+))\s*(?:楼|层|F))",
    re.IGNORECASE,
)

_STOP = (r"紧急停止", r"停止", r"停下", r"别动", r"^停$", r"\bstop\b")
_CANCEL = (r"取消", r"不要了", r"算了", r"不用了")
_STATUS = (r"状态", r"现在怎么样", r"做到哪")
_WAKE = (r"^(?:小[艺易一]){2}$",)
_BARE_CONFIRM = (r"^确认(?:一下)?$", r"^确定$", r"^(?:好的?|是的?|对|可以|行|没问题|ok|okay)$")
_NEGATED_FLOOR = (r"不要(?:去|按)?", r"别(?:去|按)?", r"不去", r"不想去", r"无需(?:去|按)?")
_SELF_DIRECTED_ACTION = (
    r"我(?:(?:现在|还是|又|就|其实|真的|重新|突然))*(?:想|要)"
)
_FLOOR_ACTION_PREFIX = re.compile(
    rf"^(?:请|麻烦)?(?:(?:帮我|{_SELF_DIRECTED_ACTION}|想|直接|带我|送我)?"
    r"(?:去|到|按|前往|选择)(?:一下)?)?$"
)
_FLOOR_CHANGE_PREFIX = re.compile(
    r"^(?:请|麻烦)?(?:我想|我要)?"
    r"(?:改成|改为|改去|换成|换到|换去|还是去|还是到)"
    r"(?:直接)?(?:去|到|按)?$"
)
_FLOOR_POLITE_SUFFIX = re.compile(r"^(?:吧|呀|啊|谢谢|谢谢你|麻烦了)?$")
_EMBEDDED_FLOOR_ACTION_SUFFIX = re.compile(
    r"(?:"
    rf"{_SELF_DIRECTED_ACTION}(?:直接)?(?:去|到|按|前往|选择)(?:一下)?|"
    r"帮我(?:直接)?(?:去|到|按|选择)(?:一下)?|"
    r"带我(?:直接)?(?:去|到)|送我(?:直接)?(?:去|到)|给我按|"
    r"我改主意了(?:我想|我要)?(?:直接)?(?:去|到|按)?|"
    r"(?:改成|改为|改去|换成|换到|换去|还是去|还是到)"
    r"(?:直接)?(?:去|到|按)?"
    r")$"
)
_CORRECTED_SOURCE_SUFFIX = re.compile(r"(?:不是|不去|不要去|别去|不想去)$")
_CORRECTED_TARGET_CONNECTOR = re.compile(
    rf"^(?:而是|是|{_SELF_DIRECTED_ACTION}(?:直接)?(?:去|到|按)|"
    r"(?:改成|改为|改去|换成|换到|换去)(?:直接)?(?:去|到|按)?)$"
)
_CHANGE_TARGET_CONNECTOR = re.compile(
    r"^(?:改成|改为|改去|换成|换到|换去)(?:直接)?(?:去|到|按)?$"
)


def _english_number_words() -> tuple[dict[str, int], dict[str, int]]:
    cardinals = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40,
    }
    for tens_word, tens_value in (("twenty", 20), ("thirty", 30)):
        for unit_word, unit_value in list(cardinals.items())[1:10]:
            cardinals[f"{tens_word} {unit_word}"] = tens_value + unit_value
    ordinals = {
        "first": 1, "second": 2, "third": 3, "fourth": 4,
        "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
        "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
        "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
        "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
        "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
        "fortieth": 40,
    }
    ordinal_units = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
    }
    for tens_word, tens_value in (("twenty", 20), ("thirty", 30)):
        for unit_value, unit_word in ordinal_units.items():
            ordinals[f"{tens_word} {unit_word}"] = tens_value + unit_value
    return cardinals, ordinals


_EN_CARDINALS, _EN_ORDINALS = _english_number_words()
_EN_NUMBER_WORD_PATTERN = "|".join(
    re.escape(word)
    for word in sorted((*_EN_CARDINALS, *_EN_ORDINALS), key=len, reverse=True)
)
_EN_NUMBER_PATTERN = rf"(?:-?\d+(?:st|nd|rd|th)?|{_EN_NUMBER_WORD_PATTERN})"
_EN_FLOOR_TOKEN = re.compile(
    rf"\bb(?P<bcode>\d+)\b|"
    rf"\bbasement(?: level)? (?P<basement>{_EN_NUMBER_PATTERN})\b|"
    rf"\b(?:floor|level) (?P<prefix>{_EN_NUMBER_PATTERN})\b|"
    rf"\b(?P<suffix>{_EN_NUMBER_PATTERN}) (?:floor|level)\b",
)
_EN_ACTION_SUFFIX = re.compile(
    r"(?:go to(?: the)?|press|select|choose|take me to(?: the)?|"
    r"i (?:(?:now|actually|really) )?(?:want|would like|d like) to go to(?: the)?|"
    r"change(?: it)? to|switch(?: it)? to|make it|"
    r"actually(?: go to)?(?: the)?|instead(?: go to)?(?: the)?)$"
)
_EN_POLITE_SUFFIX = re.compile(r"^(?:please|thanks|thank you|now)?$")


def response_language(text: str) -> str:
    """Return the response language selected by a wake utterance."""
    return "en" if _is_english_wake(text) else "zh"


def parse_utterance(
        text: str,
        *,
        floor_min: int = -2,
        floor_max: int = 40,
        allow_english_no_floor_correction: bool = False,
) -> ParseResult:
    raw = (text or "").strip()
    if not raw:
        return ParseResult(Intent.UNKNOWN, raw_text=raw, reject_reason="empty")
    normalized = re.sub(r"[\s，。！？、,.!?]", "", raw).lower()
    english = _normalize_english(raw)

    # STOP remains highest priority in either language.
    if re.search(r"\b(?:stop|halt)\b", english):
        return ParseResult(Intent.STOP, raw_text=raw)
    if _is_english_wake(raw):
        return ParseResult(Intent.WAKE, raw_text=raw)
    if re.search(r"[a-z]{2,}", english):
        return _parse_english(
            raw,
            english,
            floor_min=floor_min,
            floor_max=floor_max,
            allow_no_floor_correction=allow_english_no_floor_correction,
        )

    # Safety rule: STOP wins even if the same utterance contains another intent.
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _STOP):
        return ParseResult(Intent.STOP, raw_text=raw)

    floor_matches = list(_FLOOR_TOKEN.finditer(normalized))
    floors, malformed = _extract_floors(normalized)
    unique_floors = sorted(set(floors))
    if any(not floor_min <= candidate <= floor_max for candidate in unique_floors):
        return ParseResult(Intent.UNKNOWN, raw_text=raw, reject_reason="floor_out_of_range")

    # Extract the final, explicit user-directed floor action from conversational
    # framing instead of enumerating every possible lead-in ("等下", "不对",
    # "现在", etc.).  STOP still wins above, and a floor request still requires
    # an action such as "我想去", "帮我按", "改成", or "换到".
    corrected_floor = _corrected_floor_target(normalized, floor_matches)
    if corrected_floor is not None:
        return ParseResult(Intent.PRESS_FLOOR, floor=corrected_floor, raw_text=raw)

    if len(unique_floors) > 1:
        return ParseResult(Intent.UNKNOWN, raw_text=raw, reject_reason="ambiguous_floor")
    floor = unique_floors[0] if unique_floors else None
    explicit_floor_request = (
        floor is not None
        and not any(re.search(pattern, normalized) for pattern in _NEGATED_FLOOR)
        and (_looks_like_floor_command(normalized)
             or _has_embedded_floor_action(normalized, floor_matches))
    )
    if explicit_floor_request:
        return ParseResult(Intent.PRESS_FLOOR, floor=floor, raw_text=raw)

    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _CANCEL):
        return ParseResult(Intent.CANCEL, raw_text=raw)
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _STATUS):
        return ParseResult(Intent.STATUS, raw_text=raw)
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _WAKE):
        return ParseResult(Intent.WAKE, raw_text=raw)

    # Spoken confirmation must repeat the floor. The controller then binds it
    # to the pending command_id and TTL; a bare "确认" is deliberately unsafe.
    if normalized.startswith(("确认", "确定")):
        if floor is None:
            return ParseResult(Intent.UNKNOWN, raw_text=raw,
                               reject_reason="confirmation_missing_floor")
        return ParseResult(Intent.CONFIRM, floor=floor, raw_text=raw)
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _BARE_CONFIRM):
        return ParseResult(Intent.UNKNOWN, raw_text=raw,
                           reject_reason="confirmation_missing_floor")

    if malformed:
        return ParseResult(Intent.UNKNOWN, raw_text=raw, reject_reason="invalid_floor")
    if floor is not None:
        if any(re.search(pattern, normalized) for pattern in _NEGATED_FLOOR):
            return ParseResult(Intent.UNKNOWN, raw_text=raw,
                               reject_reason="negated_floor_command")
        return ParseResult(Intent.UNKNOWN, raw_text=raw,
                           reject_reason="unsafe_floor_context")
    return ParseResult(Intent.UNKNOWN, raw_text=raw, reject_reason="unrecognized")


def _normalize_english(text: str) -> str:
    lowered = text.lower().replace("’", "'").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9'\-]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _is_english_wake(text: str) -> bool:
    key = re.sub(r"[^a-z0-9嘿]+", "", text.lower())
    key = key.replace("嘿", "hey")
    key = re.sub(r"^heyy+", "hey", key)
    return key in {"heysiri", "heysri"}


def _parse_english(
        raw: str,
        text: str,
        *,
        floor_min: int,
        floor_max: int,
        allow_no_floor_correction: bool,
) -> ParseResult:
    text = _normalize_english_floor_artifacts(text)
    matches = list(_EN_FLOOR_TOKEN.finditer(text))
    floors = [_english_floor_value(match) for match in matches]
    floors = [floor for floor in floors if floor is not None]
    unique = sorted(set(floors))
    if any(not floor_min <= floor <= floor_max for floor in unique):
        return ParseResult(Intent.UNKNOWN, raw_text=raw, reject_reason="floor_out_of_range")

    if len(unique) > 1:
        corrected = _english_corrected_target(text, matches)
        if corrected is None:
            return ParseResult(Intent.UNKNOWN, raw_text=raw,
                               reject_reason="ambiguous_floor")
        return ParseResult(Intent.PRESS_FLOOR, floor=corrected, raw_text=raw)
    floor = unique[0] if unique else None

    if re.search(r"\bconfirm\b", text):
        if floor is None:
            return ParseResult(Intent.UNKNOWN, raw_text=raw,
                               reject_reason="confirmation_missing_floor")
        return ParseResult(Intent.CONFIRM, floor=floor, raw_text=raw)

    if (
        allow_no_floor_correction
        and floor is not None
        and _is_english_no_floor_correction(text, matches)
    ):
        return ParseResult(Intent.PRESS_FLOOR, floor=floor, raw_text=raw)

    if floor is not None and _looks_like_english_floor_command(text, matches):
        return ParseResult(Intent.PRESS_FLOOR, floor=floor, raw_text=raw)

    if re.search(r"\b(?:cancel|never mind|forget it)\b", text):
        return ParseResult(Intent.CANCEL, raw_text=raw)
    if re.search(r"\b(?:status|current status|how is it going)\b", text):
        return ParseResult(Intent.STATUS, raw_text=raw)
    # English sessions allow one deliberately narrow contextual confirmation.
    # The state machine binds bare "yes" to the currently pending command ID,
    # floor, and TTL; without a pending command it still rejects fail-closed.
    if text == "yes":
        return ParseResult(Intent.CONFIRM, raw_text=raw)
    if re.fullmatch(r"(?:okay|ok|sure|confirm)", text):
        return ParseResult(Intent.UNKNOWN, raw_text=raw,
                           reject_reason="confirmation_missing_floor")
    if floor is not None:
        if re.search(r"\b(?:do not|don't|not)\b", text):
            return ParseResult(Intent.UNKNOWN, raw_text=raw,
                               reject_reason="negated_floor_command")
        return ParseResult(Intent.UNKNOWN, raw_text=raw,
                           reject_reason="unsafe_floor_context")
    return ParseResult(Intent.UNKNOWN, raw_text=raw, reject_reason="unrecognized")


def _normalize_english_floor_artifacts(text: str) -> str:
    """Repair narrowly observed SenseVoice spellings of English ``floor``."""
    text = re.sub(
        r"\b(floor|level)(?=\d+(?:st|nd|rd|th)?\b)",
        r"\1 ",
        text,
    )
    following_number = rf"(?=(?:{_EN_NUMBER_PATTERN})\b)"
    text = re.sub(rf"\bf\s+{following_number}", "floor ", text)
    action = r"(?:go to|take me to|change(?: it)? to|switch(?: it)? to)"
    text = re.sub(
        rf"\b({action})\s*4\s+{following_number}",
        r"\1 floor ",
        text,
    )
    return text


def _english_floor_value(match: re.Match[str]) -> int | None:
    if match.group("bcode") is not None:
        return -int(match.group("bcode"))
    token = match.group("basement") or match.group("prefix") or match.group("suffix")
    if token is None:
        return None
    value = _english_number_value(token)
    if value is None:
        return None
    return -value if match.group("basement") is not None else value


def _english_number_value(token: str) -> int | None:
    token = token.strip()
    digit = re.fullmatch(r"(-?\d+)(?:st|nd|rd|th)?", token)
    if digit:
        return int(digit.group(1))
    return _EN_CARDINALS.get(token, _EN_ORDINALS.get(token))


def _looks_like_english_floor_command(
        text: str, matches: list[re.Match[str]]) -> bool:
    if len(matches) != 1:
        return False
    match = matches[0]
    before = text[:match.start()].strip()
    after = text[match.end():].strip()
    if _EN_POLITE_SUFFIX.fullmatch(after) is None:
        return False
    if re.search(r"\b(?:do not|don't|not)\b", before):
        return False
    if not before or before == "please":
        return True
    return _EN_ACTION_SUFFIX.search(before) is not None


def _is_english_no_floor_correction(
        text: str, matches: list[re.Match[str]]) -> bool:
    """Accept a concise ``No, floor one`` only in confirmation context."""
    if len(matches) != 1:
        return False
    match = matches[0]
    before = text[:match.start()].strip()
    after = text[match.end():].strip()
    return before in {"no", "no the"} and _EN_POLITE_SUFFIX.fullmatch(after) is not None


def _english_corrected_target(
        text: str, matches: list[re.Match[str]]) -> int | None:
    if len(matches) != 2:
        return None
    first, second = matches
    first_floor = _english_floor_value(first)
    second_floor = _english_floor_value(second)
    if first_floor is None or second_floor is None or first_floor == second_floor:
        return None
    before = text[:first.start()].strip()
    connector = text[first.end():second.start()].strip()
    after = text[second.end():].strip()
    if _EN_POLITE_SUFFIX.fullmatch(after) is None or re.search(r"\bor\b", connector):
        return None
    if re.search(r"\b(?:change|switch)\b", before) and connector in {"to", "instead"}:
        return second_floor
    if re.search(r"\bnot\b", before) and connector in {"", "but", "instead", "but instead"}:
        return second_floor
    return None


def _extract_floors(text: str) -> tuple[list[int], bool]:
    floors: list[int] = []
    malformed = False
    for match in _FLOOR_TOKEN.finditer(text):
        try:
            if match.group("basement"):
                if match.group("bdigit") is not None:
                    floors.append(-int(match.group("bdigit")))
                else:
                    value = _parse_chinese_number(match.group("bchinese"))
                    if value is None:
                        malformed = True
                    else:
                        floors.append(-value)
            elif match.group("arabic") is not None:
                floors.append(int(match.group("arabic")))
            else:
                token = match.group("chinese")
                negative = token.startswith(("地下", "负"))
                token = token.removeprefix("地下").removeprefix("负")
                value = _parse_chinese_number(token)
                if value is None:
                    malformed = True
                else:
                    floors.append(-value if negative else value)
        except ValueError:
            malformed = True
    return floors, malformed


def _has_embedded_floor_action(text: str, matches: list[re.Match[str]]) -> bool:
    """Accept one explicit action clause even when natural framing precedes it."""
    if len(matches) != 1:
        return False
    match = matches[0]
    prefix = text[:match.start()]
    suffix = text[match.end():]
    return (_EMBEDDED_FLOOR_ACTION_SUFFIX.search(prefix) is not None
            and _FLOOR_POLITE_SUFFIX.fullmatch(suffix) is not None)


def _corrected_floor_target(
        text: str, matches: list[re.Match[str]]) -> int | None:
    """Resolve a controlled old-floor/new-floor correction, never a choice."""
    if len(matches) != 2:
        return None
    first, second = matches
    first_floor = _floor_from_match(first)
    second_floor = _floor_from_match(second)
    if first_floor is None or second_floor is None or first_floor == second_floor:
        return None
    source_prefix = text[:first.start()]
    connector = text[first.end():second.start()]
    suffix = text[second.end():]
    if _FLOOR_POLITE_SUFFIX.fullmatch(suffix) is None:
        return None
    if _CHANGE_TARGET_CONNECTOR.fullmatch(connector) is not None:
        return second_floor
    if (_CORRECTED_SOURCE_SUFFIX.search(source_prefix) is not None
            and _CORRECTED_TARGET_CONNECTOR.fullmatch(connector) is not None):
        return second_floor
    return None


def _floor_from_match(match: re.Match[str]) -> int | None:
    """Convert one floor-token match without changing its surrounding text."""
    try:
        if match.group("basement"):
            if match.group("bdigit") is not None:
                return -int(match.group("bdigit"))
            value = _parse_chinese_number(match.group("bchinese"))
            return None if value is None else -value
        if match.group("arabic") is not None:
            return int(match.group("arabic"))
        token = match.group("chinese")
        negative = token.startswith(("地下", "负"))
        token = token.removeprefix("地下").removeprefix("负")
        value = _parse_chinese_number(token)
        return None if value is None else (-value if negative else value)
    except ValueError:
        return None


def _parse_chinese_number(token: str) -> int | None:
    if token in _DIGITS:
        return _DIGITS[token]
    if token == "十":
        return 10
    if token.count("十") == 1:
        tens_text, ones_text = token.split("十")
        tens = 1 if not tens_text else _DIGITS.get(tens_text)
        ones = 0 if not ones_text else _DIGITS.get(ones_text)
        if tens is not None and ones is not None:
            return tens * 10 + ones
    return None


def _looks_like_floor_command(text: str) -> bool:
    """Accept only controlled action text around every recognized floor token.

    Checking the complete context prevents an ASR homophone such as
    ``去富三楼`` (spoken as ``去负三楼``) from silently degrading into the
    valid positive-floor command ``去三楼``. Repeated identical floor tokens
    remain valid because SenseVoice can duplicate a short utterance.
    """
    matches = list(_FLOOR_TOKEN.finditer(text))
    if not matches:
        return False

    cursor = 0
    for match in matches:
        prefix = text[cursor:match.start()]
        if (_FLOOR_ACTION_PREFIX.fullmatch(prefix) is None
                and _FLOOR_CHANGE_PREFIX.fullmatch(prefix) is None):
            return False
        cursor = match.end()
    return _FLOOR_POLITE_SUFFIX.fullmatch(text[cursor:]) is not None
