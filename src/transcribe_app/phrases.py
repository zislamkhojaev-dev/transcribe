from dataclasses import dataclass

PAUSE_SECONDS = 0.7


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Phrase:
    start: float
    end: float
    text: str
    speaker: str | None = None


def group_phrases(words: list[Word], pause_seconds: float = PAUSE_SECONDS) -> list[Phrase]:
    """Join words into phrases on pauses, keeping chronological order."""
    ordered = [word for word in sorted(words, key=lambda item: (item.start, item.end)) if word.text.strip()]
    if not ordered:
        return []

    groups: list[list[Word]] = [[ordered[0]]]
    for word in ordered[1:]:
        gap = word.start - groups[-1][-1].end
        if gap >= pause_seconds:
            groups.append([word])
        else:
            groups[-1].append(word)

    phrases: list[Phrase] = []
    for group in groups:
        text = " ".join(word.text.strip() for word in group)
        text = " ".join(text.split())
        if not text:
            continue
        phrases.append(Phrase(start=group[0].start, end=group[-1].end, text=text))
    return phrases


def assign_speakers(phrases: list[Phrase], turns: list[tuple[float, float, str]]) -> list[Phrase]:
    """Label phrases by overlap. Labels become Спикер N in order of first appearance."""
    raw_labels: list[str | None] = []
    for phrase in phrases:
        raw_labels.append(_best_turn(phrase.start, phrase.end, turns))

    names: dict[str, str] = {}
    for label in raw_labels:
        if label and label not in names:
            names[label] = f"Спикер {len(names) + 1}"

    labeled: list[Phrase] = []
    for phrase, label in zip(phrases, raw_labels):
        speaker = names[label] if label else None
        labeled.append(Phrase(phrase.start, phrase.end, phrase.text, speaker))
    return labeled


def _best_turn(start: float, end: float, turns: list[tuple[float, float, str]]) -> str | None:
    best_label = None
    best_overlap = 0.0
    for turn_start, turn_end, label in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label
