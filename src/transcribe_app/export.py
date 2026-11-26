from transcribe_app.phrases import Phrase


def format_timestamp(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def render_txt(phrases: list[Phrase], *, with_timestamps: bool = True) -> str:
    lines: list[str] = []
    for phrase in phrases:
        body = f"{phrase.speaker}: {phrase.text}" if phrase.speaker else phrase.text
        if with_timestamps:
            lines.append(f"[{format_timestamp(phrase.start)}] {body}")
        else:
            lines.append(body)
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
