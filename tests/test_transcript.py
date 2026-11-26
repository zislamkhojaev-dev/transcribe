from transcribe_app.export import format_timestamp, render_txt
from transcribe_app.phrases import Phrase, Word, assign_speakers, group_phrases
from transcribe_app.vad import pack_segments


def test_phrases_split_on_pause_and_keep_order():
    words = [
        Word("сначала", 12.0, 12.4),
        Word("заявка", 12.5, 12.9),
        Word("если", 14.2, 14.5),
        Word("сумма", 14.6, 15.0),
    ]
    phrases = group_phrases(words, pause_seconds=0.7)
    assert [phrase.text for phrase in phrases] == ["сначала заявка", "если сумма"]
    assert phrases[0].start == 12.0
    assert phrases[1].start == 14.2


def test_phrases_sort_by_time():
    words = [Word("вторая", 5.0, 5.2), Word("первая", 1.0, 1.2)]
    phrases = group_phrases(words)
    assert [phrase.text for phrase in phrases] == ["первая", "вторая"]


def test_speakers_follow_first_appearance_without_reordering():
    phrases = [
        Phrase(1.0, 2.0, "вопрос"),
        Phrase(3.0, 4.0, "ответ"),
        Phrase(5.0, 6.0, "уточнение"),
    ]
    turns = [(0.5, 2.5, "B"), (2.5, 4.5, "A"), (4.5, 6.5, "B")]
    labeled = assign_speakers(phrases, turns)
    assert [phrase.speaker for phrase in labeled] == ["Спикер 1", "Спикер 2", "Спикер 1"]
    assert [phrase.text for phrase in labeled] == ["вопрос", "ответ", "уточнение"]


def test_txt_format():
    text = render_txt([Phrase(12.05, 13.0, "Сначала заявка.")])
    assert text == "[00:00:12.050] Сначала заявка.\n"
    stamped = render_txt([Phrase(0, 1, "Да", speaker="Спикер 1")])
    assert stamped == "[00:00:00.000] Спикер 1: Да\n"
    plain = render_txt(
        [Phrase(12.05, 13.0, "Сначала заявка.", speaker="Спикер 1")],
        with_timestamps=False,
    )
    assert plain == "Спикер 1: Сначала заявка.\n"
    assert format_timestamp(3661.25) == "01:01:01.250"


def test_chunks_do_not_overlap_and_stay_short():
    segments = [(0.0, 10.0), (10.2, 21.0), (21.1, 40.0), (40.0, 41.0)]
    chunks = pack_segments(segments, max_duration=22.0)
    assert chunks[0] == (0.0, 21.0)
    for index in range(1, len(chunks)):
        assert chunks[index][0] >= chunks[index - 1][1]
        assert chunks[index][1] - chunks[index][0] <= 22.0 + 1e-9
    assert chunks[-1][1] == 41.0
