"""Regression cover for audit finding E6 — the auto-titles were unusable.

With calendar detection dead, every recording fell back to the heuristic
titler, which shipped these to disk as filenames:

    Putain with Alexis · Basically with August · Parce with Terra
    Jete with Auguste · Little with Italy · Because with August

Two failures compounded: a capitalised mid-sentence token seen twice became a
"person", and the top content word became the "topic" — but the most frequent
words in conversation are conversational.

The rule now: names come only from the calendar, topics must clear a
confidence floor, and "no title" is an acceptable answer.
"""

from __future__ import annotations

import pytest

from src.transcription.titling import suggest_title


def transcript(*, filler: str, topic: str = "", topic_n: int = 0,
               length: int = 400) -> str:
    """Build speech that is mostly ``filler``, optionally about ``topic``."""
    words = (filler + " ") * length
    return words + (" " + topic) * topic_n


# ---------------------------------------------------------------------------
# The exact shapes that produced the bad titles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filler",
    ["basically", "parce", "putain", "jete", "because", "little", "vraiment"],
)
def test_a_filler_word_never_becomes_the_topic(filler: str) -> None:
    title = suggest_title(transcript(filler=filler))
    assert title is None or filler not in title.lower(), (
        f"{filler!r} is a filler, not what the meeting was about"
    )


@pytest.mark.parametrize("name", ["August", "Italy", "Terra", "Alexis", "Auguste"])
def test_capitalised_tokens_are_never_guessed_as_people(name: str) -> None:
    """These were 'people' only because they appeared capitalised twice."""
    text = " ".join([f"So {name} was mentioned again and again here."] * 30)
    title = suggest_title(text)
    assert title is None or " with " not in title, (
        f"{title!r} invented a participant from a capitalised token"
    )


def test_no_title_is_a_valid_answer() -> None:
    """Small talk names nothing; a timestamped fallback beats a wrong guess."""
    assert suggest_title(transcript(filler="yeah okay right sure")) is None


def test_too_short_returns_none() -> None:
    assert suggest_title("hello hi how are you") is None
    assert suggest_title("") is None
    assert suggest_title(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Real participants are ground truth
# ---------------------------------------------------------------------------
def test_calendar_participants_do_name_the_meeting() -> None:
    text = transcript(filler="yeah okay", length=100)
    title = suggest_title(text, participants=[{"name": "Marie Dupont"}])
    assert title is not None
    assert "Marie" in title


def test_participant_plus_strong_topic() -> None:
    text = transcript(filler="yeah okay", length=200, topic="roadmap", topic_n=40)
    title = suggest_title(text, participants=[{"name": "Marie Dupont"}])
    assert title == "Roadmap with Marie"


def test_participant_string_form_is_parsed() -> None:
    title = suggest_title(
        transcript(filler="yeah okay", length=100),
        participants=["Marie Dupont <marie@example.com>"],
    )
    assert title is not None and "Marie" in title


# ---------------------------------------------------------------------------
# Confidence floor
# ---------------------------------------------------------------------------
def test_a_dominant_pair_of_topics_names_the_meeting() -> None:
    text = transcript(filler="yeah okay", length=200,
                      topic="roadmap pricing", topic_n=40)
    title = suggest_title(text)
    assert title is not None
    assert "roadmap" in title and "pricing" in title


def test_one_marginal_topic_is_not_enough() -> None:
    """A single word just over the floor is a guess, not a title."""
    text = transcript(filler="yeah okay right", length=400, topic="budget", topic_n=7)
    assert suggest_title(text) is None


def test_floor_scales_with_content_volume() -> None:
    """The same number of mentions names a short call but not a long one.

    The floor is a share of *content* words, so padding must be content, not
    filler — filler is excluded before the ratio is taken.
    """
    import string

    # Distinct alphabetic padding — the tokenizer drops digits, so "word1"
    # and "word2" would both collapse to "word" and become the topic.
    distinct = [
        f"zz{a}{b}{c}"
        for a in string.ascii_lowercase
        for b in string.ascii_lowercase
        for c in string.ascii_lowercase
    ]
    mentions = " budget review" * 20
    short = " ".join(distinct[:30]) + mentions
    long = " ".join(distinct[:3000]) + mentions

    assert suggest_title(short) is not None
    assert suggest_title(long) is None


# ---------------------------------------------------------------------------
# Language handling
# ---------------------------------------------------------------------------
def test_french_elisions_expose_the_stem() -> None:
    """"d'offre" carries "offre"; "qu'il" and "c'était" carry nothing."""
    text = " ".join(["Alors qu'il disait que c'était un appel d'offre important."] * 40)
    title = suggest_title(text)
    assert title is None or ("qu'il" not in title and "c'était" not in title)


def test_french_inversion_is_not_a_topic() -> None:
    text = " ".join(["Alors aura-t-on le budget aujourd'hui aura-t-on vraiment"] * 40)
    title = suggest_title(text)
    assert title is None or "aura-t-on" not in title


def test_english_contractions_are_not_topics() -> None:
    text = " ".join(["I don't think that's what we should do here"] * 60)
    title = suggest_title(text)
    assert title is None or ("don't" not in title and "that's" not in title)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------
def test_title_is_length_capped() -> None:
    text = transcript(filler="yeah", length=200,
                      topic="internationalisation reconceptualisation", topic_n=40)
    title = suggest_title(text)
    assert title is None or len(title) <= 60
