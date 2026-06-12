"""Tests for src/transcription/titling.py — heuristic ad-hoc meeting titles."""

from __future__ import annotations

from src.transcription.titling import suggest_title


def _talk(*sentences: str, repeat: int = 1) -> str:
    return " ".join(" ".join(sentences) for _ in range(repeat))


def test_topic_and_detected_name_compose_title() -> None:
    text = _talk(
        "So let's walk through the onboarding flow together.",
        "The onboarding starts with the import screen, as Acme requested.",
        "I showed Acme the onboarding checklist and the import settings.",
        "For Acme the onboarding should skip the import of legacy data.",
        "Then onboarding emails go out and the import finishes overnight.",
    )
    title = suggest_title(text)
    assert title is not None
    assert "Onboarding" in title
    assert "Acme" in title
    assert title == "Onboarding with Acme"


def test_topics_only_when_no_proper_nouns() -> None:
    text = _talk(
        "the budget needs another review before friday",
        "we should compare the budget against the roadmap line by line",
        "the roadmap depends on the budget approval anyway",
        "once the budget clears we publish the roadmap update",
        "any budget overrun pushes the roadmap by a quarter",
    )
    title = suggest_title(text)
    assert title is not None
    assert title.startswith("Call about ")
    assert "budget" in title


def test_known_participant_beats_detected_names() -> None:
    text = _talk(
        "So let's walk through the onboarding flow together.",
        "The onboarding starts with the import screen, as Acme requested.",
        "I showed Acme the onboarding checklist and the import settings.",
        "For Acme the onboarding should skip the import of legacy data.",
        "Then onboarding emails go out and the import finishes overnight.",
    )
    title = suggest_title(
        text, participants=[{"name": "Marie Dupont", "email": "m@x.com"}]
    )
    assert title is not None
    assert "Marie" in title


def test_short_transcript_returns_none() -> None:
    assert suggest_title("hello can you hear me okay great") is None
    assert suggest_title("") is None
    assert suggest_title(None) is None  # type: ignore[arg-type]


def test_filler_only_transcript_returns_none() -> None:
    text = _talk("yeah okay so well you know I think we should maybe", repeat=20)
    assert suggest_title(text) is None


def test_french_stopwords_filtered() -> None:
    text = _talk(
        "alors on regarde le contrat ensemble si tu veux bien",
        "le contrat de maintenance couvre aussi la facturation mensuelle",
        "pour la facturation il faut valider le contrat avant juillet",
        "je renvoie le contrat avec la facturation corrigée demain",
        "et donc la facturation suit le contrat comme convenu",
        "la signature du contrat conditionne la première facturation",
        "ensuite la facturation passe en mode automatique chaque mois",
    )
    title = suggest_title(text)
    assert title is not None
    assert "contrat" in title.lower()
    # Pure function words must never become the topic.
    for stop in ("alors", "donc", "avec", "pour"):
        assert stop not in title.lower().split()


def test_sentence_initial_capitals_are_not_names() -> None:
    """Capitalized sentence starters ('The', 'Demain') must not be mistaken
    for people/companies."""
    text = _talk(
        "The roadmap looks solid overall right now.",
        "The pricing model needs more work before the launch window.",
        "The roadmap and the pricing both ship next quarter regardless.",
        "The pricing review happens after the roadmap freeze deadline.",
        "The roadmap freeze lands monday and pricing follows shortly.",
    )
    title = suggest_title(text)
    assert title is not None
    assert "The" not in title.split()


def test_title_is_capped_in_length() -> None:
    long_word = "hyperpersonnalisation"
    text = _talk(f"we discussed the {long_word} strategy again today", repeat=15)
    title = suggest_title(text)
    assert title is not None
    assert len(title) <= 60
