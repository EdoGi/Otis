"""Suggest a meeting title from the transcript text — fully on-device.

Used for ad-hoc recordings only (no calendar event to take a title from):
instead of saving everything as "Ad-hoc Recording", we mine the transcript
for the dominant topic words and the people/companies mentioned, and build
something like::

    Onboarding with Acme
    Call with Marie — budget
    Call about roadmap & pricing

Heuristic, not semantic: frequency-scored content words (multilingual
stopword filtering for the six supported languages) plus capitalized-
mid-sentence tokens as proper-noun candidates. When the transcript is too
short or nothing stands out, returns ``None`` and the caller keeps its
existing fallback.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

MIN_MEANINGFUL_WORDS = 25
MAX_TITLE_LEN = 60

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")

# Function words + conversational fillers across en/fr/es/it/pt/de. Compact
# on purpose — we only need to keep noise out of the top-frequency slots.
_STOPWORDS: frozenset[str] = frozenset("""
the a an and or but if then else so of to in on at for from by with without
about as into over under again once here there all any both each few more
most other some such only own same than too very can will just should now
this that these those is are was were be been being have has had having do
does did doing would could may might must shall not no nor it its it's i'm
i've you he she they we you're they're them his her their our your my me him
us who whom which what when where why how yes yeah okay ok right well like
know think going get got want said say really actually maybe kind sort lot
thing things stuff one two also let need make sure mean bit time good great
hello hi thanks thank bye let's gonna
le la les un une des et ou mais si alors donc de du au aux dans sur pour par
avec sans chez vers entre pendant après avant comme plus moins très peu tout
toute tous toutes autre autres même mêmes ce cet cette ces que qui quoi dont
où quand comment pourquoi est sont était étaient être été avoir ai as avons
avez ont avait fait faire faisait peux peut pouvez pouvons veux veut voulez
dois doit devez il elle ils elles nous vous je tu on mon ma mes ton ta tes
son sa ses notre votre leur leurs oui non ouais bon ben voilà euh hein quoi
c'est j'ai n'est qu'on d'accord peut-être vraiment juste truc trucs chose
choses aussi donc enfin bref allez salut bonjour bonsoir merci
el los las uno una unos unas y o pero si entonces de del en sobre para por
con sin hacia entre durante después antes como más menos muy poco todo toda
todos todas otro otra que quien cual cuando donde como porque es son era
eran ser sido estar estaba tener tengo tiene hacer hace puede quiero quiere
debo debe nosotros vosotros ellos ellas yo tú usted sí no vale bueno pues
gracias hola
il lo la i gli le un uno una e o ma se allora di del della in su per da con
senza verso tra fra durante dopo prima come più meno molto poco tutto tutta
tutti tutte altro altra che chi quale quando dove perché è sono era erano
essere stato avere ho hai abbiamo hanno fare faccio può voglio vuole devo
deve noi voi io tu lei sì no va bene allora grazie ciao
o os as um uma uns umas e ou mas se então de do da em sobre para por com sem
entre durante depois antes como mais menos muito pouco todo toda todos todas
outro outra que quem qual quando onde porque é são era eram ser sido estar
estava ter tenho tem fazer faço pode quero quer devo deve nós vocês eu tu
ele ela eles elas sim não tá bom então obrigado obrigada olá
der die das den dem des ein eine einen einem eines und oder aber wenn dann
also von zu in auf für aus bei mit ohne nach vor über unter zwischen während
wie mehr weniger sehr wenig alle alles andere anderen dass was wer wann wo
warum ist sind war waren sein gewesen haben habe hat hatte machen mache kann
will wollen muss soll wir ihr sie ich du er es mein dein sein unser euer ja
nein gut genau danke hallo tschüss
""".split())


def _participant_names(participants: list[Any]) -> list[str]:
    """First names from MeetingSnapshot-style participant entries."""
    names: list[str] = []
    for p in participants or []:
        if isinstance(p, dict):
            raw = str(p.get("name") or "")
        else:
            raw = str(p).split("<")[0]
        first = raw.strip().split(" ")[0].strip()
        if len(first) >= 2 and first[0].isalpha() and "@" not in first:
            names.append(first[0].upper() + first[1:])
    return names


def _tokenize(text: str) -> list[tuple[str, bool]]:
    """``(token, is_sentence_start)`` pairs, in document order."""
    out: list[tuple[str, bool]] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        for i, m in enumerate(_WORD_RE.finditer(sentence)):
            out.append((m.group(0), i == 0))
    return out


def suggest_title(
    text: str,
    *,
    participants: list[Any] | None = None,
    max_topics: int = 2,
) -> str | None:
    """Best-effort meeting title from transcript ``text``; None if nothing usable."""
    tokens = _tokenize(text or "")
    meaningful = [
        t for t, _ in tokens
        if len(t) >= 3 and t.lower() not in _STOPWORDS
    ]
    if len(meaningful) < MIN_MEANINGFUL_WORDS:
        return None

    lowercase_seen = {t for t, _ in tokens if t[0].islower()}

    # Topic candidates: content words by frequency (longer words win ties).
    topic_counts: Counter[str] = Counter()
    # Name candidates: capitalized mid-sentence AND never seen lowercase —
    # the signature of a person/company name rather than a sentence start.
    name_counts: Counter[str] = Counter()

    for token, is_start in tokens:
        lower = token.lower()
        if lower in _STOPWORDS or len(token) < 4:
            continue
        topic_counts[lower] += 1
        if (
            not is_start
            and token[0].isupper()
            and lower not in lowercase_seen
            and len(token) >= 3
        ):
            name_counts[token] += 1

    names = [n for n, c in name_counts.most_common() if c >= 2]
    # Known participants (Generate-Transcript path may carry some) outrank
    # detected ones — they're ground truth.
    known = _participant_names(list(participants or []))
    name = known[0] if known else (names[0] if names else None)

    excluded = {n.lower() for n in names} | {k.lower() for k in known}
    if name:
        excluded.add(name.lower())
    topics = [
        t for t, c in topic_counts.most_common(10)
        if c >= 3 and t not in excluded
    ][:max_topics]

    if name and topics:
        title = f"{topics[0].capitalize()} with {name}"
    elif name:
        title = f"Call with {name}"
    elif topics:
        title = "Call about " + " & ".join(topics)
    else:
        return None
    return title[:MAX_TITLE_LEN].rstrip()


__all__ = ["suggest_title"]
