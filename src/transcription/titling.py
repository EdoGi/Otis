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

#: A topic must appear at least this many times, whatever the length.
MIN_TOPIC_COUNT = 6

#: ...and at least this fraction of the transcript's content words, so a
#: two-hour meeting needs far more than six mentions to name itself.
TOPIC_FLOOR_RATIO = 0.008

#: A lone topic has to clear the floor by this much to name the meeting on its
#: own. Two corroborating topics are better evidence than one frequent word.
SOLO_TOPIC_MULTIPLIER = 2.0

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*")  # noqa: RUF001 - Whisper emits U+2019 (curly) apostrophes
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

# Conversational fillers that survived the list above and ended up *as the
# title* on real recordings: "Basically with August", "Parce with Terra",
# "Jete with Auguste", "Putain with Alexis", "Because with August". They're
# frequent, long enough to pass the length filter, and carry no topic at all.
_FILLERS: frozenset[str] = frozenset("""
basically actually literally obviously essentially certainly definitely
probably honestly frankly seriously totally completely absolutely exactly
anyway anyways somehow whatever whenever wherever however therefore because
though although unless until while since before after during always never
often sometimes usually already still even ever else enough quite rather
pretty almost mostly simply clearly perhaps indeed alright okay sorry please
guys folks stuff thing point idea part side case way ways today tomorrow
yesterday week month year day days weeks months years minute minutes hour
hours second seconds first last next previous other another same different
little big small long short high low early late many much few less better
best worse worst sure fine nice cool yeah yep nope hmm uhm erm
parce parceque pourquoi vraiment carrément franchement honnêtement clairement
évidemment forcément justement effectivement absolument complètement
totalement exactement finalement simplement genre truc machin bidule putain
merde bordel voilà voila ouais ouaip bref ensuite ainsi cependant pourtant
toutefois néanmoins jete jeté déjà encore toujours jamais souvent parfois
quelquefois maintenant aujourdhui demain hier semaine mois année jour jours
semaines heures minute minutes seconde secondes petit petite grand grande
autre autres pareil pareille chose choses côté partie point idée façon
manière moment vachement grave trop assez plutôt presque surtout sinon
entonces bueno pues claro vale digamos realmente verdad cosa cosas
cioè quindi allora insomma praticamente veramente davvero magari cosa cose
então tipo realmente verdade coisa coisas
also halt eben ganz eigentlich wirklich vielleicht sowieso trotzdem
""".split())

# Measured, not guessed: the most frequent surviving tokens across the 25 real
# transcripts on this machine. Conversation is *made* of these words, so a
# frequency heuristic picks them every time unless they're excluded — which is
# how "Call about pense & bien" happens. Genuinely topical words in the same
# corpus (projet, rapport, mémoire, technique, problème) are deliberately kept.
_CONVERSATIONAL: frozenset[str] = frozenset("""
pense penses pensez dire dis dit dites vais vas allez aller sais sait savez
savoir coup coups temps deux trois quatre cinq beaucoup voir vois voyez suis
êtes sommes fois gens faut faudra falloir mettre mets met besoin pouvoir
peux-tu soit soient donner donne donnes train puis ensuite quelque quelques
quelqu chose choses venir viens vient venu came prendre prends prend pris
passer passe passé arriver arrive arrivé trouver trouve trouvé regarder
regarde parler parle parlé demander demande dire redis commencer commence
continuer continue finir finit rester reste garder garde laisser laisse
appeler appelle utiliser utilise essayer essaie devoir dois doit falloir
oui non peut-être bon bien mieux mal moins plus très trop assez
est-ce qu'est-ce n'est-ce aura auras auront aurait avait avais avions
something anything nothing everything someone anyone everyone somebody
echt gaan gaat willen wil kunnen kan
going gonna wanna gotta getting doing making taking saying seeing looking
talking coming trying needing wanting having being giving putting thinking
don't doesn't didn't won't can't cannot isn't aren't wasn't weren't haven't
hasn't hadn't wouldn't couldn't shouldn't that's there's what's here's who's
he's she's we're we've we'll you'll they'll i'll i'd you'd we'd they'd
""".split())

_ALL_STOPWORDS: frozenset[str] = _STOPWORDS | _FILLERS | _CONVERSATIONAL

#: Elided articles/pronouns in French and Italian. "d'offre" carries the topic
#: in "offre"; "qu'il" and "c'était" carry nothing at all. Splitting these lets
#: the stopword list do its job instead of treating each contraction as a
#: distinct, unrecognised — and therefore topical-looking — word.
_ELISIONS: frozenset[str] = frozenset(
    "c d j l m n s t qu qu' aujourd all dell nell sull quell".split()
)


#: French interrogative inversion ("aura-t-on", "y a-t-il"). The hyphens make
#: these look like rare distinctive words when they are pure grammar.
_INVERSION_RE = re.compile(r"-t-(on|il|elle|ils|elles)$", re.IGNORECASE)


def _normalise(token: str) -> str:
    """Reduce a token to the part that carries meaning, for stopword matching."""
    if _INVERSION_RE.search(token):
        token = _INVERSION_RE.sub("", token)
    for sep in ("'", "’"):  # noqa: RUF001 - Whisper emits both apostrophes
        if sep in token:
            head, _, tail = token.partition(sep)
            if head.lower() in _ELISIONS and tail:
                return tail
    return token


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
    """Best-effort meeting title from transcript ``text``; None if nothing usable.

    Deliberately conservative. The previous version guessed people from any
    capitalised mid-sentence token seen twice, and took the top content word as
    the topic, which on real recordings produced "Putain with Alexis",
    "Little with Italy" and "Basically with August" — the filler word plus a
    capitalised token that often wasn't a person at all.

    Two rules now:

    * **Names are never guessed.** Only ``participants`` (ground truth from the
      calendar) can put a person in the title.
    * **Topics must clear a confidence floor** — repeated well above the noise,
      and not in the filler list.

    Returning ``None`` is a perfectly good outcome; the caller falls back to a
    plain timestamped name, which beats a confidently wrong one.
    """
    known = _participant_names(list(participants or []))
    tokens = [_normalise(t) for t, _ in _tokenize(text or "")]
    meaningful = [
        t for t in tokens
        if len(t) >= 3 and t.lower() not in _ALL_STOPWORDS
    ]
    if len(meaningful) < MIN_MEANINGFUL_WORDS:
        # Too little to mine a topic from — but a calendar participant is
        # ground truth, not a guess, and still names the meeting.
        return f"Call with {known[0]}"[:MAX_TITLE_LEN] if known else None

    topic_counts: Counter[str] = Counter()
    for token in tokens:
        lower = token.lower()
        if lower in _ALL_STOPWORDS or len(token) < 4:
            continue
        topic_counts[lower] += 1

    # Confidence floor: a topic has to recur enough to be *about* something,
    # scaled to transcript length so a long meeting needs more than three
    # mentions. Without this, any word that squeaked past the stopword list
    # became the title.
    floor = max(MIN_TOPIC_COUNT, int(len(meaningful) * TOPIC_FLOOR_RATIO))
    excluded = {k.lower() for k in known}
    topics = [
        t for t, c in topic_counts.most_common(12)
        if c >= floor and t not in excluded
    ][:max_topics]

    name = known[0] if known else None
    if name and topics:
        title = f"{topics[0].capitalize()} with {name}"
    elif name:
        title = f"Call with {name}"
    elif len(topics) >= 2:
        title = "Call about " + " & ".join(topics)
    elif topics and topic_counts[topics[0]] >= floor * SOLO_TOPIC_MULTIPLIER:
        # A single topic names the meeting only if it really dominates —
        # otherwise "Call about <one common word>" is just a guess wearing a
        # confident face.
        title = f"Call about {topics[0]}"
    else:
        return None
    return title[:MAX_TITLE_LEN].rstrip()


__all__ = ["suggest_title"]
