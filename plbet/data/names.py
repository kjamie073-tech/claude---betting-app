"""Team and player name matching across data sources.

football-data.co.uk names are the canonical team names everywhere in this
package ("Man City", "Nott'm Forest", ...). Understat and FPL names, and the
names people type ("Spurs", "Man Utd", "Forest"), are mapped onto them.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

CANONICAL_ALIASES: dict[str, list[str]] = {
    "Arsenal": ["arsenal", "the gunners"],
    "Aston Villa": ["aston villa", "villa"],
    "Bournemouth": ["bournemouth", "afc bournemouth"],
    "Brentford": ["brentford"],
    "Brighton": ["brighton", "brighton and hove albion", "brighton & hove albion", "brighton hove"],
    "Burnley": ["burnley"],
    "Cardiff": ["cardiff", "cardiff city"],
    "Chelsea": ["chelsea"],
    "Coventry": ["coventry", "coventry city"],
    "Crystal Palace": ["crystal palace", "palace"],
    "Everton": ["everton"],
    "Fulham": ["fulham"],
    "Huddersfield": ["huddersfield", "huddersfield town"],
    "Hull": ["hull", "hull city"],
    "Ipswich": ["ipswich", "ipswich town"],
    "Leeds": ["leeds", "leeds united"],
    "Leicester": ["leicester", "leicester city"],
    "Liverpool": ["liverpool"],
    "Luton": ["luton", "luton town"],
    "Man City": ["man city", "manchester city", "mcfc"],
    "Man United": ["man united", "manchester united", "man utd", "man u", "mufc"],
    "Middlesbrough": ["middlesbrough", "boro"],
    "Newcastle": ["newcastle", "newcastle united", "newcastle utd"],
    "Norwich": ["norwich", "norwich city"],
    "Nott'm Forest": ["nott'm forest", "nottingham forest", "forest", "nottm forest", "notts forest"],
    "Sheffield United": ["sheffield united", "sheffield utd", "sheff utd", "sheffield united fc"],
    "Southampton": ["southampton", "saints"],
    "Stoke": ["stoke", "stoke city"],
    "Sunderland": ["sunderland"],
    "Swansea": ["swansea", "swansea city"],
    "Tottenham": ["tottenham", "tottenham hotspur", "spurs"],
    "Watford": ["watford"],
    "West Brom": ["west brom", "west bromwich albion", "west bromwich", "wba"],
    "West Ham": ["west ham", "west ham united"],
    "Wolves": ["wolves", "wolverhampton wanderers", "wolverhampton"],
}

# Understat team titles that differ from the canonical names.
UNDERSTAT_TEAMS = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "Sheffield United": "Sheffield United",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
}

# FPL team names that differ from the canonical names.
FPL_TEAMS = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Wolves": "Wolves",
    "Man City": "Man City",
    "Newcastle": "Newcastle",
}


def strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def norm(text: str) -> str:
    """Lowercase, accent-free, punctuation-light form used for matching."""
    text = strip_accents(str(text)).lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_ALIAS_INDEX: dict[str, str] = {}
for _canon, _aliases in CANONICAL_ALIASES.items():
    _ALIAS_INDEX[norm(_canon)] = _canon
    for _a in _aliases:
        _ALIAS_INDEX[norm(_a)] = _canon


def team(name: str) -> str:
    """Map any team spelling to its canonical name.

    Unknown names come back unchanged (a newly promoted club with a plain name
    usually needs no mapping). Raises if a fuzzy match is ambiguous.
    """
    if name in UNDERSTAT_TEAMS:
        return UNDERSTAT_TEAMS[name]
    if name in FPL_TEAMS:
        return FPL_TEAMS[name]
    key = norm(name)
    if key in _ALIAS_INDEX:
        return _ALIAS_INDEX[key]
    close = difflib.get_close_matches(key, list(_ALIAS_INDEX), n=2, cutoff=0.85)
    if len(close) == 1 or (len(close) == 2 and _ALIAS_INDEX[close[0]] == _ALIAS_INDEX[close[1]]):
        return _ALIAS_INDEX[close[0]]
    return name.strip()


def resolve_team(name: str, known: list[str]) -> str:
    """Resolve user input to one of the known canonical team names."""
    canon = team(name)
    if canon in known:
        return canon
    key = norm(name)
    for k in known:
        if norm(k) == key:
            return k
    close = difflib.get_close_matches(key, [norm(k) for k in known], n=1, cutoff=0.6)
    if close:
        return next(k for k in known if norm(k) == close[0])
    raise KeyError(f"Unknown team '{name}'. Known teams: {', '.join(sorted(known))}")


def player_key(name: str) -> str:
    return norm(name).replace("'", "")


def name_score(a: str, b: str) -> float:
    """Similarity of two player names, tolerant of missing middle names."""
    ka, kb = player_key(a), player_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    ta, tb = ka.split(), kb.split()
    # One name fully contained in the other ("Gabriel" vs "Gabriel Magalhaes").
    if set(ta) <= set(tb) or set(tb) <= set(ta):
        return 0.92
    # Same surname and same first initial.
    if ta[-1] == tb[-1] and ta[0][0] == tb[0][0]:
        return 0.9
    return difflib.SequenceMatcher(None, ka, kb).ratio()


def best_match(name: str, candidates, threshold: float = 0.8,
               surname_fallback: bool = True) -> tuple[int | None, float]:
    """Position of the best-matching name in ``candidates``, and its score.

    Falls back to the surname when exactly one candidate shares it, which
    covers players listed under different first names by different sources
    ("Rayan Cherki" in FPL, "Mathis Cherki" in Understat). Returns
    (None, best score) if nothing is close enough.
    """
    cands = list(candidates)
    if not cands:
        return None, 0.0
    scores = [name_score(name, c) for c in cands]
    j = max(range(len(cands)), key=scores.__getitem__)
    if scores[j] >= threshold:
        return j, scores[j]
    key = player_key(name).split()
    if key and surname_fallback:
        same = [i for i, c in enumerate(cands) if player_key(c).split()[-1:] == key[-1:]]
        if len(same) == 1 and len(key[-1]) >= 4:
            return same[0], 0.85
    return None, scores[j]
