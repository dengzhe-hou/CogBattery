"""Scoring for the P9 same-stimulus study.  Pure functions, no model calls.

n-back response coding (SPEC r2): the first-line rule (nb_code_first_line) is primary; the
archived CogArena rule (nb_code = run_eval.score_multiturn_item, re-expressed with an explicit
unparseable state as in g1_extract.nb_code) is the sensitivity parse.  Trial types
follow sro_human.py / g1_extract.py: target = same letter as n back (case-insensitive);
lure = non-target equal to n-1 or n+1 back (n >= 2); control = any other non-target.

Strachan false-belief coding implements the published protocol: a response is correct when
it names the location where the protagonist put the object (for false- and true-belief
versions alike); naming the other location, or neither location, is incorrect (Strachan et
al. 2024 Results: "failing to specify one of the two locations" was an error).  The rule
was developed on the authors' coded responses (all 882 coded false/true-belief responses
in the OSF workbook); r0 before any model call in this study, r1 and r1a after
pilot 25552 and before any formal call; see materials.py and SPEC revisions.
"""
import re

from scipy.stats import norm


# ---------------------------------------------------------------- n-back
# Two response parsers (SPEC r2, user decision 2026-10-12):
# - nb_code_first_line: primary.  Moved unchanged from audit_pilot.first_line_code (r1a), where it
#   was the candidate counted on pilot 25552 (deepseek-r1:7b 104/135, mixtral:8x7b 135/135).
# - nb_code: the archived run_eval prefix rule, sensitivity.
PARSE_RULES = ('first_line', 'archived_prefix')
PRIMARY_PARSE = 'first_line'
ECHO_CURRENT_TRIAL = r'Trial (\d+):[^\n]*\n\s*Your response:\s*'


def nb_code(resp):
    """Archived prefix rule.  True = MATCH, False = NO MATCH, None = unparseable (same strings as run_eval)."""
    act = str(resp or '').strip().lower()
    core = act.strip().strip('."\'!').strip()
    if core == 'no match' or core.startswith('no match '):
        return False
    if (core == 'match' or core.startswith('match ')) and 'no' not in act:
        return True
    return None


def nb_code_first_line(resp, trial):
    r"""First-line rule.  True = MATCH, False = NO MATCH, None = no decision on the first line.

    trial is the 1-based trial number of this call.
    1. Strip outer whitespace.
    2. While the reply starts with an echo of the current trial, remove it: 'Trial k:', any characters
       up to the end of that line, a newline, optional whitespace, 'Your response:', optional
       whitespace (ECHO_CURRENT_TRIAL, matched at the start), with k equal to this trial number
       (case-sensitive; the echoed letter is not checked).
    3. Remove, once, the leading run of non-whitespace characters up to '.png' and the whitespace
       after it (^\S*\.png\s*; with several '.png' in that run, up to the last one).
    4. Remove every leading character that is not an ASCII letter (^[^A-Za-z]+).
    5. Take the first line, strip whitespace, lower-case, strip . " ' ! * at both ends, strip whitespace.
    6. Starts with 'no match' followed by a word boundary -> NO MATCH; else starts with 'match'
       followed by a word boundary -> MATCH; else None."""
    s = str(resp or '').strip()
    while True:
        m = re.match(ECHO_CURRENT_TRIAL, s)
        if not (m and int(m.group(1)) == trial):
            break
        s = s[m.end():]
    s = re.sub(r'^\S*\.png\s*', '', s)
    s = re.sub(r'^[^A-Za-z]+', '', s)
    core = s.split('\n', 1)[0].strip().lower().strip('."\'!*').strip()
    if re.match(r'no match\b', core):
        return False
    if re.match(r'match\b', core):
        return True
    return None


def nb_codes(resp, trial):
    """Both parses of one reply, keyed by PARSE_RULES."""
    return {'first_line': nb_code_first_line(resp, trial), 'archived_prefix': nb_code(resp)}


def classify(seq, n):
    """seq: letters as presented.  Returns per-trial dicts (identity is case-insensitive)."""
    s = [x.lower() for x in seq]
    out = []
    for i in range(len(s)):
        tgt = i >= n and s[i - n] == s[i]
        lm1 = (not tgt) and n >= 2 and i >= n - 1 and s[i - (n - 1)] == s[i]
        lp1 = (not tgt) and n >= 2 and i >= n + 1 and s[i - (n + 1)] == s[i]
        kind = 'target' if tgt else ('lure' if (lm1 or lp1) else 'control')
        out.append(dict(pos=i, kind=kind, lm1=lm1, lp1=lp1))
    return out


def loglinear_dprime(hits, n_signal, fas, n_noise):
    """Hautus (1995) log-linear correction, as in hr_common.loglinear_dprime."""
    h = min(.999, max(.001, (hits + .5) / (n_signal + 1)))
    f = min(.999, max(.001, (fas + .5) / (n_noise + 1)))
    return float(norm.ppf(h) - norm.ppf(f))


def _rate(trials, kind, rule):
    if kind == 'nontarget':
        xs = [t for t in trials if t['kind'] != 'target']
    else:
        xs = [t for t in trials if t['kind'] == kind]
    if rule == 'R1':
        xs = [t for t in xs if t['code'] is not None]
    n = len(xs)
    k = sum(t['code'] is True for t in xs)
    return dict(k=k, n=n, rate=(k / n) if n else None)


def episode_dprime(trials, min_each=5):
    """d' on trials with a response (R1); None if fewer than min_each targets or non-targets."""
    r = [t for t in trials if t['code'] is not None]
    tg = [t for t in r if t['kind'] == 'target']
    nt = [t for t in r if t['kind'] != 'target']
    if len(tg) < min_each or len(nt) < min_each:
        return None
    return loglinear_dprime(sum(t['code'] is True for t in tg), len(tg), sum(t['code'] is True for t in nt), len(nt))


def endpoints(episodes):
    """episodes: list of trial lists; each trial has kind and code (True/False/None).

    Rates pool trials over episodes.  R1 (primary, SPEC r2): denominators are trials with a
    response; R2 (sensitivity): all trials, a missing/unparseable response counts as not MATCH.
    Both are always reported.  d' is computed per episode on R1 trials and averaged over episodes.
    valid_trial_coverage = share of trials with a response (overall and by trial kind)."""
    allt = [t for e in episodes for t in e]
    res = {}
    for rule in ('R1', 'R2'):
        res[rule] = {k: _rate(allt, k, rule) for k in ('target', 'lure', 'control', 'nontarget')}
    dps = [episode_dprime(e) for e in episodes]
    ok = [d for d in dps if d is not None]
    res['dprime_episode_mean'] = (sum(ok) / len(ok)) if ok else None
    res['dprime_episodes_used'] = len(ok)
    res['n_episodes'] = len(episodes)
    res['n_trials'] = len(allt)
    res['no_response_rate'] = (sum(t['code'] is None for t in allt) / len(allt)) if allt else None
    res['valid_trial_coverage'] = {k: dict(n_valid=res['R1'][k]['n'], n_all=res['R2'][k]['n'],
                                           rate=(res['R1'][k]['n'] / res['R2'][k]['n']) if res['R2'][k]['n'] else None)
                                   for k in ('target', 'lure', 'control', 'nontarget')}
    res['valid_trial_coverage']['all'] = dict(n_valid=sum(t['code'] is not None for t in allt), n_all=len(allt),
                                              rate=(sum(t['code'] is not None for t in allt) / len(allt)) if allt else None)
    return res


def mback_consistency(seq, codes, ms=(1, 2, 3, 4), start=4):
    """Share of responded trials (positions >= start) whose MATCH/NO MATCH agrees with an
    m-back rule (MATCH iff letter equals the letter m back), for each m."""
    s = [x.lower() for x in seq]
    out = {}
    for m in ms:
        agree = n = 0
        for i in range(max(start, m), len(s)):
            if codes[i] is None:
                continue
            n += 1
            agree += (codes[i] is True) == (s[i] == s[i - m])
        out[m] = dict(agree=agree, n=n)
    return out


# ---------------------------------------------------------------- Strachan false belief
# Story number = item position 1-9 (same story in lists A and B).  ORIG = where the protagonist
# put the object; NEW = the other location named in the story.
ORIG = {1: r"\btop\b", 2: r"sand ?castle|\bsand\b|sandbox|sand ?pit", 3: r"daffodil",
        4: r"play ?box|toy ?box", 5: r"\btmp\b|\btemp\b|temporary", 6: r"\bside\b", 8: r"\btop\b",
        9: r"\btin\b|living ?room"}
NEW = {1: r"\bbottom\b", 2: r"bucket", 3: r"\broses?\b", 4: r"\bsofa\b|\bcouch\b",
       5: r"(?<![a-z])photos\b", 6: r"\bfront\b", 8: r"\bbottom\b", 9: r"shelf|kitchen"}
# Descriptions of the original location without naming it ("where he put it").  Used only
# when neither location is named.
DESC = (r"where (?:he|she|they) (?:had |has )?(?:originally |first |last |initially )?"
        r"(?:put|left|placed|buried|burried|hid|uploaded|saved|stored|kept)\b"
        r"|(?:folder|place|location|spot|pocket|drawer) (?:that )?(?:he|she|they) (?:had |has )?"
        r"(?:originally |first |initially )?(?:put|left|placed|uploaded|saved|buried|hid)\b"
        r"|\boriginal(?:ly)?\s+(?:location|place|folder|spot|pocket|drawer)")
COMMIT = (r"\b(?:look|looks|looking|search|searches|searching|check|checks|checking|expect|expects|"
          r"expecting|think|thinks|believe|believes)\b")
ECHO_NGRAM = 15  # a sentence sharing a 15-word run with the item text is a restatement of the story
# Revision r1 (2026-10-11, after the pilot, before any formal call): commitment is read per
# clause, not per sentence, so that "X is in <real>, but he will look in <believed>" and
# "he will not look in <real>; he will look in <believed>" are scored on the believed location.
CLAUSE_SPLIT = r";|\bbut\b"
NEGATION = r"\b(?:not|never|won't|wouldn't|doesn't|don't|didn't|isn't|cannot|can't|shouldn't)\s+(?:[a-z']+\s+){0,2}$"
# Revision r1a (2026-10-11 evening, before any formal call): under a negated verb, an exclusive
# construction ("not look anywhere other than / except / but X") commits to X, and an affirmative
# alternative after the negated location in the same clause ("won't look in Y, rather in X")
# commits to X.  'but' right after the "anywhere / nowhere / any + up to 3 words (+ else)" part
# of an exclusive construction does not split the clause.
EXCL_HEAD = r"\b(?:anywhere|nowhere|any(?:\s+[a-z']+){0,3}?)(?:\s+else)?\s+"
EXCLUSIVE = EXCL_HEAD + r"(?:other than|except|besides|apart from|aside from|but)\b"
ALTERNATIVE = r"\b(?:rather|instead)\b(?!\s+(?:than|of)\b)"


def _norm(t):
    return (t or '').lower().replace('’', "'").replace('‘', "'")


def _words(t):
    return re.findall(r"[a-z0-9_/]+", t)


def _grams(ws, k):
    return {tuple(ws[i:i + k]) for i in range(len(ws) - k + 1)}


def _sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]


def correct_is_orig(story, lst):
    """Only the true-belief catch story B9 (Grace sees the move) has the new location as answer."""
    return not (story == 9 and lst == 'B')


def _explicit(story, s, cio):
    o = [m.start() for m in re.finditer(ORIG[story], s)]
    n = [m.start() for m in re.finditer(NEW[story], s)]
    return (o, n) if cio else (n, o)


def _decide(c, w):
    return 1 if (c and (not w or min(c) < min(w))) else 0


def _clauses(sentence):
    """Split at ';' and 'but', except a 'but' that ends the head of an exclusive construction
    ("anywhere but X", "any drawer but X")."""
    out, start = [], 0
    for m in re.finditer(CLAUSE_SPLIT, sentence):
        if m.group() == 'but' and re.search(EXCL_HEAD + r"$", sentence[start:m.start()]):
            continue
        out.append(sentence[start:m.start()])
        start = m.end()
    out.append(sentence[start:])
    return out


def _commit_decision(story, sentence, cio):
    """Step 2 on one sentence.  Returns (score, path) or None when no commitment names a location.

    The sentence is cut into clauses at ';' and 'but' (not at commas: "look for it, next to the
    sandcastle" keeps its location; see _clauses for the exclusive 'but').  A verb's segment runs
    from the commitment verb to the next commitment verb.  Clauses with a commitment verb are read
    in order; within a clause, the segments that name a location are read in order and the first
    one that decides wins:
    - verb not negated: the first location in the segment decides (path commit / commit_both);
    - verb negated ("will not look ...", "look not ..."), path commit_negated in all four cases:
      (a) exclusive construction ("anywhere / any ... other than / except / besides / apart
          from / aside from / but X"): the first location after the connective decides;
      (b) else an affirmative alternative ("rather" / "instead", not "rather than" / "instead
          of") followed by a location: the first location after it decides;
      (c) else one location named: the other location (each story has two locations);
      (d) else (both named, no exclusive or alternative): the segment is skipped.
    If no segment in the clause decides (a verb names its location only before the verb, "the
    top drawer is where he will look", or every naming segment was skipped under (d)), the first
    location in the clause decides; after (d) that is path commit_both and can be the negated
    location."""
    for clause in _clauses(sentence):
        verbs = list(re.finditer(COMMIT, clause))
        if not verbs:
            continue
        for k, v in enumerate(verbs):
            end = verbs[k + 1].start() if k + 1 < len(verbs) else len(clause)
            seg = clause[v.start():end]
            c, w = _explicit(story, seg, cio)
            if not (c or w):
                continue
            negated = bool(re.search(NEGATION, clause[:v.start()]) or re.match(COMMIT + r"\s+not\b", seg))
            if negated:
                for marker in (EXCLUSIVE, ALTERNATIVE):
                    m = re.search(marker, seg)
                    if m:
                        c2, w2 = _explicit(story, seg[m.end():], cio)
                        if c2 or w2:
                            return _decide(c2, w2), 'commit_negated'
                if c and w:
                    continue
                return (0 if c else 1), 'commit_negated'
            return _decide(c, w), ('commit_both' if (c and w) else 'commit')
        c, w = _explicit(story, clause, cio)
        if c or w:
            return _decide(c, w), ('commit_both' if (c and w) else 'commit')
    return None


def score_ice_cube(text):
    """Catch story 7: correct = not surprised (the ice would have melted)."""
    t = _norm(text).strip()
    if (re.search(r"\b(?:won't|will not|wouldn't|would not|not|isn't|is not|probably not|unlikely to)"
                  r" (?:be )?(?:that |very |too )?su(?:r)?prised", t)
            or re.search(r"\bdon't think (?:he|he'd|he would|he will|he might)\b.{0,20}su(?:r)?prised", t)
            or re.match(r"^\W*(?:no|nope|not really|probably not|maybe not)\b", t)):
        return 1, 'not_surprised'
    return 0, 'other'


def score_fb(story, lst, text, item_text):
    """Returns (score 0/1, path, chosen) where chosen in {'correct','other','none'}.

    1. Sentences that restate the story (share a 15-word run with the item text) are ignored.
    2. The first commitment (look, check, expect, ...) that names a location decides, read per
       clause (see _commit_decision; revisions r1 and r1a).
    3. Otherwise the location named first in the remaining text decides.
    4. Otherwise a description of the original location ("where he put it") counts as correct.
    5. Otherwise 0 (no location given)."""
    if story == 7:
        s, path = score_ice_cube(text)
        return s, path, 'correct' if s else 'other'
    t = _norm(text)
    ig = _grams(_words(_norm(item_text)), ECHO_NGRAM)
    cio = correct_is_orig(story, lst)
    sents = [s for s in _sentences(t) if not (_grams(_words(s), ECHO_NGRAM) & ig)]
    for s in sents:
        hit = _commit_decision(story, s, cio)
        if hit:
            d, path = hit
            return d, path, 'correct' if d else 'other'
    rest = ' \n '.join(sents)
    c, w = _explicit(story, rest, cio)
    if c or w:
        d = _decide(c, w)
        return d, 'first_both' if (c and w) else 'first', 'correct' if d else 'other'
    if cio and re.search(DESC, rest):
        return 1, 'description', 'correct'
    return 0, 'none', 'none'

