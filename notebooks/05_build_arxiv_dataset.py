#!/usr/bin/env python3
import argparse
import contextlib
import ctypes
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import signal
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, OrderedDict

import numpy as np

try:
    import pysbd
except ImportError:
    pysbd = None

ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"
OPEN, CLOSE = "\ue000", "\ue001"

FILE_TIMEOUT_SECONDS = 5 * 60


class FileTimeout(BaseException):
    """Tiempo de espera superado indicado en el parámetro FILE_TIMEOUT_SECONDS."""


def sleep_interruptible(seconds):
    end = time.time() + seconds
    while True:
        left = end - time.time()
        if left <= 0:
            return
        time.sleep(min(left, 0.5))


@contextlib.contextmanager
def file_timeout(seconds):
    if not seconds or seconds <= 0:
        yield
        return

    if hasattr(signal, "SIGALRM"):
        def handler(signum, frame):
            raise FileTimeout(f"timeout de {seconds:g}s superado")

        old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        return

    tid = ctypes.c_ulong(threading.get_ident())
    done = threading.Event()

    def watchdog():
        if not done.wait(seconds):
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(FileTimeout))

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    try:
        yield
    finally:
        done.set()
        t.join()
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)


class FetchError(Exception):
    pass


class TooLarge(Exception):
    pass


class Client:

    def __init__(self, contact, delay=3.0):
        self.ua = f"arxiv-cs-citation-dataset/0.1 (academic research; mailto:{contact})"
        self.delay = max(3.0, float(delay))
        self.last = 0.0

    def get(self, url, max_bytes=None, retries=4):
        for attempt in range(retries + 1):
            wait = self.last + self.delay - time.time()
            if wait > 0:
                sleep_interruptible(wait)
            self.last = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": self.ua})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read(max_bytes + 1) if max_bytes else r.read()
                self.last = time.time()
                if max_bytes and len(data) > max_bytes:
                    raise TooLarge(f">{max_bytes} bytes")
                return data
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    ra = e.headers.get("Retry-After") if e.headers else None
                    pause = float(ra) if ra and ra.isdigit() else 15 * (2 ** attempt)
                    print(f"  HTTP {e.code}; espero {min(pause, 300):.0f}s", file=sys.stderr)
                    sleep_interruptible(min(pause, 300))
                    continue
                raise FetchError(f"HTTP {e.code}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                sleep_interruptible(5 * (attempt + 1))
                last = e
        raise FetchError(f"sin respuesta tras {retries + 1} intentos")


def parse_atom(data):
    root = ET.fromstring(data)
    out = []
    for e in root.findall(ATOM + "entry"):
        url = (e.findtext(ATOM + "id") or "").strip()
        m = re.search(r"/abs/(.+?)(?:v\d+)?$", url)
        if not m:
            continue
        cats = [c.get("term") for c in e.findall(ATOM + "category") if c.get("term")]
        prim = e.find(ARX + "primary_category")
        out.append({
            "id": m.group(1),
            "title": re.sub(r"\s+", " ", e.findtext(ATOM + "title") or "").strip(),
            "abstract": re.sub(r"\s+", " ", e.findtext(ATOM + "summary") or "").strip(),
            "published": (e.findtext(ATOM + "published") or "")[:10],
            "categories": cats,
            "primary_category": prim.get("term") if prim is not None else (cats[0] if cats else ""),
            "authors": [a.findtext(ATOM + "name") for a in e.findall(ATOM + "author")],
        })
    return out


def entries_from_ids(client, ids):
    out = []
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(batch), "max_results": len(batch)})
        out.extend(parse_atom(client.get(url)))
    return out


def _decode(b):
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "ignore")


def read_archive(data, max_unpacked=60 * 1024 * 1024):
    files, total = {}, 0
    bio = io.BytesIO(data)
    if tarfile.is_tarfile(bio):
        bio.seek(0)
        with tarfile.open(fileobj=bio, mode="r:*") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                name = m.name.lstrip("./")
                if not name.lower().endswith((".tex", ".bbl", ".bib")):
                    continue
                total += m.size
                if total > max_unpacked or m.size > 20 * 1024 * 1024:
                    return None
                f = tar.extractfile(m)
                if f:
                    files[name] = _decode(f.read())
        return files or None
    
    try:
        raw = gzip.decompress(data)
        text = _decode(raw)
    except OSError:
        text = _decode(data)

    if "% ===== FILE:" in text:
        parts = re.split(r"%\s*=====\s*FILE:\s*(.*?)\s*=====", text)
        for i in range(1, len(parts), 2):
            files[parts[i].strip()] = parts[i+1]
        if files:
            return files

    if "\\begin{document}" in text or "\\documentclass" in text:
        return {"main.tex": text}
    return None


def read_braced(s, i):
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def remove_command_with_arg(s, names):
    pat = re.compile(r"\\(?:%s)\*?\s*(?:\[[^\]]*\]\s*)*\{" % "|".join(names))
    pos = 0
    while True:
        m = pat.search(s, pos)
        if not m:
            return s
        _, end = read_braced(s, m.end() - 1)
        s = s[:m.start()] + s[end:]
        pos = m.start()


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*$", "", s, flags=re.M)


def find_file(name, files):
    name = name.strip().lstrip("./")
    for cand in (name, name + ".tex"):
        if cand in files:
            return cand
    base = os.path.basename(name)
    for cand in (base, base + ".tex"):
        for k in files:
            if k == cand or k.endswith("/" + cand):
                return k
    return None


def inline_inputs(text, files, depth=0):
    if depth > 5:
        return text

    def repl(m):
        k = find_file(m.group(1), files)
        if not k:
            return ""
        return inline_inputs(strip_comments(files[k]), files, depth + 1)

    return re.sub(r"\\(?:input|include|subfile)\s*\{([^}]+)\}", repl, text)


def pick_main(files):
    cands = [k for k, v in files.items()
             if k.lower().endswith(".tex") and "\\begin{document}" in v]
    if not cands:
        return None
    return max(cands, key=lambda k: len(files[k]))


def mk(kind, payload):
    return f"{OPEN}{kind}:{payload}{CLOSE}"


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


DROP_ENVS = ["figure", "figure\\*", "table", "table\\*", "tabular", "tabular\\*",
             "tabularx", "longtable", "algorithm", "algorithm\\*", "algorithmic",
             "lstlisting", "verbatim", "tikzpicture", "wrapfigure", "sidewaystable",
             "sidewaysfigure", "minted", "comment", "abstract", "IEEEkeywords",
             "filecontents", "filecontents\\*", "subfigure", "minipage"]
MATH_ENVS = ["equation", "equation\\*", "align", "align\\*", "eqnarray", "eqnarray\\*",
             "gather", "gather\\*", "multline", "multline\\*", "displaymath", "math",
             "flalign", "flalign\\*"]
META_CMDS = ["title", "author", "date", "affiliation", "address", "email", "thanks",
             "keywords", "institute", "icmlauthor", "icmlaffiliation", "icmlcorrespondingauthor",
             "footnote", "footnotetext", "footnotemark", "maketitle", "caption",
             "captionof", "tableofcontents"]

CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyearpar|parencite|textcite|"
    r"autocite|footcite|Citep|Citet|Cite|citenum)\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
SEC_RE = re.compile(
    r"\\(chapter|section|subsection|subsubsection|paragraph|subparagraph)(\*?)\s*"
    r"(?:\[[^\]]*\]\s*)?\{")


def replace_math(s):
    s = s.replace("\\$", "\ue002")
    for env in MATH_ENVS:
        s = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % env,
                   lambda m: mk("formula", md5(m.group(0))), s, flags=re.S)
    s = re.sub(r"\$\$.*?\$\$", lambda m: mk("formula", md5(m.group(0))), s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", lambda m: mk("formula", md5(m.group(0))), s, flags=re.S)
    s = re.sub(r"\\\(.*?\\\)", lambda m: mk("formula", md5(m.group(0))), s, flags=re.S)
    s = re.sub(r"\$([^$]{1,500}?)\$", lambda m: mk("formula", md5(m.group(0))), s)
    return s.replace("\ue002", "$")


ACCENT_RE = re.compile(r"""\\["'`^~=.]\s*\{?([A-Za-z])\}?""")


def clean_inline(s):
    s = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", s)
    s = re.sub(r"\\(?:eqref|autoref|cref|Cref|ref|pageref|nameref|vref)\*?\{[^}]*\}", "[REF]", s)
    s = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\url\{[^}]*\}", "[URL]", s)
    s = re.sub(r"\\textcolor\{[^}]*\}\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:label|index|vspace\*?|hspace\*?|includegraphics|bibliographystyle|"
               r"bibliography|pagestyle|thispagestyle|setlength|setcounter|usepackage|"
               r"newblock|bibitem)\s*(?:\[[^\]]*\])?(?:\{[^{}]*\})?", "", s)
    s = re.sub(r"\\begin\{[^}]*\}(?:\[[^\]]*\])?|\\end\{[^}]*\}", " ", s)
    for _ in range(5):  # \cmd{arg} -> arg (de dentro hacia fuera)
        new = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\{\\(?:em|it|bf|sc|sl|tt|rm|sf)\s+([^{}]*)\}", r"\1", s)
    s = ACCENT_RE.sub(r"\1", s)
    for a, b in (("\\&", "&"), ("\\%", "%"), ("\\_", "_"), ("\\#", "#"),
                 ("\\{", "{"), ("\\}", "}"), ("\\ldots", "..."), ("\\dots", "...")):
        s = s.replace(a, b)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("~", " ").replace("``", '"').replace("''", '"')
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"---|--", "-", s)
    return re.sub(r"\s+", " ", s).strip()

def split_sections(body):
    segs, buf = [], []
    top, sub, num = "", "", ""
    counters = [0, 0, 0]
    pos = 0

    def flush():
        txt = "".join(buf)
        if txt.strip():
            segs.append((top, sub, num, txt))
        buf.clear()

    while True:
        m = SEC_RE.search(body, pos)
        if not m:
            buf.append(body[pos:])
            break
        buf.append(body[pos:m.start()])
        title, end = read_braced(body, m.end() - 1)
        kind, star = m.group(1), m.group(2)
        pos = end
        if kind in ("paragraph", "subparagraph"):
            buf.append("\n\n")
            continue
        flush()
        level = {"chapter": 1, "section": 1, "subsection": 2, "subsubsection": 3}[kind]
        t = clean_inline(title)
        if not star:
            counters[level - 1] += 1
            for i in range(level, 3):
                counters[i] = 0
            num = ".".join(str(c) for c in counters[:level])
        else:
            num = ""
        if level == 1:
            top, sub = t, ""
        else:
            sub = t
    flush()
    return segs


def looks_english(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 50:
        return True
    common = {"the", "of", "and", "in", "to", "is", "we", "a", "for", "that", "with"}
    return sum(w in common for w in words) / len(words) > 0.12


def finalize_text(t):
    return t.replace(OPEN, "{{").replace(CLOSE, "}}")



ARXIV_ID = r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})"
ARXIV_KEY_RE = re.compile(r"^arxiv\s*:\s*" + ARXIV_ID + r"(?:v\d+)?$", re.I)
FILE_RE = re.compile(
    r"^(?P<id>\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?_\d{7})(?:v\d+)?-(?P<cat>.+)\.tex$", re.I)


def arxiv_from_key(key):
    m = ARXIV_KEY_RE.match(key.strip())
    return m.group(1) if m else None


def safe(pid):
    return pid.replace("/", "_")


def id_from_filename(name):
    m = FILE_RE.match(name)
    if not m:
        return None
    pid = m.group("id")
    return pid.replace("_", "/", 1) if "_" in pid else pid


def category_from_filename(name):
    m = FILE_RE.match(name)
    return m.group("cat") if m else ""


def read_text(path):
    with open(path, "rb") as f:
        return _decode(f.read())


class LocalSources:

    def __init__(self, dirs):
        self.by_id = {}
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                pid = id_from_filename(name)
                if pid and pid not in self.by_id:
                    self.by_id[pid] = os.path.join(d, name)

    def find(self, pid):
        return self.by_id.get(pid)

    def add(self, pid, path):
        self.by_id[pid] = path


MARK_RE = re.compile(r"\{\{cite:([^{}]+)\}\}")
FORMULA_RE = re.compile(r"\{\{formula:[0-9a-f\-]+\}\}")

HOLD_OPEN, HOLD_CLOSE = "\ue020", "\ue021"
PROT = "\ue010"
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z{\[(" + HOLD_OPEN + r"])")
EG_IE_RE = re.compile(r"\b(?:e\.g|i\.e)\.")
ABBREV_RE = re.compile(
    r"\b(Eqs?|Figs?|Secs?|Tabs?|Refs?|Alg|Thm|Lem|Prop|Cor|Def|Ch|Vol|Dr|Prof|"
    r"eq|fig|cf|vs|pp|resp|approx|viz)\.(?=\s)")
ET_AL_RE = re.compile(r"\bet al\.(?=\s*(?:[a-z0-9,;:)\[(]|\{\{|" + HOLD_OPEN + "))")


def protect_abbreviations(text):
    text = EG_IE_RE.sub(lambda m: m.group(0).replace(".", PROT), text)
    text = ABBREV_RE.sub(lambda m: m.group(1) + PROT, text)
    return ET_AL_RE.sub("et al" + PROT, text)


class SentenceSplitter:
    def __init__(self, kind="auto"):
        if kind == "auto":
            kind = "pysbd" if pysbd is not None else "regex"
            if kind == "regex":
                print("pysbd no está instalado; uso la regex simple. "
                      "Instálalo con: pip install pysbd", file=sys.stderr)
        if kind == "pysbd" and pysbd is None:
            sys.exit("Falta pysbd. Instálalo con: pip install pysbd")
        self.kind = kind
        self.seg = pysbd.Segmenter(language="en", clean=False) if kind == "pysbd" else None

    def __call__(self, text):
        marks = []

        def hold(m):
            marks.append(m.group(0))
            return f"{HOLD_OPEN}{len(marks) - 1}{HOLD_CLOSE}"

        prot = protect_abbreviations(MARK_RE.sub(hold, text))
        sents = None
        if self.seg is not None:
            try:
                sents = self.seg.segment(prot)
            except Exception:
                sents = None
        if sents is None:
            sents = SENT_SPLIT_RE.split(prot)
        restore = re.compile(HOLD_OPEN + r"(\d+)" + HOLD_CLOSE)
        out = []
        for s in sents:
            s = restore.sub(lambda m: marks[int(m.group(1))], s.replace(PROT, ".")).strip()
            if s:
                out.append(s)
        return out


def mask(text, target):
    def repl(m):
        return "#CITATION_TAG" if m.group(1) == target else "#OTHER_CITATION"
    return FORMULA_RE.sub("[MATH]", MARK_RE.sub(repl, text))


def plain_text(text):
    return re.sub(r"\s+", " ", FORMULA_RE.sub("[MATH]", MARK_RE.sub("", text))).strip()

def replace_cites(s, found):
    def repl(m):
        marks = []
        for k in (k.strip() for k in m.group(1).split(",")):
            if not k:
                continue
            aid = arxiv_from_key(k)
            if aid:
                found.add(aid)
            marks.append(mk("cite", aid or "other"))
        return ", ".join(marks)

    s = re.sub(r"\\citeyear\*?\s*(?:\[[^\]]*\]\s*)?\{[^}]*\}", "", s)
    return CITE_RE.sub(repl, s)

HEAD_RE = re.compile(r"\\begin\{abstract\}|\\(?:chapter|section)\*?\s*(?:\[[^\]]*\]\s*)?\{")
INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fname(n):
    base = n.strip().split("/")[-1]
    return base if base.lower().endswith(".tex") else base + ".tex"


def _common_prefix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def split_concatenated(raw_tex):
    m = re.search(r"\\end\{document\}", raw_tex)
    if not m or not raw_tex[m.end():].strip():
        return None
    head, tail = raw_tex[:m.end()], strip_comments(raw_tex[m.end():])
    names = sorted({_fname(n) for n in INPUT_RE.findall(head)})
    if not names:
        return None

    heads = []
    for h in HEAD_RE.finditer(tail):
        if h.group(0).startswith("\\begin"):
            heads.append((h.start(), "abstract"))
        else:
            title, _ = read_braced(tail, h.end() - 1)
            heads.append((h.start(), _norm(title)))

    starts = {}
    if len(heads) == len(names):
        starts = {n: heads[i][0] for i, n in enumerate(names)}
        starts[names[0]] = 0
    else:
        last = -1
        for n in names:
            key = _norm(n[:-4])
            for pos, t in heads:
                if pos > last and _common_prefix(key, t) >= min(4, len(key)):
                    starts[n], last = pos, pos
                    break
    if not starts:
        return None
    order = sorted(starts.items(), key=lambda kv: kv[1])
    files = {}
    for i, (n, pos) in enumerate(order):
        end = order[i + 1][1] if i + 1 < len(order) else len(tail)
        files[n] = tail[pos:end]
    return head, files


def fallback_append(raw_tex):
    m = re.search(r"\\end\{document\}", raw_tex)
    if m and raw_tex[m.end():].strip():
        head, tail = raw_tex[:m.end()], raw_tex[m.end():]
        head = INPUT_RE.sub("", head).replace("\\end{document}", tail + "\n\\end{document}")
        return head, {}
    return None


def parse_local_tex(raw_tex):
    if "% ===== FILE:" in raw_tex:
        parts = re.split(r"%\s*=====\s*FILE:\s*(.*?)\s*=====", raw_tex)
        files = {parts[i].strip(): parts[i + 1] for i in range(1, len(parts), 2)}
        main_file = pick_main(files)
        if main_file:
            return files[main_file], files
    return split_concatenated(raw_tex) or fallback_append(raw_tex) or (raw_tex, {})

FOOT_RE = re.compile(r"\\footnote(?:text)?\*?\s*(?:\[[^\]]*\]\s*)*\{")
SENT_END_RE = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\bal)(?<!\bEq)(?<!\bEqs)(?<!\bFig)(?<!\bFigs)[.!?](?=\s|$)")
PARA_END_RE = re.compile(r"\n\s*\n")

def hoist_footnotes(s):
    pos = 0
    while True:
        m = FOOT_RE.search(s, pos)
        if not m:
            return s
        txt, end = read_braced(s, m.end() - 1)
        s = s[:m.start()] + s[end:]
        if not CITE_RE.search(txt):
            pos = m.start()
            continue
        limit = len(s)
        for rx in (PARA_END_RE, SEC_RE):
            lim = rx.search(s, m.start())
            if lim:
                limit = min(limit, lim.start())
        if re.search(r"[.!?]\s*$", s[:m.start()]):
            ins = m.start()
        else:
            e = SENT_END_RE.search(s, m.start(), limit)
            ins = e.end() if e else limit
        note = " " + txt.strip() + " "
        s = s[:ins] + note + s[ins:]
        pos = ins + len(note)

def build_paragraphs(raw_tex, english_filter=True, require_cites=True, min_paras=5):
    main_tex, files = parse_local_tex(raw_tex)
    text = inline_inputs(strip_comments(main_tex), files)
    
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", text, re.S)
    body = m.group(1) if m else text
    body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", body, flags=re.S)
    body = re.sub(r"\\(?:bibliographystyle|bibliography|printbibliography)\*?\s*(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", body)
    body = body.replace("\\appendix", "")
    for env in DROP_ENVS:
        body = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % env, "\n\n", body, flags=re.S)
    body = hoist_footnotes(body)
    body = remove_command_with_arg(body, META_CMDS)
    body = replace_math(body)
    found = set()
    body = replace_cites(body, found)
    if require_cites and not found:
        return None, "no_arxiv_citations"

    paras = []
    for top, _sub, num, raw in split_sections(body):
        sec = (top, num.split(".")[0] if num else "")
        raw = re.sub(r"\\item(?:\[[^\]]*\])?", "\n\n", raw)
        raw = raw.replace("\\par", "\n\n")
        for para in re.split(r"\n\s*\n", raw):
            t = clean_inline(para)
            plain = re.sub(OPEN + "[^" + CLOSE + "]*" + CLOSE, "", t)
            if len(plain) < 40 or sum(c.isalpha() for c in plain) / max(len(plain), 1) < 0.5:
                continue
            paras.append({"text": finalize_text(t), "sec": sec})
    if len(paras) < min_paras:
        return None, "too_few_paragraphs"
    if english_filter and not looks_english(" ".join(p["text"] for p in paras[:30])):
        return None, "not_english"
    return paras, None


STOP = {"the", "of", "and", "in", "to", "is", "we", "a", "for", "that", "with", "on", "as",
        "by", "this", "are", "be", "an", "it", "our", "from", "at", "which", "can"}


class Embedder:

    def __init__(self, kind):
        self.kind = kind
        self.model = None
        if kind != "lexical":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                sys.exit("Falta sentence-transformers: pip install sentence-transformers "
                         "(o use --embedder lexical para una alternativa sin modelo).")
            self.model = SentenceTransformer(kind)

    def encode(self, texts):
        if self.model is not None:
            v = np.asarray(self.model.encode(texts, convert_to_numpy=True,
                                             show_progress_bar=False), dtype="float32")
        else:
            v = np.zeros((len(texts), 2048), dtype="float32")
            for i, t in enumerate(texts):
                for w in re.findall(r"[a-z]{2,}", t.lower()):
                    if w not in STOP:
                        v[i, zlib.crc32(w.encode()) % 2048] += 1.0
            v = np.log1p(v)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norm, 1e-12)


def dynamic_context(paras, sents_of, p, s, embedder, cache, k, neighbors, min_sim, min_chars):
    lo = p
    while lo - 1 >= max(0, p - neighbors) and paras[lo - 1]["sec"] == paras[p]["sec"]:
        lo -= 1
    hi = p
    while hi + 1 <= min(len(paras) - 1, p + neighbors) and paras[hi + 1]["sec"] == paras[p]["sec"]:
        hi += 1
    cand = []
    for pi in range(lo, hi + 1):
        for si, sent in enumerate(sents_of(pi)):
            if (pi, si) != (p, s) and len(plain_text(sent)) >= min_chars:
                cand.append((pi, si))
    need = [key for key in cand + [(p, s)] if key not in cache]
    if need:
        vecs = embedder.encode([plain_text(sents_of(pi)[si]) for pi, si in need])
        for key, vec in zip(need, vecs):
            cache[key] = vec
    q = cache[(p, s)]

    def pick(keys):
        if not keys:
            return []
        sims = np.array([float(cache[x] @ q) for x in keys])
        order = np.argsort(-sims, kind="stable")[:k]
        return sorted(keys[i] for i in order if min_sim is None or sims[i] >= min_sim)

    return pick([x for x in cand if x < (p, s)]), pick([x for x in cand if x > (p, s)])

def extract_metadata_from_tex(path):
    try:
        with open(path, "rb") as f:
            content = _decode(f.read())
        
        title, abstract = None, None
        
        title_m = re.search(r"\\title\s*(?:\[[^\]]*\]\s*)?\{", content)
        if title_m:
            t_raw, _ = read_braced(content, title_m.end() - 1)
            title = clean_inline(t_raw)
        
        abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", content, re.S | re.I)
        if abs_m:
            abstract = clean_inline(abs_m.group(1))
        
        return title, abstract
    except Exception:
        return None, None

class MetaCache:

    def __init__(self, client, path):
        self.client, self.path = client, path
        self.cache, self.errors = {}, set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    d = json.loads(line)
                    self.cache[d["id"]] = d["entry"]
                except (ValueError, KeyError):
                    pass

    def prefetch(self, ids):
        need = [i for i in dict.fromkeys(ids) if i not in self.cache and i not in self.errors]
        for i in range(0, len(need), 50):
            batch = need[i:i + 50]
            try:
                got = {e["id"]: e for e in entries_from_ids(self.client, batch)}
            except (FetchError, ET.ParseError) as e:
                print(f"  no pude consultar metadatos de {len(batch)} documentos ({e})",
                      file=sys.stderr)
                self.errors.update(batch)
                continue
            with open(self.path, "a", encoding="utf-8") as f:
                for j in batch:
                    self.cache[j] = got.get(j)
                    f.write(json.dumps({"id": j, "entry": self.cache[j]}, ensure_ascii=False) + "\n")

    def get(self, pid):
        if pid not in self.cache and pid not in self.errors:
            self.prefetch([pid])
        if pid in self.errors:
            return None, "error"
        entry = self.cache.get(pid)
        return (entry, "ok") if entry else (None, "not_found")


class CitedDocs:

    PERMANENT = ("no_latex_source", "no_main_tex", "source_too_large", "not_found")

    def __init__(self, args, client, local, meta, failed_path):
        self.args, self.client, self.local, self.meta = args, client, local, meta
        self.failed_path, self.failed, self.memo = failed_path, {}, {}
        if os.path.exists(failed_path) and not args.retry_failed:
            for line in open(failed_path, encoding="utf-8"):
                pid, _, why = line.rstrip("\n").partition("\t")
                if pid:
                    self.failed[pid] = why

    def _fail(self, pid, why):
        if why in self.PERMANENT:
            self.failed[pid] = why
            with open(self.failed_path, "a", encoding="utf-8") as f:
                f.write(f"{pid}\t{why}\n")
        return None, why

    def ensure(self, pid):
        if pid not in self.memo:
            self.memo[pid] = self._ensure(pid)
        return self.memo[pid]

    def _ensure(self, pid):
        path = self.local.find(pid)
        entry = None

        if not path:
            if pid in self.failed:
                return None, self.failed[pid]
            if self.args.no_download:
                return None, "not_local"
            
            entry, status = self.meta.get(pid)
            if status == "error":
                return None, "metadata_error"
            if status == "not_found":
                return self._fail(pid, "not_found")
                
            path, why = self._download(pid, entry)
            if not path:
                return None, why

            print(f"  Paper {pid} descargado de arXiv exitosamente", file=sys.stderr)

        title, abstract = extract_metadata_from_tex(path)
        
        if title and abstract:
            category = category_from_filename(os.path.basename(path))
            return {"id": pid, "category": category, "title": title,
                    "abstract": abstract, "path": path}, "ok"

        if not entry:
            entry, status = self.meta.get(pid)
            if status == "error":
                return None, "metadata_error"
            if status == "not_found":
                return self._fail(pid, "not_found")

        category = entry.get("primary_category") or category_from_filename(os.path.basename(path))
        return {"id": pid, "category": category, "title": entry["title"],
                "abstract": entry["abstract"], "path": path}, "ok"

    def _download(self, pid, entry):
        try:
            data = self.client.get(f"https://arxiv.org/e-print/{pid}",
                                   max_bytes=int(self.args.max_source_mb * 1024 * 1024))
        except TooLarge:
            return self._fail(pid, "source_too_large")
        except FetchError as e:
            if str(e).startswith("HTTP 4"):
                return self._fail(pid, "no_latex_source")
            return None, "fetch_error"
        files = read_archive(data)
        if not files:
            return self._fail(pid, "no_latex_source")
        main = pick_main(files)
        if not main:
            return self._fail(pid, "no_main_tex")
        text = inline_inputs(strip_comments(files[main]), files)
        cat = re.sub(r"[^\w.\-]", "_", entry.get("primary_category") or "unknown")
        os.makedirs(self.args.cited_dir, exist_ok=True)
        path = os.path.join(self.args.cited_dir, f"{safe(pid)}-{cat}.tex")
        with open(path + ".part", "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(path + ".part", path)
        self.local.add(pid, path)
        return path, "ok"


TOP_K = 3


class Ranker:

    def __init__(self, kind, adapter, batch_size=16, max_length=512):
        self.kind, self.adapter = kind, adapter
        self.batch_size, self.max_length = batch_size, max_length
        self.lex = Embedder("lexical") if kind == "lexical" else None
        self.tag = "lexical" if self.lex else f"{kind}|{adapter}"
        if self.lex:
            return
        try:
            import torch
            from transformers import AutoTokenizer
            from adapters import AutoAdapterModel
        except ImportError:
            sys.exit("Faltan dependencias de SPECTER2: pip install torch transformers adapters "
                     "(o use --ranker lexical para una alternativa sin modelo).")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(kind)
        self.model = AutoAdapterModel.from_pretrained(kind)
        self.model.load_adapter(adapter, source="hf", load_as="specter2", set_active=True)
        self.model.to(self.device).eval()

    def encode(self, texts):
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        if self.lex:
            return self.lex.encode(texts)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                inp = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt",
                               return_token_type_ids=False).to(self.device)
                out.append(self.model(**inp).last_hidden_state[:, 0, :].float().cpu().numpy())
        v = np.concatenate(out).astype("float32")
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class ChunkCache:

    def __init__(self, cache_dir, ranker, max_mem=64):
        self.dir = os.path.join(cache_dir, "chunks")
        os.makedirs(self.dir, exist_ok=True)
        self.ranker, self.max_mem, self.mem = ranker, max_mem, OrderedDict()

    def get(self, doc):
        pid = doc["id"]
        if pid in self.mem:
            self.mem.move_to_end(pid)
            return self.mem[pid]
        base = os.path.join(self.dir, safe(pid))
        entry = self._load(base, doc) or self._build(base, doc)
        self.mem[pid] = entry
        while len(self.mem) > self.max_mem:
            self.mem.popitem(last=False)
        return entry

    def _load(self, base, doc):
        try:
            with open(base + ".json", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("ranker") != self.ranker.tag or meta.get("source") != os.path.basename(doc["path"]):
                return None
            chunks = meta["chunks"]
            vecs = np.load(base + ".npy") if chunks else np.zeros((0, 1), dtype="float32")
            return (chunks, vecs) if len(vecs) == len(chunks) else None
        except (OSError, ValueError, KeyError):
            return None

    def _build(self, base, doc):
        paras, _ = build_paragraphs(read_text(doc["path"]), english_filter=False,
                                    require_cites=False, min_paras=1)
        paras = paras or []
        chunks = [{"text": mask(p["text"], None), "section": p["sec"][0]} for p in paras]
        vecs = self.ranker.encode([plain_text(p["text"]) for p in paras])
        if chunks:
            with open(base + ".npy.part", "wb") as f:
                np.save(f, vecs)
            os.replace(base + ".npy.part", base + ".npy")
        with open(base + ".json.part", "w", encoding="utf-8") as f:
            json.dump({"ranker": self.ranker.tag, "source": os.path.basename(doc["path"]),
                       "chunks": chunks}, f, ensure_ascii=False)
        os.replace(base + ".json.part", base + ".json")
        return chunks, vecs


def top_chunks(chunks, vecs, query_vec, k=TOP_K):
    if not chunks:
        return []
    sims = vecs @ query_vec
    order = np.argsort(-sims, kind="stable")[:k]
    return [(chunks[i]["text"], round(float(sims[i]), 4), chunks[i]["section"]) for i in order]


def build_record(pid, citing_cat, p, s, target, sents_of, left, right, doc, top, extra, sec_title):
    sent = sents_of(p)[s]
    ctx = lambda keys: " ".join(mask(sents_of(pi)[si], target) for pi, si in keys)
    rec = {
        "citing_arxiv_id": pid,
        "citing_category": citing_cat,
        "left_context": ctx(left),
        "citing_sentence": mask(sent, target),
        "right_context": ctx(right),
        "cited_arxiv_id": target,
        "cited_category": doc["category"],
        "cited_title": doc["title"],
        "cited_abstract": doc["abstract"],
    }
    for i in range(TOP_K):
        text, score, section = top[i] if i < len(top) else ("", "", "")
        rec[f"chunk_text{i + 1}"] = text
        rec[f"chunk_score{i + 1}"] = score
        rec[f"chunk_section{i + 1}"] = section
    rec["annotator"] = ""
    rec["selected_function"] = ""
    if extra:
        rec["section"] = sec_title
        rec["n_citations_in_sentence"] = len(MARK_RE.findall(sent))
    return rec


class JsonArrayWriter:

    def __init__(self, path):
        self.path = path
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write("[\n\n]")
            self.count = 0
        else:
            with open(path, encoding="utf-8") as f:
                self.count = len(json.load(f))

    def append(self, records):
        if not records:
            return
        body = ",\n".join(json.dumps(r, ensure_ascii=False, indent=1) for r in records)
        with open(self.path, "r+b") as f:
            f.seek(0, os.SEEK_END)
            f.truncate(f.tell() - 2)
            f.seek(0, os.SEEK_END)
            f.write((",\n" if self.count else "").encode("utf-8") + body.encode("utf-8") + b"\n]")
        self.count += len(records)


def process_citing(pid, path, args, split, embedder, cited, ranker, chunk_cache):
    citing_cat = category_from_filename(os.path.basename(path))
    paras, reason = build_paragraphs(read_text(path), not args.no_english_filter)
    if reason:
        return [], reason, Counter()

    cache_s = {}

    def sents_of(i):
        if i not in cache_s:
            cache_s[i] = split(paras[i]["text"])
        return cache_s[i]

    cands = []
    for p, para in enumerate(paras):
        if not MARK_RE.search(para["text"]):
            continue
        for s, sent in enumerate(sents_of(p)):
            targets = [t for t in dict.fromkeys(MARK_RE.findall(sent)) if t != "other" and t != pid]
            if not targets or not (args.min_sentence_chars <= len(plain_text(sent)) <= args.max_sentence_chars):
                continue
            cands.extend((p, s, t) for t in targets)
    if not cands:
        return [], "no_valid_citations", Counter()

    random.Random(f"{args.seed}|{pid}").shuffle(cands)
    cited.meta.prefetch([t for _, _, t in cands][:50])

    records, discards, vec_cache = [], Counter(), {}
    for p, s, target in cands:
        if len(records) >= args.max_per_paper:
            break
        doc, why = cited.ensure(target)
        if doc is None:
            discards[why] += 1
            continue
        left, right = dynamic_context(paras, sents_of, p, s, embedder, vec_cache, args.ctx_k,
                                      args.ctx_neighbors, args.ctx_min_sim, args.ctx_min_chars)

        query = " ".join(plain_text(sents_of(pi)[si]) for pi, si in sorted(left + [(p, s)] + right))
        try:
            chunks, vecs = chunk_cache.get(doc)
            top = top_chunks(chunks, vecs, ranker.encode([query])[0])
        except Exception as e:
            print(f"  {pid} -> sin chunks para {target} ({type(e).__name__}: {e})", file=sys.stderr)
            top = []

        if not top:
            print(f"  {pid} -> {target} descartado porque no produjo chunks", file=sys.stderr)
            discards["no_chunks_generated"] += 1
            continue

        records.append(build_record(pid, citing_cat, p, s, target, sents_of, left, right, doc,
                                    top, args.extra_fields, paras[p]["sec"][0]))
    return records, None, discards


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact", help="tu correo (va en el User-Agent)")
    ap.add_argument("--citing-dir", default="latex-citing", help="documentos origen (.tex)")
    ap.add_argument("--cited-dir", default="latex-cited", help="documentos destino descargados (.tex)")
    ap.add_argument("--output", default="data/citations_dataset.json")
    ap.add_argument("--cache-dir", default="cache", help="metadatos de arXiv y fallos de descarga")
    ap.add_argument("--max-per-paper", type=int, default=40)
    ap.add_argument("--max-papers", type=int, default=0, help="procesa solo N documentos origen (0 = todos)")
    ap.add_argument("--max-source-mb", type=float, default=25.0)
    ap.add_argument("--delay", type=float, default=3.0, help="segundos entre peticiones (mínimo 3)")
    ap.add_argument("--no-download", action="store_true", help="no descarga: usa solo lo que hay en disco")
    ap.add_argument("--retry-failed", action="store_true", help="reintenta destinos que fallaron antes")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="modelo de sentence-transformers para el contexto dinámico, o 'lexical'")
    ap.add_argument("--ranker", default="allenai/specter2_base", help="modelo base de SPECTER2 para rankear chunks, o 'lexical' (solo pruebas)")
    ap.add_argument("--ranker-adapter", default="allenai/specter2", help="adaptador de SPECTER2 (proximity por defecto)")
    ap.add_argument("--ranker-batch", type=int, default=16, help="tamaño de lote para SPECTER2")
    ap.add_argument("--ranker-max-len", type=int, default=512, help="máx. tokens por chunk")
    ap.add_argument("--ctx-k", type=int, default=2, help="oraciones de contexto por lado")
    ap.add_argument("--ctx-neighbors", type=int, default=1, help="párrafos vecinos a cada lado")
    ap.add_argument("--ctx-min-sim", type=float, default=None, help="similitud mínima (opcional)")
    ap.add_argument("--ctx-min-chars", type=int, default=15, help="longitud mínima de una oración candidata")
    ap.add_argument("--splitter", choices=["auto", "pysbd", "regex"], default="auto")
    ap.add_argument("--min-sentence-chars", type=int, default=20)
    ap.add_argument("--max-sentence-chars", type=int, default=1200)
    ap.add_argument("--extra-fields", action="store_true", help="añade section y n_citations_in_sentence a cada registro")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-english-filter", action="store_true")
    ap.add_argument("--timeout", type=float, default=FILE_TIMEOUT_SECONDS, help="segundos máximos por documento origen (0 = sin límite)")
    args = ap.parse_args()

    if not args.no_download and not args.contact:
        ap.error("--contact es obligatorio cuando se descarga (use --no-download para trabajar sin red)")

    split = SentenceSplitter(args.splitter)
    embedder = Embedder(args.embedder)
    ranker = Ranker(args.ranker, args.ranker_adapter, args.ranker_batch, args.ranker_max_len)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.cited_dir, exist_ok=True)

    client = Client(args.contact or "no-download", args.delay)
    local = LocalSources([args.citing_dir, args.cited_dir])
    meta = MetaCache(client, os.path.join(args.cache_dir, "cited_meta.jsonl"))
    cited = CitedDocs(args, client, local, meta, os.path.join(args.cache_dir, "cited_failed.tsv"))
    chunk_cache = ChunkCache(args.cache_dir, ranker)

    todo = []
    for name in sorted(os.listdir(args.citing_dir)):
        if not name.lower().endswith(".tex"):
            continue
        pid = id_from_filename(name)
        if pid is None:
            print(f"'{name}' no sigue el formato {{id}}-{{categoría}}.tex; se omite", file=sys.stderr)
            continue
        todo.append((pid, os.path.join(args.citing_dir, name)))

    done_path, fail_path = args.output + ".done", args.output + ".failed.tsv"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    todo = [(pid, path) for pid, path in todo if pid not in done]
    if args.max_papers:
        todo = todo[:args.max_papers]
    print(f"{len(todo)} documentos origen pendientes | {len(done)} ya procesados | "
          f"segmentador: {split.kind} | embeddings contexto: {embedder.kind} | ranking: {ranker.kind}")

    writer = JsonArrayWriter(args.output)
    papers, discards = Counter(), Counter()

    total_start_time = time.time()
    with open(done_path, "a") as dn, open(fail_path, "a") as fl:
        for n, (pid, path) in enumerate(todo, 1):
            doc_start_time = time.time()
            try:
                print(f"Procesando: {pid}")
                with file_timeout(args.timeout):
                    records, reason, disc = process_citing(pid, path, args, split, embedder, cited, ranker, chunk_cache)
            except FileTimeout:
                records, reason, disc = [], "timeout_exceeded", Counter()
            except Exception as e:
                records, reason, disc = [], f"parse_error:{type(e).__name__}", Counter()
            
            if len(records) == 0 or reason:
                if not reason:
                    reason = "zero_records" 
                
                dest_path = os.path.join(args.cited_dir, os.path.basename(path))
                try:
                    shutil.move(path, dest_path)
                    extra_msg = f" (Movido a {args.cited_dir})"
                except Exception as e:
                    extra_msg = f" (Error al mover: {e})"
            else:
                extra_msg = ""

            writer.append(records)
            discards.update(disc)
            papers[reason or "ok"] += 1
            
            if reason:
                fl.write(f"{pid}\t{reason}\n")
                fl.flush()
                
            dn.write(pid + "\n")
            dn.flush()

            doc_elapsed = time.time() - doc_start_time
            doc_mins, doc_secs = int(doc_elapsed // 60), int(doc_elapsed % 60)
            time_msg = f" (Tiempo: {doc_mins}m {doc_secs}s)"
            
            print(f"  [{n}/{len(todo)}] {pid}: {len(records)} registros{extra_msg}{time_msg}"
                  + (f" | omitido: {reason}" if reason else "")
                  + (f" | citas descartadas: {dict(disc)}" if disc else ""))
    print("Documentos origen:", dict(papers))
    print("Citas descartadas por motivo:", dict(discards))
    print(f"Registros en {args.output}: {writer.count}")

    total_elapsed = time.time() - total_start_time
    t_h = int(total_elapsed // 3600)
    t_m = int((total_elapsed % 3600) // 60)
    t_s = int(total_elapsed % 60)
    
    print(f"Tiempo total del proceso: {t_h}h:{t_m}m:{t_s}s")


if __name__ == "__main__":
    main()
