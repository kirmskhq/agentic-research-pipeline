#!/usr/bin/env python3
"""Agentic market-research pipeline: search-grounded draft, per-section critic loop
with tool-choice re-search, and a ship/hold quality gate.

Given one line of text describing a line of business, the pipeline produces a
six-section market report in Russian, grounded in live web search, and ends with a
machine-readable verdict on whether the result is fit to deliver.

Stages:
  1. outline          — plan the report for this specific business
  2. search category  — condense the raw brief into a 2-4 word search phrase
  3. web search       — 3 queries x 4 results for each of sections 2-5
  4. sections         — all six written in parallel, grounded in the search results
  5. critic loop      — each section judged separately; only failing sections are
                        rewritten; a section leaves the loop the moment it passes
                        (RESEARCH_MAX_CRITIC_ROUNDS is a safety cap, not the exit
                        condition). When a section fails for lack of sourced facts,
                        the critic returns its OWN search queries, those are run,
                        and the section is rewritten with the fresh results.
  6. grammar + audit  — whole-report proofread and cross-section contradiction check
  7. link checks      — every footnote URL is fetched; footnotes whose domain never
                        appeared in the search results are flagged as ungrounded
  8. gate             — deterministic checks + an editor model decide SHIP or HOLD.
                        HOLD means "do not deliver, call a human": it is written into
                        the report's internal block, printed to stderr, optionally
                        written to $RESEARCH_VERDICT_FILE, and returned as exit code 3.

Usage:
  python pipeline.py "ваша сфера"

Env:
  LLM_API_KEY                               — required
  LLM_BASE_URL   (default Yandex AI Studio) — any OpenAI-compatible endpoint
  LLM_MODEL      (default deepseek-v32)     — model id for that endpoint
  YANDEX_FOLDER_ID                          — only for Yandex-style gpt://<folder>/<model> ids
  RESEARCH_MAX_CRITIC_ROUNDS   (default 3)  — hard cap on critic->fixer rounds
  RESEARCH_EXTRA_SEARCH_BUDGET (default 6)  — critic-requested search batches per run
  RESEARCH_HOLD_BROKEN_URLS    (default 2)  — this many dead footnote URLs -> hold
  RESEARCH_HOLD_UNGROUNDED     (default 3)  — this many off-search footnotes -> hold
  RESEARCH_VERDICT_FILE        (optional)   — path to write "SHIP"/"HOLD + reasons"
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from openai import AsyncOpenAI

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

API_KEY = os.environ.get("LLM_API_KEY") or os.environ["YANDEX_API_KEY"]
BASE_URL = os.environ.get("LLM_BASE_URL", "https://llm.api.cloud.yandex.net/v1/")
FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")
# Yandex AI Studio addresses models as gpt://<folder>/<model>; other OpenAI-compatible
# endpoints take a plain model id. Set LLM_MODEL to override either way.
MODEL = os.environ.get("LLM_MODEL") or (
    f"gpt://{FOLDER_ID}/deepseek-v32/latest" if FOLDER_ID else "deepseek-chat"
)

MAX_CRITIC_ROUNDS = int(os.environ.get("RESEARCH_MAX_CRITIC_ROUNDS", "3"))
EXTRA_SEARCH_BUDGET = int(os.environ.get("RESEARCH_EXTRA_SEARCH_BUDGET", "6"))
HOLD_BROKEN_URLS = int(os.environ.get("RESEARCH_HOLD_BROKEN_URLS", "2"))
HOLD_UNGROUNDED = int(os.environ.get("RESEARCH_HOLD_UNGROUNDED", "3"))
VERDICT_FILE = os.environ.get("RESEARCH_VERDICT_FILE", "")

MIN_SECTION_CHARS = 400  # deterministic "раздел пустой/куцый" check

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)


async def llm(system: str, user: str, temperature: float = 0.4, max_tokens: int | None = None, retries: int = 2) -> str:
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    for attempt in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < retries:
                wait = 3 * (attempt + 1)
                print(f"[llm] error ({e!r}); retry {attempt+1}/{retries} in {wait}s", file=sys.stderr, flush=True)
                await asyncio.sleep(wait)
            else:
                raise


def search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """Synchronous DDG search. Returns [{title, href, body}, ...]."""
    try:
        with DDGS() as ddg:
            return list(ddg.text(query=query, region="ru-ru", max_results=max_results))
    except Exception as e:
        print(f"[DDG ERROR] {query!r}: {e}", file=sys.stderr)
        return []


def check_url(url: str) -> str:
    """Sync HEAD (falling back to GET) check. Returns 'OK <code>' or 'FAIL <reason>'."""
    try:
        url.encode("ascii")
    except UnicodeEncodeError:
        # percent-encode raw non-ASCII chars (e.g. Cyrillic in a query string)
        parts = urllib.parse.urlsplit(url)
        path = urllib.parse.quote(parts.path)
        query = urllib.parse.quote(parts.query, safe="=&")
        url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))
    req_headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchPipelineBot/1.0)"}
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers=req_headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                return f"OK {resp.status}"
        except urllib.error.HTTPError as e:
            if e.code == 405 and method == "HEAD":
                continue
            return f"FAIL HTTP {e.code}"
        except Exception as e:
            return f"FAIL {e}"
    return "FAIL HTTP 405"


def _domain(url: str) -> str:
    """netloc without leading 'www.', lowercased; '' on failure."""
    try:
        d = urllib.parse.urlsplit(url).netloc.lower()
    except Exception:
        return ""
    return d[4:] if d.startswith("www.") else d


async def search_for_section(queries: list[str], max_per_query: int = 4) -> str:
    """Run multiple DDG searches in parallel (via asyncio.to_thread), format for LLM."""
    tasks = [asyncio.to_thread(search_ddg, q, max_per_query) for q in queries]
    results_lists = await asyncio.gather(*tasks)

    formatted = []
    for q, results in zip(queries, results_lists):
        formatted.append(f"### Поиск: «{q}»")
        if not results:
            formatted.append("(ничего не найдено)")
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            href = r.get("href", r.get("url", ""))
            body = r.get("body", "").replace("\n", " ")[:300]
            formatted.append(f"{i}. **{title}** — {href}\n   {body}")
        formatted.append("")
    return "\n".join(formatted)


def parse_json_object(text: str) -> dict | None:
    """Extract the first JSON object from an LLM answer. Tolerates ``` fences and
    surrounding prose. Returns None when nothing parseable is there."""
    if not text:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])
    for c in candidates:
        try:
            parsed = json.loads(c)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ───────────────────────────────────────────────────────────────────────────
# Section titles — MUST match /research/ UI numbered list
# ───────────────────────────────────────────────────────────────────────────
SECTION_TITLES = [
    "Описание сферы и ключевая терминология",
    "Топ существующих игроков (РФ + 2-3 глобальных)",
    "Ценовая раскладка и модели монетизации",
    "Портрет клиента и чем он пользуется сейчас",
    "Незакрытые сегменты и пробелы рынка",
    "Рекомендации по позиционированию",
]

# Search query templates per section. Empty list = no web search for that section.
SEARCH_QUERIES = {
    2: [
        "{category} сервисы россия 2024",
        "{category} приложения аналоги",
        "{category} зарубежные компании конкуренты",
    ],
    3: [
        "{category} цены подписка тариф",
        "{category} сколько стоит платный",
        "{category} freemium модель монетизации",
    ],
    4: [
        "{category} как сейчас решают пользователи",
        "{category} жалобы отзывы пользователей",
        "{category} аналоги ручные способы",
    ],
    5: [
        "{category} проблемы существующих решений",
        "{category} чего не хватает на рынке",
        "{category} незакрытые сегменты возможности",
    ],
}


CATEGORY_PROMPT = """Сформулируйте короткую (2-4 слова) поисковую категорию для
описанной сферы — такую фразу, по которой в поисковике можно найти статьи о
конкурентах, ценах и отзывах пользователей в этой сфере.

Пример: «Локальный сервис знакомств для владельцев собак и людей, которым
нравятся собаки. Telegram WebApp, визуально похожий на дейтинг-приложение...»
→ сервис прогулок с собаками

Сфера:
{niche}

Ответьте ТОЛЬКО короткой категорией (2-4 слова), без кавычек и пояснений."""


OUTLINE_PROMPT = """Вы составляете план отчёта по сфере для основателя стартапа, который
впервые погружается в анализ рынка. Читатель без опыта — терминам нужно
объяснение, выводы должны быть конкретными.

Структура отчёта — 6 разделов:
1. Описание сферы и ключевая терминология
2. Топ существующих игроков (РФ + 2-3 глобальных)
3. Ценовая раскладка и модели монетизации
4. Портрет клиента и чем он пользуется сейчас
5. Незакрытые сегменты и пробелы рынка
6. Рекомендации по позиционированию

Верните план обычным текстом: заголовок раздела + 2-3 пункта под каждым,
что именно туда войдёт применительно к этой конкретной сфере.

Сфера основателя:
{niche}
"""


SECTION_PROMPT = """Вы пишете раздел №{n} «{title}» отчёта для основателя стартапа БЕЗ ОПЫТА
в анализе рынка.

Сфера основателя:
{niche}

Краткий план целого отчёта:
{outline}

{search_context}

Правила:
— Обращение «вы», не «ты».
— Термины из бизнес-лексики объясняйте при первом использовании
  (например: «retention — доля пользователей, возвращающихся через месяц»).
— Конкретика лучше общих формулировок. Числа, имена компаний, ссылки.
  Один конкретный факт лучше пяти расплывчатых утверждений.
— Если данных нет — честно: «по открытым источникам найти не удалось»,
  «оценочно», «по приблизительной оценке». НИКОГДА не выдумывайте
  цифры, названия компаний, размеры рынка.
— Запрещены штампы: «инновационный», «уникальный», «революционный»,
  «лидер рынка», «прорывной», «передовой», «комплексное решение».
— Источники цитируйте сносками: ...факт[^1]. В конце раздела:
    [^1]: <URL>
    [^2]: <URL>
— Длина: 2-4 абзаца. Короткий честный раздел лучше длинного с водой.
— Пишите обычным текстом, без markdown-выделения: не используйте
  **жирный** или *курсив* для терминов и фраз.
— В самом конце раздела — отдельная строка без выделения:
  «Что это значит для вас:» + 1-2 практических вывода для основателя.

Напишите ТОЛЬКО раздел №{n} «{title}». Без введения, без повтора
заголовка (заголовок будет добавлен снаружи).
"""


SECTION_CRITIC_PROMPT = """Вы редактор бизнес-отчётов. Проверьте ОДИН раздел отчёта
по 5 критериям:

1. Выдуманные факты — имена компаний / числа / даты без источника
   или пометки «оценочно».
2. Штампы — «инновационный», «уникальный», «лидер», «прорывной»,
   «комплексный», «передовой».
3. Размытые формулировки там, где должна быть конкретика
   («многие пользователи», «довольно высокий процент» без цифр).
4. Пустой или одно-фразный раздел. Норма — 2-4 абзаца.
5. Markdown-выделение внутри текста — **жирный** или *курсив*
   (заголовки и сноски [^N] не считаются).

Раздел №{n} «{title}»:
---
{text}
---

{search_note}

Если раздел можно выпускать — verdict = "ok".
Если нет — verdict = "fix", а в issues перечислите проблемы: дословная
цитата + чем плоха.

ОСОБЫЙ СЛУЧАЙ. Если проблема в том, что фактов не хватает или они без
источника, и исправить это можно только новыми данными — добавьте в
search_queries 1-3 поисковых запроса НА РУССКОМ, которые вы бы сами
ввели в поисковике, чтобы закрыть именно этот пробел. Запросы должны
быть конкретными (с названием компании, цены, года), а не общими.
Если новых данных не требуется — search_queries оставьте пустым.

Ответьте ТОЛЬКО JSON без пояснений и без ``` :
{{"verdict": "ok" | "fix", "issues": ["..."], "search_queries": ["..."]}}
"""


SECTION_FIXER_PROMPT = """Вы переписываете ОДИН раздел отчёта по замечаниям редактора.

Раздел №{n} «{title}».
Сфера основателя: {niche}

ТЕКУЩИЙ ТЕКСТ РАЗДЕЛА:
---
{text}
---

ЗАМЕЧАНИЯ РЕДАКТОРА:
{issues}

{search_block}

Правила:
— Исправьте ВСЕ места, на которые указал редактор.
— Не добавляйте НОВЫХ выдуманных фактов. Если факта нет в данных поиска —
  напишите «по открытым источникам найти не удалось».
— Новые факты берите из данных поиска выше и цитируйте сносками [^N]
  с URL в конце раздела. Уже имеющиеся сноски сохраните.
— Без markdown-выделения в тексте (**жирный**, *курсив*).
— Тон: основатель без опыта, обращение «вы», термины объясняйте.
— В конце раздела оставьте строку «Что это значит для вас:» с 1-2 выводами.

Верните ТОЛЬКО исправленный текст раздела, без заголовка и без пояснений.
"""


AUDIT_PROMPT = """Вы проверяете финальный отчёт на несостыковки МЕЖДУ РАЗДЕЛАМИ.

Найдите все упомянутые в отчёте компании/продукты/конкуренты (имена
собственные). Для каждого имени, которое встречается в нескольких
разделах, проверьте: не противоречат ли утверждения о нём друг другу?
Типичный случай — в одном разделе сказано «информация не найдена» /
«по открытым источникам не обнаружено» про X, а в другом разделе про
тот же X приведены конкретные цифры, цены, ссылки или другие детали.

Формат ответа: нумерованный список найденных несостыковок — имя +
короткие цитаты из обоих разделов + в чём конфликт.

Если несостыковок нет — ответьте: «Несостыковок нет.»

ОТЧЁТ:
---
{report}
---
"""


GRAMMAR_PROMPT = """Вы корректор. Исправьте ТОЛЬКО грамматические и
орфографические ошибки в тексте отчёта — например, неправильные формы
повелительного наклонения («арендайте» → «арендуйте»), падежные и
согласовательные ошибки, опечатки.

НЕ меняйте:
— структуру, заголовки разделов (## N. ...), их порядок;
— факты, цифры, названия компаний, ссылки;
— сноски [^N] и их содержимое — формат и количество должны остаться теми же;
— стиль, формулировки и длину там, где нет ошибки.

Если ошибок нет — верните отчёт без изменений.

Верните ПОЛНЫЙ текст отчёта (с исправлениями или без, если ошибок не было).

ОТЧЁТ:
---
{report}
---
"""


GATE_PROMPT = """Вы главный редактор. Решите судьбу готового отчёта: можно ли
отправлять его платящему клиенту или нужно придержать и позвать автора.

Придержать («hold») следует, если в отчёте есть хотя бы одно:
— факты, которые выглядят выдуманными (компании, цифры, даты без источника);
— раздел, который фактически пуст или не отвечает на свой заголовок;
— прямые противоречия между разделами;
— текст обрывается на середине.

Мелкие стилистические шероховатости — это НЕ причина придерживать.

Технические результаты автоматических проверок:
{tech_summary}

ОТЧЁТ:
---
{report}
---

Ответьте ТОЛЬКО JSON без пояснений и без ``` :
{{"verdict": "ship" | "hold", "reason": "одно предложение по-русски"}}
"""


async def generate_report(niche: str) -> tuple[str, str, list[str]]:
    """Returns (report_markdown, verdict, reasons). verdict is 'SHIP' or 'HOLD'."""
    t0 = time.time()
    print(f"=== Сфера: {niche!r} (model: {MODEL}) ===", file=sys.stderr)

    print("[1/N] Outline...", file=sys.stderr, flush=True)
    outline = await llm(OUTLINE_PROMPT.format(niche=niche), f"Сфера: {niche}")
    print(f"      ✓ {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    print("[category] extracting search category...", file=sys.stderr, flush=True)
    search_category = (await llm(CATEGORY_PROMPT.format(niche=niche), "Категория:", temperature=0.2)).strip().strip('"«»').rstrip(".")
    print(f"          ✓ {search_category!r} ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)

    # Pre-search for sections that need it (2, 3, 4, 5)
    print(f"[search] DDG queries for sections {sorted(SEARCH_QUERIES.keys())}...", file=sys.stderr, flush=True)
    search_contexts: dict[int, str] = {}
    for n, q_templates in SEARCH_QUERIES.items():
        queries = [q.format(category=search_category) for q in q_templates]
        ctx = await search_for_section(queries, max_per_query=4)
        search_contexts[n] = ctx
        print(f"         ✓ section {n}: {len(ctx)} chars", file=sys.stderr, flush=True)
    print(f"         ✓ search done at {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    print("[sections] generating in parallel...", file=sys.stderr, flush=True)

    async def gen_section(n: int, title: str) -> str:
        search_block = ""
        if n in search_contexts:
            search_block = (
                "=== ДАННЫЕ ИЗ ПОИСКА (используйте имена/факты отсюда, "
                "цитируйте URL как Markdown-footnote) ===\n"
                + search_contexts[n]
                + "\n=== КОНЕЦ ДАННЫХ ПОИСКА ===\n"
            )
        return await llm(
            "Вы аналитик рынка, пишущий для founder'а без опыта. Конкретно, с числами, "
            "без штампов и без выдумки. Источники — Markdown-footnotes.",
            SECTION_PROMPT.format(
                n=n,
                title=title,
                outline=outline,
                niche=niche,
                search_context=search_block,
            ),
        )

    tasks = [gen_section(i + 1, title) for i, title in enumerate(SECTION_TITLES)]
    section_texts = await asyncio.gather(*tasks)
    print(f"           ✓ {time.time()-t0:.1f}s total", file=sys.stderr, flush=True)

    sections: dict[int, str] = {i + 1: (t or "").strip() for i, t in enumerate(section_texts)}

    # ───────────────────────────────────────────────────────────────────────
    # Per-section critic → (optional re-search) → fixer loop.
    # A section leaves the loop as soon as its critic says "ok"; the round
    # cap only stops runaway loops.
    # ───────────────────────────────────────────────────────────────────────
    history: dict[int, dict] = {n: {"rounds": 0, "issues": 0, "extra_queries": []} for n in sections}
    searches_used = 0
    pending = set(sections)
    rounds_done = 0

    async def critique_section(n: int) -> dict:
        search_note = (
            "Для этого раздела уже запускался веб-поиск, его результаты были у автора."
            if n in search_contexts else
            "Для этого раздела веб-поиск не запускался."
        )
        raw = await llm(
            "Вы придирчивый редактор. Отвечаете строго JSON.",
            SECTION_CRITIC_PROMPT.format(
                n=n, title=SECTION_TITLES[n - 1], text=sections[n], search_note=search_note
            ),
            temperature=0.3,
        )
        parsed = parse_json_object(raw)
        if not parsed:
            # Unparseable critique: treat the raw text as one issue rather than
            # silently passing the section.
            return {"verdict": "fix", "issues": [raw.strip()[:800]], "search_queries": []}
        verdict = str(parsed.get("verdict", "fix")).strip().lower()
        issues = [str(x) for x in parsed.get("issues", []) if str(x).strip()]
        queries = [str(q).strip() for q in parsed.get("search_queries", []) if str(q).strip()][:3]
        if verdict != "ok" and not issues:
            issues = ["Редактор не пояснил замечание — перепроверьте раздел на выдумки и воду."]
        return {"verdict": "ok" if verdict == "ok" else "fix", "issues": issues, "search_queries": queries}

    async def fix_section(n: int, issues: list[str], extra_ctx: str) -> str:
        search_block = ""
        ctx = (search_contexts.get(n, "") + ("\n" + extra_ctx if extra_ctx else "")).strip()
        if ctx:
            search_block = (
                "=== ДАННЫЕ ИЗ ПОИСКА (используйте имена/факты отсюда, "
                "цитируйте URL как Markdown-footnote) ===\n"
                + ctx
                + "\n=== КОНЕЦ ДАННЫХ ПОИСКА ===\n"
            )
        return (await llm(
            "Вы переписываете раздел отчёта по замечаниям редактора. Без выдумки, без воды.",
            SECTION_FIXER_PROMPT.format(
                n=n,
                title=SECTION_TITLES[n - 1],
                niche=niche,
                text=sections[n],
                issues="\n".join(f"— {i}" for i in issues),
                search_block=search_block,
            ),
            temperature=0.3,
        )).strip()

    while pending and rounds_done < MAX_CRITIC_ROUNDS:
        rounds_done += 1
        order = sorted(pending)
        print(f"[critic] round {rounds_done}/{MAX_CRITIC_ROUNDS}, sections {order}...", file=sys.stderr, flush=True)
        verdicts = await asyncio.gather(*[critique_section(n) for n in order])

        to_fix: list[tuple[int, dict]] = []
        for n, v in zip(order, verdicts):
            if v["verdict"] == "ok":
                print(f"         ✓ section {n}: ok", file=sys.stderr, flush=True)
            else:
                print(f"         ✗ section {n}: {len(v['issues'])} замечаний, "
                      f"{len(v['search_queries'])} запросов на догуглить", file=sys.stderr, flush=True)
                to_fix.append((n, v))

        if not to_fix:
            pending = set()
            break

        # Critic-requested searches: the model picks the queries, we run them.
        extra_contexts: dict[int, str] = {}
        for n, v in to_fix:
            queries = v["search_queries"]
            if not queries:
                continue
            if searches_used >= EXTRA_SEARCH_BUDGET:
                print(f"         ! search budget spent, skipping re-search for section {n}", file=sys.stderr, flush=True)
                continue
            searches_used += 1
            print(f"[research] section {n}: критик заказал поиск {queries}", file=sys.stderr, flush=True)
            fresh = await search_for_section(queries, max_per_query=4)
            extra_contexts[n] = fresh
            search_contexts[n] = (search_contexts.get(n, "") + "\n" + fresh).strip()
            history[n]["extra_queries"].extend(queries)

        fixed = await asyncio.gather(*[
            fix_section(n, v["issues"], extra_contexts.get(n, "")) for n, v in to_fix
        ])
        for (n, v), text in zip(to_fix, fixed):
            if text:
                sections[n] = text
            history[n]["rounds"] += 1
            history[n]["issues"] += len(v["issues"])

        pending = {n for n, _ in to_fix}  # re-check what we just rewrote
        print(f"         ✓ round {rounds_done} done at {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    unresolved = sorted(pending)
    if unresolved:
        print(f"[critic] sections still failing after {MAX_CRITIC_ROUNDS} rounds: {unresolved}", file=sys.stderr, flush=True)

    draft = f"# Анализ и отчёт по сфере «{niche}»\n\n"
    for n, title in enumerate(SECTION_TITLES, start=1):
        draft += f"## {n}. {title}\n\n{sections[n].strip()}\n\n"

    print("[grammar] proofreading...", file=sys.stderr, flush=True)
    final = await llm(GRAMMAR_PROMPT.format(report=draft), "Исправьте отчёт.", temperature=0.2)
    print(f"         ✓ {time.time()-t0:.1f}s total", file=sys.stderr, flush=True)

    print("[audit] checking cross-section consistency...", file=sys.stderr, flush=True)
    audit = await llm(AUDIT_PROMPT.format(report=final), "Проверьте отчёт.", temperature=0.2)
    print(f"        ✓ {time.time()-t0:.1f}s total", file=sys.stderr, flush=True)

    print("[urlcheck] checking footnote URLs...", file=sys.stderr, flush=True)
    footnote_urls = re.findall(r"^\[\^(\d+)\]:\s*<?([^\s>]+)>?\s*$", final, re.MULTILINE)
    url_statuses = await asyncio.gather(*[asyncio.to_thread(check_url, url) for _, url in footnote_urls])
    broken_urls = [(num, url, status) for (num, url), status in zip(footnote_urls, url_statuses) if status.startswith("FAIL")]
    print(f"          ✓ {len(footnote_urls)} ссылок, {len(broken_urls)} проблемных, "
          f"{time.time()-t0:.1f}s total", file=sys.stderr, flush=True)

    print("[grounding] checking footnote↔search grounding (sections 2-5)...", file=sys.stderr, flush=True)
    search_domains = {
        n: {_domain(h) for h in re.findall(r"\*\* — (\S+)", ctx)}
        for n, ctx in search_contexts.items()
    }
    headers = list(re.finditer(r"^## (\d+)\. .*$", final, re.MULTILINE))
    section_blocks = {
        int(headers[i].group(1)): final[headers[i].end(): (headers[i + 1].start() if i + 1 < len(headers) else len(final))]
        for i in range(len(headers))
    }
    ungrounded = []
    for n, domains in search_domains.items():
        block = section_blocks.get(n, "")
        for num, url in re.findall(r"^\[\^(\d+)\]:\s*<?([^\s>]+)>?\s*$", block, re.MULTILINE):
            if _domain(url) and _domain(url) not in domains:
                ungrounded.append((n, num, url))
    print(f"            ✓ {len(ungrounded)} вне результатов поиска, "
          f"{time.time()-t0:.1f}s total", file=sys.stderr, flush=True)

    # ───────────────────────────────────────────────────────────────────────
    # Ship / hold gate: deterministic checks first, then the model's verdict.
    # ───────────────────────────────────────────────────────────────────────
    audit_clean = audit.strip()
    audit_ok = audit_clean.lower().startswith("несостыковок нет")
    missing_sections = [n for n in range(1, len(SECTION_TITLES) + 1) if n not in section_blocks]
    short_sections = [n for n, b in section_blocks.items() if len(b.strip()) < MIN_SECTION_CHARS]

    reasons: list[str] = []
    if unresolved:
        reasons.append(f"разделы {unresolved} не прошли критика за {MAX_CRITIC_ROUNDS} круга")
    if missing_sections:
        reasons.append(f"в отчёте нет разделов {missing_sections}")
    if short_sections:
        reasons.append(f"слишком короткие разделы {sorted(short_sections)} (<{MIN_SECTION_CHARS} символов)")
    if not audit_ok:
        reasons.append("аудит нашёл противоречия между разделами")
    if len(broken_urls) >= HOLD_BROKEN_URLS:
        reasons.append(f"нерабочих ссылок в сносках: {len(broken_urls)}")
    if len(ungrounded) >= HOLD_UNGROUNDED:
        reasons.append(f"сносок вне результатов поиска: {len(ungrounded)}")

    tech_summary = (
        f"- разделы, не прошедшие критика: {unresolved or 'нет'}\n"
        f"- пропущенные разделы: {missing_sections or 'нет'}\n"
        f"- короткие разделы: {sorted(short_sections) or 'нет'}\n"
        f"- противоречия между разделами: {'нет' if audit_ok else 'есть'}\n"
        f"- нерабочих ссылок: {len(broken_urls)} из {len(footnote_urls)}\n"
        f"- сносок вне результатов поиска: {len(ungrounded)}"
    )

    print("[gate] final ship/hold decision...", file=sys.stderr, flush=True)
    gate_raw = await llm(
        "Вы главный редактор. Отвечаете строго JSON.",
        GATE_PROMPT.format(tech_summary=tech_summary, report=final),
        temperature=0.2,
    )
    gate = parse_json_object(gate_raw) or {}
    gate_verdict = str(gate.get("verdict", "")).strip().lower()
    gate_reason = str(gate.get("reason", "")).strip()
    if gate_verdict == "hold":
        reasons.append(f"редактор: {gate_reason or 'без пояснения'}")
    elif gate_verdict != "ship":
        reasons.append("редактор не вернул внятный вердикт — нужна ручная проверка")

    verdict = "HOLD" if reasons else "SHIP"
    print(f"[gate] VERDICT={verdict} reasons={reasons or ['-']} ({time.time()-t0:.1f}s total)",
          file=sys.stderr, flush=True)

    loop_note = ", ".join(
        f"р.{n}: {h['rounds']} правк(и), {len(h['extra_queries'])} доп. запрос(ов)"
        for n, h in sorted(history.items()) if h["rounds"] or h["extra_queries"]
    ) or "правок не потребовалось"
    final += (f"\n---\n_{MODEL} • {time.time()-t0:.1f} сек • "
              f"кругов критика: {rounds_done} • {loop_note}_\n")

    warn_lines = ["\n---\n## ⚠️ Проверка перед отправкой (для автора, удалить перед отправкой клиенту)\n"]
    warn_lines.append(f"- ВЕРДИКТ: {'ВЫПУСКАТЬ' if verdict == 'SHIP' else 'ПРИДЕРЖАТЬ — нужна ручная проверка'}")
    if reasons:
        for r in reasons:
            warn_lines.append(f"  - {r}")
    if gate_reason:
        warn_lines.append(f"- Комментарий редактора: {gate_reason}")
    warn_lines.append(f"- Кругов критика: {rounds_done} из {MAX_CRITIC_ROUNDS}; {loop_note}")
    extra_q = [q for h in history.values() for q in h["extra_queries"]]
    if extra_q:
        warn_lines.append("- Критик сам заказал поисковые запросы:")
        for q in extra_q:
            warn_lines.append(f"  - «{q}»")
    if audit_ok:
        warn_lines.append("- Несостыковки между разделами: не найдено.")
    else:
        warn_lines.append("- Несостыковки между разделами:\n" + audit_clean)
    if broken_urls:
        warn_lines.append("- Проблемные ссылки в сносках:")
        for num, url, status in broken_urls:
            warn_lines.append(f"  - [^{num}]: {url} — {status}")
    else:
        warn_lines.append(f"- Ссылки в сносках: все {len(footnote_urls)} доступны.")
    if ungrounded:
        warn_lines.append("- Сноски вне результатов поиска (источник может быть из общих знаний модели — проверьте вручную):")
        for n, num, url in ungrounded:
            warn_lines.append(f"  - Раздел {n}, [^{num}]: {url}")
    else:
        warn_lines.append("- Сноски в разделах 2-5: все источники присутствуют среди результатов поиска.")
    final += "\n".join(warn_lines) + "\n"

    return final, verdict, reasons


async def main():
    if len(sys.argv) < 2:
        print("Usage: pipeline.py 'ваша сфера'", file=sys.stderr)
        sys.exit(1)
    niche = sys.argv[1]
    report, verdict, reasons = await generate_report(niche)
    print(report)

    if VERDICT_FILE:
        try:
            with open(VERDICT_FILE, "w", encoding="utf-8") as f:
                f.write(verdict + "\n")
                for r in reasons:
                    f.write(f"- {r}\n")
        except Exception as e:
            print(f"[gate] could not write verdict file {VERDICT_FILE!r}: {e}", file=sys.stderr)

    # 0 = ship, 3 = hold. The caller decides what to do with a hold; nothing in
    # the current webhook wiring looks at this yet.
    sys.exit(0 if verdict == "SHIP" else 3)


if __name__ == "__main__":
    asyncio.run(main())
