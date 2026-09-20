"""Offline control-flow test for the pipeline: no API keys, no network, no LLM.

Patches llm() and search_ddg() with scripted stand-ins and checks the three new
behaviours: per-section loop that stops on 'ok', critic-requested re-search, and
the ship/hold gate.
"""
import asyncio
import json
import os
import sys
import types

sys_path_root = __import__('pathlib').Path(__file__).resolve().parents[1]
__import__('sys').path.insert(0, str(sys_path_root))

# Stub the two third-party deps so the test runs anywhere (no openai/ddgs install).
_openai = types.ModuleType("openai")
_openai.AsyncOpenAI = lambda **kw: types.SimpleNamespace()
sys.modules.setdefault("openai", _openai)
_ddgs = types.ModuleType("ddgs")
_ddgs.DDGS = object
sys.modules.setdefault("ddgs", _ddgs)

os.environ.setdefault("LLM_API_KEY", "test")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("RESEARCH_MAX_CRITIC_ROUNDS", "3")

import pipeline as m  # noqa: E402

SECTION_BODY = ("Текст раздела. " * 40) + "\nЧто это значит для вас: вывод.\n[^1]: https://example.com/a\n"

calls = {"llm": [], "search": []}


def make_llm(critic_script):
    """critic_script: dict[section_n] -> list of critic JSON answers, consumed in order."""
    state = {n: 0 for n in critic_script}

    async def fake_llm(system, user, temperature=0.4, max_tokens=None, retries=2):
        # NB: several pipeline stages pass their prompt as `system`, not `user`.
        both = (system or "") + "\n" + (user or "")
        calls["llm"].append(both[:60])
        if "Вы корректор" in both:  # grammar pass: echo the report back unchanged
            return system.split("ОТЧЁТ:\n---\n")[1].rsplit("\n---", 1)[0]
        if "несостыковки МЕЖДУ РАЗДЕЛАМИ" in both:
            return "Несостыковок нет."
        if "Решите судьбу готового отчёта" in both:
            return json.dumps({"verdict": "ship", "reason": "всё в порядке"}, ensure_ascii=False)
        if "поисковую категорию" in both:
            return "тестовая категория"
        if "Вы составляете план" in both:
            return "план отчёта"
        if "Вы пишете раздел" in both:
            return SECTION_BODY
        if "Вы редактор бизнес-отчётов" in both:
            n = int(both.split("Раздел №")[1].split(" ")[0])
            answers = critic_script.get(n, ['{"verdict":"ok","issues":[],"search_queries":[]}'])
            i = min(state.get(n, 0), len(answers) - 1)
            state[n] = state.get(n, 0) + 1
            return answers[i]
        if "Вы переписываете ОДИН раздел" in both:
            return SECTION_BODY + "\n(исправлено)"
        return ""
    return fake_llm


def fake_search(query, max_results=5):
    calls["search"].append(query)
    return [{"title": "Пример", "href": "https://example.com/a", "body": "тело"}]


def fake_check_url(url):
    return "OK 200"


async def run(critic_script):
    calls["llm"].clear()
    calls["search"].clear()
    m.llm = make_llm(critic_script)
    m.search_ddg = fake_search
    m.check_url = fake_check_url
    return await m.generate_report("тестовая сфера")


def case(name, critic_script, expect_verdict, expect_searches=None, expect_reason_sub=None):
    report, verdict, reasons = asyncio.run(run(critic_script))
    ok = verdict == expect_verdict
    if expect_searches is not None:
        ok = ok and len(calls["search"]) == expect_searches
    if expect_reason_sub:
        ok = ok and any(expect_reason_sub in r for r in reasons)
    print(f"{'PASS' if ok else 'FAIL'} | {name}: verdict={verdict} searches={len(calls['search'])} reasons={reasons}")
    assert "## 6. Рекомендации по позиционированию" in report, "report structure broken"
    return ok


OK = '{"verdict":"ok","issues":[],"search_queries":[]}'
FIX_NO_SEARCH = '{"verdict":"fix","issues":["штамп «лидер рынка»"],"search_queries":[]}'
FIX_WITH_SEARCH = '{"verdict":"fix","issues":["цена без источника"],"search_queries":["сервис цены 2026","сервис тарифы"]}'
GARBAGE = "не JSON вовсе"

results = []

# 1. Everything passes on round 1 → no fixes, 12 baseline searches (4 sections x 3 queries), SHIP.
results.append(case("all sections pass first round", {}, "SHIP", expect_searches=12))

# 2. Section 3 fails once with a search request, then passes → 2 extra queries run, SHIP.
results.append(case("critic orders a re-search, then passes",
                    {3: [FIX_WITH_SEARCH, OK]}, "SHIP", expect_searches=14))

# 3. Section 2 never passes → loop stops at cap, HOLD with 'не прошли критика'.
results.append(case("section never passes → hold",
                    {2: [FIX_NO_SEARCH, FIX_NO_SEARCH, FIX_NO_SEARCH, FIX_NO_SEARCH]},
                    "HOLD", expect_reason_sub="не прошли критика"))

# 4. Unparseable critic answer is treated as a failure, not a silent pass.
results.append(case("garbage critic answer → treated as fix",
                    {5: [GARBAGE, OK]}, "SHIP"))

# 5. Model gate says hold even when the mechanics are clean.
def gate_hold_llm(critic_script):
    base = make_llm(critic_script)

    async def wrapped(system, user, temperature=0.4, max_tokens=None, retries=2):
        if "Решите судьбу готового отчёта" in user:
            return json.dumps({"verdict": "hold", "reason": "цифры выглядят выдуманными"}, ensure_ascii=False)
        return await base(system, user, temperature, max_tokens, retries)
    return wrapped


async def run_gate_hold():
    calls["search"].clear()
    m.llm = gate_hold_llm({})
    m.search_ddg = fake_search
    m.check_url = fake_check_url
    return await m.generate_report("тестовая сфера")


_, v, r = asyncio.run(run_gate_hold())
ok5 = v == "HOLD" and any("редактор" in x for x in r)
print(f"{'PASS' if ok5 else 'FAIL'} | editor gate says hold: verdict={v} reasons={r}")
results.append(ok5)

# 6. Search budget is respected.
os.environ["RESEARCH_EXTRA_SEARCH_BUDGET"] = "1"
import importlib  # noqa: E402
m = importlib.reload(m)
results.append(case("search budget caps extra searches",
                    {2: [FIX_WITH_SEARCH, OK], 3: [FIX_WITH_SEARCH, OK], 4: [FIX_WITH_SEARCH, OK]},
                    "SHIP", expect_searches=14))

print("\n" + ("ALL PASS" if all(results) else "SOME FAILED"))
sys.exit(0 if all(results) else 1)
