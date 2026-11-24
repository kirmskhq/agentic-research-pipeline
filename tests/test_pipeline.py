"""Offline control-flow test for the pipeline: no API keys, no network, no LLM.

Patches llm() and search_ddg() with scripted stand-ins and checks the behaviours
that matter: per-section loop that stops on 'ok', every rewrite re-checked,
critic-requested re-search, report-wide footnote numbering, review notes kept out
of the client report, and the ship/hold gate.
"""
import asyncio
import json
import os
import re
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

REAL_LLM = m.llm  # kept before the cases below swap in scripted stand-ins

SECTION_BODY = ("Текст раздела. " * 40) + "\nЧто это значит для вас: вывод[^1].\n[^1]: https://example.com/a\n"

calls = {"llm": [], "search": [], "critic": {}, "fix": {}}


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
            calls["critic"][n] = calls["critic"].get(n, 0) + 1
            answers = critic_script.get(n, ['{"verdict":"ok","issues":[],"search_queries":[]}'])
            i = min(state.get(n, 0), len(answers) - 1)
            state[n] = state.get(n, 0) + 1
            return answers[i]
        if "Вы переписываете ОДИН раздел" in both:
            n = int(both.split("Раздел №")[1].split(" ")[0])
            calls["fix"][n] = calls["fix"].get(n, 0) + 1
            return SECTION_BODY + "\n(исправлено)"
        return ""
    return fake_llm


def fake_search(query, max_results=5):
    calls["search"].append(query)
    return [{"title": "Пример", "href": "https://example.com/a", "body": "тело"}]


def fake_check_url(url):
    return "OK 200"


def reset_calls():
    calls["llm"].clear()
    calls["search"].clear()
    calls["critic"].clear()
    calls["fix"].clear()


async def run(critic_script, llm_factory=make_llm):
    reset_calls()
    m.llm = llm_factory(critic_script)
    m.search_ddg = fake_search
    m.check_url = fake_check_url
    return await m.generate_report("тестовая сфера")


def case(name, critic_script, expect_verdict, expect_searches=None, expect_reason_sub=None,
         expect_critic=None, expect_fix=None, llm_factory=make_llm):
    res = asyncio.run(run(critic_script, llm_factory))
    ok = res.verdict == expect_verdict
    if expect_searches is not None:
        ok = ok and len(calls["search"]) == expect_searches
    if expect_reason_sub:
        ok = ok and any(expect_reason_sub in r for r in res.reasons)
    for n, want in (expect_critic or {}).items():
        ok = ok and calls["critic"].get(n, 0) == want
    for n, want in (expect_fix or {}).items():
        ok = ok and calls["fix"].get(n, 0) == want
    print(f"{'PASS' if ok else 'FAIL'} | {name}: verdict={res.verdict} searches={len(calls['search'])} "
          f"critic={dict(sorted(calls['critic'].items()))} fix={dict(sorted(calls['fix'].items()))} "
          f"reasons={res.reasons}")
    assert "## 6. Рекомендации по позиционированию" in res.report, "report structure broken"
    return ok, res


OK = '{"verdict":"ok","issues":[],"search_queries":[]}'
FIX_NO_SEARCH = '{"verdict":"fix","issues":["штамп «лидер рынка»"],"search_queries":[]}'
FIX_WITH_SEARCH = '{"verdict":"fix","issues":["цена без источника"],"search_queries":["сервис цены 2026","сервис тарифы"]}'
GARBAGE = "не JSON вовсе"

results = []

# 1. Everything passes on round 1 → no fixes, 12 baseline searches (4 sections x 3 queries), SHIP.
ok1, res1 = case("all sections pass first round", {}, "SHIP", expect_searches=12)
results.append(ok1)

# 2. Section 3 fails once with a search request, then passes → 2 extra queries run, SHIP.
results.append(case("critic orders a re-search, then passes",
                    {3: [FIX_WITH_SEARCH, OK]}, "SHIP", expect_searches=14,
                    expect_critic={3: 2}, expect_fix={3: 1})[0])

# 3. Section 2 never passes → 3 rewrites, each re-checked (4 critiques), then HOLD,
#    and the critic's last complaint is handed to the editor.
ok3, res3 = case("section never passes → hold",
                 {2: [FIX_NO_SEARCH] * 5}, "HOLD", expect_reason_sub="не прошли критика",
                 expect_critic={2: 4}, expect_fix={2: 3})
ok3 = ok3 and "замечания критика к разделу 2" in res3.review and "лидер рынка" in res3.review \
    and res3.checks["unresolved_issues"] == {"2": ["штамп «лидер рынка»"]}
results.append(ok3)

# 4. The last allowed rewrite fixes the section → it is checked and ships, not auto-held.
results.append(case("last rewrite is re-checked and passes",
                    {2: [FIX_NO_SEARCH, FIX_NO_SEARCH, FIX_NO_SEARCH, OK]}, "SHIP",
                    expect_critic={2: 4}, expect_fix={2: 3})[0])

# 5. Unparseable critic answer is treated as a failure, not a silent pass.
results.append(case("garbage critic answer → treated as fix",
                    {5: [GARBAGE, OK]}, "SHIP", expect_fix={5: 1})[0])


# 6. Model gate says hold even when the mechanics are clean.
def gate_hold_llm(critic_script):
    base = make_llm(critic_script)

    async def wrapped(system, user, temperature=0.4, max_tokens=None, retries=2):
        if "Решите судьбу готового отчёта" in user:
            return json.dumps({"verdict": "hold", "reason": "цифры выглядят выдуманными"}, ensure_ascii=False)
        return await base(system, user, temperature, max_tokens, retries)
    return wrapped


results.append(case("editor gate says hold", {}, "HOLD", expect_reason_sub="редактор",
                    llm_factory=gate_hold_llm)[0])

# 7. Footnotes: every section numbers from [^1]; the joined report must not.
defs = re.findall(r"^\[\^(\d+)\]:", res1.report, re.MULTILINE)
refs = re.findall(r"\[\^(\d+)\](?!:)", res1.report)
ok7 = defs == [str(i) for i in range(1, 7)] and sorted(set(refs)) == sorted(set(defs))
print(f"{'PASS' if ok7 else 'FAIL'} | footnotes unique across report: defs={defs}")
results.append(ok7)

unit = m.renumber_footnotes({1: "a[^1] b[^2]\n[^1]: u\n[^2]: v", 2: "c[^1]\n[^1]: w"})
ok7b = unit == {1: "a[^1] b[^2]\n[^1]: u\n[^2]: v", 2: "c[^3]\n[^3]: w"}
print(f"{'PASS' if ok7b else 'FAIL'} | renumber_footnotes maps per section: {unit}")
results.append(ok7b)

unsorted = m.renumber_footnotes({1: "x[^b] y[^a]\n\n[^a]: u\n[^b]: v\n\nконец"})
ok7d = unsorted == {1: "x[^1] y[^2]\n\n[^1]: v\n[^2]: u\n\nконец"}
print(f"{'PASS' if ok7d else 'FAIL'} | footnote definitions listed in order: {unsorted}")
results.append(ok7d)

fdefs = m.footnote_defs("[^1]: https://a.ru/x\n[^2]: Заголовок (https://b.ru/y?q=1).\n[^3]: без ссылки\n")
ok7c = fdefs == [("1", "https://a.ru/x"), ("2", "https://b.ru/y?q=1")]
print(f"{'PASS' if ok7c else 'FAIL'} | footnote_defs reads both url forms: {fdefs}")
results.append(ok7c)

# 8. Review notes stay out of the client report.
ok8 = ("Сводка проверки" not in res1.report and "ВЕРДИКТ" not in res1.report
       and "test-model" not in res1.report and "ВЕРДИКТ" in res1.review
       and res1.checks["audit_ok"] is True and res1.checks["footnotes"] == 6)
print(f"{'PASS' if ok8 else 'FAIL'} | review notes kept out of the client report")
results.append(ok8)

# 8b. Anti-bot answers (403/429/...) are "not verified", not dead links → no HOLD.
async def run_blocked():
    reset_calls()
    m.llm = make_llm({})
    m.search_ddg = fake_search
    m.check_url = lambda url: "BLOCKED HTTP 403"
    return await m.generate_report("тестовая сфера")


resb = asyncio.run(run_blocked())
ok8b = (resb.verdict == "SHIP" and not resb.checks["broken_urls"]
        and len(resb.checks["blocked_urls"]) == 6 and "не пускает ботов" in resb.review)
print(f"{'PASS' if ok8b else 'FAIL'} | bot-blocked links are not dead links: verdict={resb.verdict} "
      f"blocked={len(resb.checks['blocked_urls'])} broken={len(resb.checks['broken_urls'])}")
results.append(ok8b)

# 9. LLM concurrency cap: 10 calls at once never put more than 3 in flight.
async def run_concurrency(cap):
    m.LLM_MAX_CONCURRENCY = cap
    state = {"now": 0, "max": 0}

    async def fake_create(**kw):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        msg = types.SimpleNamespace(content="ok")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    m.client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=fake_create)))
    out = await asyncio.gather(*[REAL_LLM("s", "u") for _ in range(10)])
    return state["max"], out


peak, outs = asyncio.run(run_concurrency(3))
ok9 = peak == 3 and outs == ["ok"] * 10
print(f"{'PASS' if ok9 else 'FAIL'} | LLM concurrency capped: peak={peak} of 10 calls, cap=3")
results.append(ok9)

# 10. Search budget is respected.
os.environ["RESEARCH_EXTRA_SEARCH_BUDGET"] = "1"
import importlib  # noqa: E402
m = importlib.reload(m)
results.append(case("search budget caps extra searches",
                    {2: [FIX_WITH_SEARCH, OK], 3: [FIX_WITH_SEARCH, OK], 4: [FIX_WITH_SEARCH, OK]},
                    "SHIP", expect_searches=14)[0])

print("\n" + ("ALL PASS" if all(results) else "SOME FAILED"))
sys.exit(0 if all(results) else 1)
