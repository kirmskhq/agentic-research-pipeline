# Example run: local dog-walking service, Moscow

One real, unedited run of the pipeline on an invented brief:

> Локальный сервис выгула собак в Москве: Telegram WebApp, где владелец за пару минут находит
> проверенного выгульщика рядом с домом

- Model: `qwen3.6-35b-a3b-noreason` (NeuralDeep free tier, 3 parallel requests), 4.5 minutes.
- `report.md` is what the client would get. `review.md` and `verdict.json` are the editor's notes.
- Verdict: **HOLD**. All six sections still had critic complaints after three rewrites: numbers
  with no source, and a footnote that does not back the claim it is attached to. The notes list
  each complaint by section, so the editor knows what to check without re-reading everything.
- 31 footnotes, no dead links. One link (Avito) refuses bots, so it is listed as "open by hand"
  rather than as broken.

This example is kept because it is a HOLD. A small model writes fluent text and makes up figures.
The pipeline does not hide that: it tells the editor exactly where it happened.

The only change made after the run: footnote definitions within each section were put in numeric
order with `sort_footnote_defs()`, a step the pipeline now does itself. No text was changed.

---

# Пример прогона: сервис выгула собак, Москва

Один настоящий прогон без правок, бриф придуман. Модель `qwen3.6-35b-a3b-noreason`, 4,5 минуты.
`report.md` получил бы клиент, `review.md` и `verdict.json` — заметки для редактора.

Вердикт — **HOLD**. После трёх правок у всех шести разделов остались замечания критика: цифры
без источника и сноски, которые не подтверждают утверждение. Сводка перечисляет их по разделам.
Битых ссылок нет. Одна ссылка (Avito) не пускает ботов и вынесена в «открыть вручную».

После прогона в каждом разделе только выстроили сноски по номерам, текст не меняли. Теперь
конвейер делает это сам.
