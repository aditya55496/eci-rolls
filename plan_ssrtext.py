"""Plan for task ssrtext: one item per AC of every 2027 SSR draft roll that the CDN projects hold (2026-11-25).

  plan_ssrtext.py RECORDS.tsv OUT.json [--skip-states S24]
RECORDS.tsv: the AC and part records of the projects eci-rolls-hi-cdn and eci-rolls-all-cdn, one per line
"<project>\\t<record_id>\\t<json>", made on the server with
  psql -d scrape_hub -At -F $'\\t' -c "select project, record_id, jsonb_build_object('key', data->'key', 'lang', data->'lang',
    'lang_used', data->'lang_used', 'extra_parts', data->'extra_parts', 'missing', data->'missing', 'state', data->'state')::text
    from records where project in ('eci-rolls-hi-cdn','eci-rolls-all-cdn') and record_id ~ '^[SU][0-9][0-9]-2027-DR/'"
AC record: record_id "<roll id>/<ac>"; part record: "<roll id>/<ac>/<part>" with data.key = the PDF's bucket key. The item keys
are the stored PDFs of the AC (part records), whatever language they are in; lang = the language of those file names
(lang_used of the AC record when they disagree is reported). UP (S24) is left out: its 2027 roll is done (eci-rolls-up-2027).
"""
import collections
import json
import os
import re
import sys

FILE_RE = re.compile(r'-([A-Z]{3})-(\d+)-WI\.pdf$')


def plan(lines, skip_states=("S24",)):
    acs, parts = {}, collections.defaultdict(list)
    for line in lines:
        if not line.strip():
            continue
        project, rid, data = line.rstrip("\n").split("\t", 2)
        d = json.loads(data)
        seg = rid.split("/")
        if len(seg) == 2:
            acs[(seg[0], int(seg[1]))] = (project, d)
        elif len(seg) == 3 and d.get("key"):
            parts[(seg[0], int(seg[1]))].append((int(seg[2]), project, d["key"]))
    items, notes, summary = [], collections.Counter(), collections.Counter()
    for (roll, ac), ps in sorted(parts.items(), key=lambda x: (x[0][0], x[0][1])):
        state = roll.split("-")[0]
        if state in skip_states:
            continue
        projects = {p for _, p, _ in ps}
        if len(projects) > 1:
            notes["ac_in_two_projects"] += 1
        project = sorted(projects)[0]
        keys = sorted({k for _, p, k in ps if p == project}, key=lambda k: int(FILE_RE.search(k).group(2)) if FILE_RE.search(k) else 0)
        langs = collections.Counter(FILE_RE.search(k).group(1) for k in keys if FILE_RE.search(k))
        lang = langs.most_common(1)[0][0]
        if len(langs) > 1:
            notes["ac_with_two_languages"] += 1
        acd = (acs.get((roll, ac)) or (None, {}))[1]
        if acd.get("lang_used") and acd["lang_used"] != lang:
            notes["lang_used_differs"] += 1
        payload = {"state": state, "ac": ac, "lang": lang, "roll": {"id": roll, "year": int(roll.split("-")[1])},
                   "src_project": project, "keys": keys, "expected": 900 * len(keys)}
        if acd.get("extra_parts"):
            payload["extra_parts"] = sorted(int(x) for x in acd["extra_parts"])
        items.append({"key": f"{roll}/{ac}", "payload": payload})
        summary[f"{state} {lang}"] += 1
        summary["pdfs"] += len(keys)
    for i, it in enumerate(items):
        it["partition_id"] = 200000 + i
    return items, dict(summary), dict(notes)


def main():
    src, out = sys.argv[1], sys.argv[2]
    skip = tuple(sys.argv[sys.argv.index("--skip-states") + 1].split(",")) if "--skip-states" in sys.argv else ("S24",)
    with open(src, encoding="utf-8") as fh:
        items, summary, notes = plan(fh, skip)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"items": items, "summary": summary, "notes": notes}, fh, ensure_ascii=False, indent=1)
    print(len(items), "items", json.dumps(summary, ensure_ascii=False), json.dumps(notes))


if __name__ == "__main__":
    main()
