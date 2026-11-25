"""Put this family's projects in the pool and register its health checks. Run by the Setup workflow (GitHub Actions: the hub
trusts this repo, s1d-labs/<family>, through its OIDC token). Safe to run again: queue items that exist are kept.

SETUP_PROJECTS (Setup workflow input 'projects'; ocrv5 review 2026-11-25): which projects to register and put in the pool.
  "eci-rolls-hi-ocr,eci-rolls-hi-repair"  only these (the others are not touched: no speed, env or worker limit is reset)
  "all"                                    every project in family.py, and the health checks. This sets threads, env and
                                           max_workers of every project back to family.py: live projects whose values were
                                           raised in the hub (SQL, dashboard) go back to the family.py values.
  unset (local runs, scrape-check)         "all"
  set but empty (the workflow without the input)  refused, exit 2: a full run must be asked for by name.
All entries are checked before the first hub call (the hub refuses max_workers above 256 with HTTP 422, and a refusal in the
middle of the loop left the projects before it already reset)."""
import os
import sys

from scrape_hub_client import register_project, seed_queue, set_pool, set_probes

from family import FAMILY, PROBES, PROJECTS

HUB_MAX_WORKERS, HUB_MAX_THREADS = 256, 32      # scrape-hub app.py ProjectPoolIn (max_workers le=256; threads le=MAX_THREADS)


def chosen(projects, spec):
    if spec is None or spec.strip() == "all":
        return list(projects), True
    names = [x.strip() for x in spec.split(",") if x.strip()]
    if not names:
        raise SystemExit("SETUP_PROJECTS is empty: name the projects (comma list) or 'all' (resets every project to family.py)")
    known = {p["name"] for p in projects}
    bad = [n for n in names if n not in known]
    if bad:
        raise SystemExit(f"not in family.py: {', '.join(bad)} (known: {', '.join(sorted(known))})")
    return [p for p in projects if p["name"] in names], False


def check(projects):
    bad = [f"{p['name']}: max_workers {p['max_workers']} > {HUB_MAX_WORKERS}" for p in projects
           if p.get("max_workers") is not None and not 1 <= p["max_workers"] <= HUB_MAX_WORKERS]
    bad += [f"{p['name']}: threads {p['threads']} > {HUB_MAX_THREADS}" for p in projects
            if p.get("threads") is not None and not 1 <= p["threads"] <= HUB_MAX_THREADS]
    if bad:
        raise SystemExit("the hub would refuse: " + "; ".join(bad))


def main():
    todo, everything = chosen(PROJECTS, os.environ.get("SETUP_PROJECTS"))
    check(todo)
    print("projects:", ", ".join(p["name"] for p in todo))
    if everything:
        print("health checks:", set_probes(FAMILY, PROBES))
    for p in todo:
        register_project(p["name"], title=p["title"], unit_label=p["unit_label"], record_label=p["record_label"],
                         expected_units=len(p["items"]), expected_records=sum(i["payload"].get("expected", 0) for i in p["items"]))
        set_pool(p["name"], FAMILY, p["task"], target=p["target"], env=p["env"], probe=p["probe"], threads=p["threads"],
                 max_workers=p["max_workers"], **{k: p[k] for k in ("proxy", "geo") if p.get(k)})
        print(p["name"], "queue:", seed_queue(p["name"], p["items"]) if p["items"] else "seeded by plan.py")


if __name__ == "__main__":
    sys.exit(main())
