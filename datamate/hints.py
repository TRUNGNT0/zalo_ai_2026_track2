"""Hints computed dynamically per question and business:
- Explicit date anchors and period boundaries
- Grain hints (count days vs orders, price increase D, profit, delivery trips)
- Lot stock calculation rules when shop has lots
- Business wording adjustments ('đh', 'o đờ', 'bữa nay', 'cất hàng', undefined qualities)
- Rulebook matching lines
- Out-of-range checks (§7: unanswerable)
- Dynamic entity value hits from the current business database
"""
from __future__ import annotations

import difflib
import re
from datetime import date, timedelta
from typing import Any, List, Optional, Tuple

from datamate.text import colnorm, norm


def vnorm(x: Any) -> str:
    """Normalize string to ascii snake_case for robust regex matching."""
    return colnorm(str(x))


def ckey(x: Any) -> str:
    """CamelCase and snake_case column key normalization."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(x))
    return colnorm(s)


def quote(x: Any) -> str:
    return '"' + str(x).replace('"', '""') + '"'


def date_anchors(as_of: str) -> str:
    d = date.fromisoformat(str(as_of)[:10])
    mon = d - timedelta(days=d.weekday())
    m1 = d.replace(day=1)
    pm_end = m1 - timedelta(days=1)
    pm1 = pm_end.replace(day=1)
    pm_same = pm1.replace(day=min(d.day, pm_end.day))
    nxt = lambda x: x + timedelta(days=1)
    return "\n".join([
        f"today (hôm nay, bữa nay, hnay) = {d} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d.weekday()]}); half-open end = {nxt(d)}",
        f"the data may go on after today; those later rows have not happened yet (rulebook), so a question naming no period covers the data up to and including today ({d}); a named period keeps its own window; cancel/return status is the order's recorded status",
        f"yesterday (hôm qua, bữa qua, hqua) = {d - timedelta(days=1)} = [{d - timedelta(days=1)}, {d})",
        f"this week (tuần này) = [{mon}, {nxt(d)}); last week (tuần trước, tuần rồi) = [{mon - timedelta(days=7)}, {mon})",
        f"this month (tháng này) = [{m1}, {nxt(d)}); last month same days = [{pm1}, {nxt(pm_same)}); whole last month = [{pm1}, {m1})",
        f"year to date (từ đầu năm) = [{d.replace(month=1, day=1)}, {nxt(d)})",
        *(f"quarters of {y}: " + "; ".join(f"Q{i} (quý {i}) = [{y}-{3*i-2:02d}-01, {y + (i == 4)}-{(3*i+1) % 12 or 12:02d}-01)" for i in range(1, 5))
          for y in (d.year - 1, d.year)),
        f"last 7 days = [{d - timedelta(days=6)}, {nxt(d)}); last 30 days = [{d - timedelta(days=29)}, {nxt(d)}); last N days = [today - (N-1) days, {nxt(d)})",
    ])


def period_hints(question: str, as_of: str, db=None) -> list[str]:
    """Explicit windows for named periods in question."""
    d, t, out = date.fromisoformat(str(as_of)[:10]), vnorm(question), []
    first = lambda y, m: date(y + (m - 1) // 12, (m - 1) % 12 + 1, 1)

    def year_of(m_start: int) -> int:
        return d.year if first(d.year, m_start) <= d else d.year - 1

    for m, y, m2, y2 in re.findall(r"(?:^|_)(?:(?:thang_|thg_?|th_?)(\d{1,2})(?:_nam|_)?_?(\d{4}|\d{2})?|t(\d{1,2})_(\d{4}|\d{2}))(?=_|$)", t):
        m, y = int(m or m2), y or y2
        if 1 <= m <= 12:
            y = (int(y) + (2000 if len(y) == 2 else 0)) if y else year_of(m)
            out.append(f"tháng {m}/{y} = [{first(y, m)}, {first(y, m + 1)})")

    for mt in re.finditer(r"(?<![\d/.-])(\d{1,2}) ?[/-] ?(20\d\d|\d\d)(?![\d/.-])", question):
        if re.search(r"(?i)(?:^|\W)(?:qu[yý]|q|th[aá]ng|thg|th|t)\s*$", question[:mt.start()]):
            continue
        m, y = mt.groups()
        m, yy = int(m), int(y)
        if 1 <= m <= 12 and (len(y) == 4 or d.year - 2005 <= yy <= d.year - 1999):
            y = yy + (2000 if len(y) == 2 else 0)
            out.append(f"tháng {m}/{y} = [{first(y, m)}, {first(y, m + 1)})")

    for qn, y in re.findall(r"(?:^|_)quy_([1-4])(?:_nam|_)?_?(\d{4})?(?=_|$)", t):
        qn = int(qn)
        y = int(y) if y else year_of(3 * qn - 2)
        out.append(f"quý {qn}/{y} = [{first(y, 3 * qn - 2)}, {first(y, 3 * qn + 1)})")

    for y in re.findall(r"(?:^|_)nam_(\d{4})(?=_|$)", re.sub(r"(?:thang|quy)_\d{1,2}_nam_\d{4}", "", t)):
        out.append(f"năm {y} = [{y}-01-01, {int(y) + 1}-01-01)")

    if re.search(r"(?:^|_)tet(?=_|$)", t) and db is not None:
        ys = [int(y) for y in re.findall(r"tet(?:_nam)?_(\d{4})", t)] or [d.year]
        for y in ys:
            try:
                a, b = db.execute("SELECT min(date) FILTER (WHERE lunar_year = ? - 1 AND lunar_month = 12 AND lunar_day = 23 AND NOT lunar_leap), "
                                  "min(date) FILTER (WHERE lunar_year = ? AND lunar_month = 1 AND lunar_day = 10 AND NOT lunar_leap) FROM calendar", [y, y]).fetchone()
            except Exception:
                break
            if a and b:
                n = (b - a).days + 1
                out.append(f"Tết {y} = [{a}, {b + timedelta(days=1)}) ({n} days); trước Tết = [{a - timedelta(days=n)}, {a})")

    for n_days in re.findall(r"(?:^|_)(\d{1,3})_ngay_(?:gan_nhat|gan_day|qua|vua_qua|truoc(?!_(?:do|tet|khi|ngay|hom|dot|lan))|cuoi)", t):
        if len(re.findall(rf"(?:^|_){n_days}_ngay(?=_|$)", t)) > 1:
            continue
        n_days = int(n_days)
        out.append(f"{n_days} ngày gần nhất (incl. today) = [{d - timedelta(days=n_days - 1)}, {d + timedelta(days=1)})")

    days = []
    for dd, mm, yy in re.findall(r"(?:^|_)(\d{1,2})_(\d{1,2})_(\d{4})(?=_|$)", t):
        try:
            x = date(int(yy), int(mm), int(dd))
            out.append(f"ngày {x} = [{x}, {x + timedelta(days=1)})")
            days.append(x)
        except ValueError:
            pass

    k = re.search(r"(?:^|_)(\d{1,3})_ngay_ke_tu(?=_|$)", t)
    if k and len(set(days)) == 1:
        n_val, x = int(k.group(1)), days[0]
        out.append(f"{n_val} ngày kể từ {x} (incl. {x}) = [{x}, {x + timedelta(days=n_val)})")
        if re.search(r"(?:^|_)truoc(?=_|$)", t[k.end():]):
            out.append(f"{n_val} ngày ngay trước {x} = [{x - timedelta(days=n_val)}, {x})")

    # Time processing from ZAIC_2026:
    try:
        from datamate.tools.timeutil import describe_period
        cal_df = None
        if db is not None:
            try:
                cal_df = db.execute("SELECT * FROM calendar").df()
            except Exception:
                pass
        desc = describe_period(question, str(as_of)[:10], cal_df)
        if desc:
            for line in desc.splitlines():
                line_str = line.strip()
                if line_str and line_str not in out:
                    out.append(line_str)
    except Exception:
        pass

    return list(dict.fromkeys(out))


COUNT_DAYS = re.compile(r"(?:^|_)(?:may|bao_nhieu|bn|bnhiu|bao_nheiu|so)_(?:ngay|bua|hom)(?:_|$)")
THRESHOLD = re.compile(r"tro_l|tro_xuong|(?:^|_)(?:tren|duoi|it_nhat|vuot|qua)_\d|≥|>=")
PRICE_UP = re.compile(r"(?:^|_)(?:len_gia|tang_gia|doi_gia|len_\d)")
N_DAYS = re.compile(r"(?:^|_)\d{1,3}_ngay")
DAY_MONTH = re.compile(r"(?:^|_)\d{1,2}_\d{1,2}(?:_\d{4})?(?=_|$)")
PROFIT = re.compile(r"(?:^|_)(?:lai_gop|loi_gop|lai_rong|loi_rong|loi_that|lai_that|loi_nhuan|tru_chi_phi)(?:_|$)")
TRIPS = re.compile(r"(?:^|_)(?:dot|luot|chuyen)_(?:\w+_)?(?:giao|hang)(?:_|$)|giao_(?:\w+_){0,3}(?:may|bao_nhieu|bn|bnhiu|bao_nhiu)_(?:dot|luot|chuyen)(?:_|$)")
LOT = re.compile(r"het_han(?:_|$)|han_dung(?!_(?:hom|bua|ngay|ky))|han_su_dung|(?:^|_)hsd(?:_|$)|(?:^|_)lo_(?:hang|ton|thuoc|gan|sap|con|nao|co)(?:_|$)")


def lot_hint(question: str, meta: dict | None) -> list[str]:
    if not LOT.search(vnorm(question)) or not meta or "tables" not in meta:
        return []
    tabs = []
    for t in meta["tables"]:
        tname = t.get("physical_name", t.get("name", ""))
        cols = [c.get("physical_name", c[0] if isinstance(c, (list, tuple)) else c) for c in t.get("columns", [])]
        if any(re.search(r"(?:^|_)(?:lo|lot|batch|so_lo)$", ckey(c)) for c in cols):
            tabs.append(f"{quote(tname)}({', '.join(c for c in cols if c != '_sheet')})")
    if len(tabs) < 2:
        return []
    return ["LOT STOCK (rulebook: only lots still in stock count): tables keyed by lot: " + "; ".join(tabs) +
            ". A lot's stock at day D = its received base quantity minus SUM(quantity taken - quantity returned) "
            "of rows linking sale lines to that lot, on orders up to D. Compute it per lot in a CTE before joining to lot."]


def grain_hints(question: str) -> list[str]:
    t, out = vnorm(question), []
    if COUNT_DAYS.search(t) and THRESHOLD.search(t + question):
        out.append("GRAIN: count DAYS, not orders: total amount per day (GROUP BY date), then count days whose daily total meets condition.")
    if PRICE_UP.search(t) and N_DAYS.search(t) and not DAY_MONTH.search(t):
        out.append("PRICE INCREASE DATE D is not given: in a CTE take D from price table = first date item's price became new price, "
                   "higher than price right before; then compare quantity sold in [D, D + N days) with [D - N days, D).")
    if PROFIT.search(t):
        out.append("PROFIT (rulebook): gross profit = revenue of valid sale lines - SUM(quantity in SALE unit * cost recorded on sale line), line by line. "
                   "Never multiply cost by base-unit qty. Net profit / lời thật of month = gross profit - SUM(expense rows dated in that month).")
    if TRIPS.search(t):
        out.append("DELIVERY TRIPS (rulebook): each trip (đợt, lượt giao) of each order counts once: COUNT(DISTINCT (order_id, trip_number)).")
    return out


def data_span(meta: dict | None) -> tuple[Optional[str], Optional[str]]:
    if not meta or "tables" not in meta:
        return None, None
    best = (0, None, None)
    for t in meta["tables"]:
        rows = t.get("row_count", t.get("rows", 0))
        tname = t.get("physical_name", t.get("name", ""))
        cols = t.get("columns", [])
        ds = []
        for c in cols:
            if isinstance(c, dict):
                col_name = c.get("physical_name", "")
                typ = c.get("duckdb_type", "")
                span = c.get("date_range")
                if span and (typ in ("DATE", "TIMESTAMP") or re.search(r"ngay|date|thoi_gian", colnorm(col_name))):
                    ds.append((str(span[0])[:10], str(span[1])[:10]))
            elif isinstance(c, (list, tuple)) and len(c) >= 4:
                col_name, typ, _, vals = c[0], c[1], c[2], c[3]
                if typ.startswith(("DATE", "TIMESTAMP")) and vals and vals[0] != "None" and re.search(r"ngay|date|thoi_gian", colnorm(col_name)):
                    ds.append((str(vals[0])[:10], str(vals[1])[:10]))
        if ds and tname != "calendar" and rows > best[0]:
            best = (rows, min(v[0] for v in ds), max(v[1] for v in ds))
    return best[1], best[2]


def outside(periods: list[str], span: tuple[Optional[str], Optional[str]]) -> bool:
    w = [m for p in periods for m in re.findall(r"\[(\d{4}-\d\d-\d\d), (\d{4}-\d\d-\d\d)\)", p)[:1]]
    lo, hi = span
    return bool(w and lo and hi) and all(e <= lo or s > hi for s, e in w)


DEBT_WORDS = re.compile(r"(?:^|_)(?:no|nowj|cong_no|mua_chiu|ghi_no)(?:_|$)")
WORDING = [
    (re.compile(r"(?:^|_)dh(?:_|$)"), "'đh' = đơn hàng (orders): 'bao nhiêu đh' / 'mấy đh' asks a NUMBER OF ORDERS, not money"),
    (re.compile(r"(?:^|_)(?:o|ot)_do(?:_|$)"), "'o đờ' / 'ọt đơ' = order = đơn hàng: 'bao nhiêu o đờ' asks a NUMBER OF ORDERS, not money"),
    (re.compile(r"(?:^|_)(?:bua_nay|bua_ni|hom_ni)(?:_|$)"), "'bữa nay' / 'bữa ni' / 'hôm ni' = TODAY ({d}), not yesterday"),
    (re.compile(r"(?:^|_)cat_hang(?:_|$)"), "'cất hàng' = nhập hàng: buying stock from suppliers (purchase receipts), not stock on hand"),
]
FINAL_VAY = re.compile(r"(?i)(?:^|\s)vay\s*[?.!]*\s*$")
QUALITY = re.compile(r"(?i)thân thiện|uy tín|đáng tin|nhiệt tình|chu đáo|tận tâm|tận tình|lễ phép|vui vẻ|vui tính|dễ thương|dễ mến|khó tính|dễ tính|tay nghề|hài lòng|đẹp trai")
QUALITY_PLAIN = re.compile(r"(?:^|_)(?:than_thien|uy_tin|nhiet_tinh|chu_dao|tan_tam|le_phep|de_thuong|tay_nghe)(?:_|$)")


def wording_hints(question: str, as_of: str, rules: list[tuple[str, str]] = ()) -> list[str]:
    t, out = vnorm(question), []
    for pat, text in WORDING:
        if pat.search(t):
            out.append("WORDING: " + text.format(d=as_of))
    if FINAL_VAY.search(question) and not DEBT_WORDS.search(t):
        out.append("WORDING: the final 'vay' is the particle 'vậy' (no meaning), not 'vay' (a loan)")
    m = QUALITY.search(question.lower()) or QUALITY_PLAIN.search(t)
    if m:
        word = m.group(0).strip("_").replace("_", " ")
        if not any(vnorm(word) in vnorm(name) for name, _ in rules):
            out.append(f"UNDEFINED QUALITY: '{word}' is a personal quality the rulebook does not define and the files do not record: TYPE: unanswerable")
    return out


def rule_rows(rulebook: str) -> list[tuple[str, str]]:
    out = []
    for line in rulebook.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m and not set(m.group(1)) <= set("-: ") and m.group(1) not in ("Luật", "Cách nói", "Cách ghi", "Từ", "type"):
            out.append((m.group(1), m.group(2)))
    return out


RULE_STOP = set("trong ky cua mot cac va theo x la co nhat".split())
RULE_GENERIC = set("nhan vien khach hang mat don tho thang tuan nam ngay quy so".split())


def rule_lines(question: str, rows: list[tuple[str, str]], k: int = 3) -> list[str]:
    words = set(vnorm(question).split("_"))
    hits = []
    for name, text in rows:
        if re.search(r"thứ X\b", name):
            term = [w for w in vnorm(name).split("_") if w and w not in RULE_STOP and w not in ("ngay", "thu")]
            if re.search(r"thu_(hai|ba|tu|nam|sau|bay|[2-7])(_|$)|chu_nhat|(^|_)cn(_|$)", vnorm(question)) and all(w in words for w in term):
                hits.append((set(term), name, text))
            continue
        best = set()
        for part in name.split(","):
            term = {w for w in vnorm(part).split("_") if w and w not in RULE_STOP and not w.isdigit()}
            if len(term) >= 2 and not term <= RULE_GENERIC and term <= words and len(term) > len(best):
                best = term
        if best:
            hits.append((best, name, text))
    hits = [h for h in hits if not any(h[0] < o[0] for o in hits)]
    hits.sort(key=lambda h: -len(h[0]))
    return [f"{name}: {text.replace('**', '')}" for _, name, text in hits[:k]]


def entity_hits(db, meta: dict, question: str, limit: int = 8) -> list[str]:
    """Dynamically search text columns of this shop's DuckDB for exact/high-overlap question entities."""
    if db is None or not meta or "tables" not in meta:
        return []
    q = vnorm(question)
    qw = set(q.split("_"))
    hits = []
    for t in meta["tables"]:
        tname = t.get("physical_name", t.get("name", ""))
        cols = t.get("columns", [])
        for c in cols:
            col_name = c.get("physical_name", c[0] if isinstance(c, (list, tuple)) else c)
            typ = c.get("duckdb_type", c[1] if isinstance(c, (list, tuple)) and len(c) > 1 else "")
            if typ != "VARCHAR" and not str(typ).startswith("VARCHAR"):
                continue
            if re.search(r"(^|_)(id|ma|code|link|url|note|ghi_chu)(_|$)", colnorm(col_name)):
                continue
            try:
                values = [v[0] for v in db.execute(f"SELECT DISTINCT {quote(col_name)} FROM {quote(tname)} WHERE {quote(col_name)} IS NOT NULL LIMIT 500").fetchall()]
                for v in values:
                    nv = vnorm(v)
                    w = set(nv.split("_"))
                    if len(nv) >= 3 and not nv.isdigit() and (("_" + nv + "_") in ("_" + q + "_") or (len(w) >= 2 and len(w & qw) / len(w) >= 0.75)):
                        hits.append((len(nv), tname, col_name, v))
            except Exception:
                continue
    hits.sort(reverse=True)
    out = []
    seen = set()
    for _, t, c, v in hits:
        item = f"{quote(t)}.{quote(c)} = '{v}'"
        if item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def question_hints(question: str, as_of: str, db=None, meta: dict | None = None, rulebook: str = "") -> list[str]:
    """Build all business-specific dynamic hints for one question."""
    rows = rule_rows(rulebook) if rulebook else []
    periods = period_hints(question, as_of, db)
    grains = grain_hints(question)
    lots = lot_hint(question, meta)
    wordings = wording_hints(question, as_of, rows)
    rules = rule_lines(question, rows)
    span = data_span(meta)
    
    out = []
    if outside(periods, span):
        out.append("RULEBOOK §7: Period requested lies completely outside the shop's transaction date range -> TYPE: unanswerable")
    out.extend(periods)
    out.extend(grains)
    out.extend(lots)
    out.extend(wordings)
    if rules:
        out.append("MATCHED RULES: " + " | ".join(rules))
    if db is not None and meta:
        entities = entity_hits(db, meta, question, limit=6)
        if entities:
            out.append("MATCHED DATA VALUES: " + ", ".join(entities))
    return out
