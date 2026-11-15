"""
viet_time.py v2 — bộ giải thời gian tham chiếu cho DataMate (Zalo AI Challenge 2026, Track 2)

Hai nửa độc lập:
  A. resolve(question, as_of, cal=None) : câu hỏi tiếng Việt -> TimeSpec (các khoảng ngày [start, end] GỒM CẢ HAI ĐẦU)
  B. parse_cell_date(value, ...)        : ô dữ liệu bẩn (ISO / dd/mm/yyyy / serial Excel / '1 thg 6 2025' / dd/mm) -> date

Kiến trúc A (v2): LEXER -> NGỮ PHÁP.
  1. normalize(): bỏ dấu, sửa Telex, viết tắt, số chữ -> số.
  2. Lexer: mỗi cụm thời gian được thay bằng token @D3@ (ngày), @M1@ (tháng), @Q0@ (quý), @R2@ (tương đối),
     @S@ (đầu tháng/quý/năm/tuần), @U@ (đến ...), @C@ (cuối ngày ...), @T@ (Tết), @B@ (mô tả kỳ so sánh) ...
  3. Ngữ pháp: ghép token thành khoảng ("từ S đến U"), mốc tồn kho (stock), kỳ so sánh (baseline), chuỗi tháng (series),
     hiệu ứng đổi giá (anchor / main / baseline).
Nguyên tắc: LLM (nếu dùng) chỉ trích *cụm từ* / điền khuyết; MỌI phép cộng trừ ngày làm bằng code ở đây.
Mọi quy ước ngày tháng theo rulebook.md (tuần bắt đầu thứ Hai, "tháng này" = 1→hôm nay, Tết = 23 Chạp → hết mùng 10 Giêng, ...).

calendar.csv: truyền vào `cal` (DataFrame | đường dẫn | LunarCalendar) hoặc để None -> tự tìm
(env DATAMATE_CALENDAR, /data/calendar.csv, ./calendar.csv, cạnh module...). Không tìm thấy -> bảng Tết dự phòng 2024-2027.
"""
from __future__ import annotations
import re, os, sys, csv, glob, unicodedata, calendar
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

# ───────────────────────── helpers ngày ─────────────────────────
def last_day(y, m): return calendar.monthrange(y, m)[1]
def month_span(y, m): return (date(y, m, 1), date(y, m, last_day(y, m)))
def add_months(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1
def clamp_day(y, m, d): return date(y, m, min(d, last_day(y, m)))
def week_monday(d): return d - timedelta(days=d.weekday())          # tuần bắt đầu THỨ HAI
def quarter_span(y, q): return (date(y, 3 * q - 2, 1), month_span(y, 3 * q)[1])
def shift_year(d, k): return clamp_day(d.year + k, d.month, d.day)  # 29/02 -> 28/02 nếu năm đích không nhuận
def shift_month(d, k):
    y, m = add_months(d.year, d.month, k); return clamp_day(y, m, d.day)
def month_end(d): return date(d.year, d.month, last_day(d.year, d.month))
def _mk(y, m, d):
    try: return date(y, m, d)
    except (ValueError, TypeError): return None

# ───────────────────────── chuẩn hoá chữ ─────────────────────────
def strip_acc(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

LEX = set("""nam thang tuan quy hom nay qua roi truoc ngay het tinh den ky cung dau ke tu gan nhat lien bua
tiep nhau ket thuc so voi tang giam nao kia toi hon kem tong cong doanh thu mau bao nhieu may ngoai giua nua
sang chieu cuoi""".split())

def _telex_base(t):
    if len(t) > 2 and t[-1] in "sfrxjz": t = t[:-1]                  # phím dấu thanh ở cuối âm tiết
    t = t.replace("dd", "d")
    for a, b in (("aa", "a"), ("ee", "e"), ("oo", "o"), ("aw", "a"), ("ow", "o"), ("uw", "u")): t = t.replace(a, b)
    return t

def fix_telex_tokens(s):
    """Chỉ sửa token ASCII mà giải mã Telex RA MỘT TỪ KHOÁ THỜI GIAN; token khác (tên hàng) giữ nguyên."""
    def f(m):
        w = m.group(0)
        if w in LEX: return w
        b = _telex_base(w)
        return b if b in LEX else w
    return re.sub(r"[a-z]+", f, s)

ABBR = [(r"\bhnay\b", "hom nay"), (r"\bhqua\b", "hom qua"), (r"\bbua nay\b", "hom nay"), (r"\bbua qua\b", "hom qua"),
        (r"\bth (truoc|trc|roi|nay)\b", r"thang \1"), (r"\bthg\b", "thang"), (r"\bth ?(\d{1,2})(?=\s*(?:/|nam|\b))", r"thang \1"),
        (r"\btrc\b", "truoc"), (r"\bhom bua\b", "hom qua"), (r"\bki\b", "ky"), (r"\bqui\b", "quy"),
        (r"\bngay do\b", "ngay do"), (r"\bhn\b", "hom nay")]

_NUMW = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
         "muoi mot": 11, "muoi hai": 12, "muoi lam": 15, "hai muoi": 20, "ba muoi": 30}

def _words_to_digits(s):
    """s: chữ thường còn dấu. 'tháng tư'->'tháng 4', 'quý ba'->'quý 3', 'hai tuần gần nhất'->'2 tuần gần nhất'."""
    def mth(m):
        w = strip_acc(m.group(2))
        return f"{m.group(1)} {_NUMW[w]}"
    s = re.sub(r"(tháng) (một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười một|mười hai|mười)(?![\wà-ỹ])(?!\s*(?:ngoái|nay|trước|qua|rồi|sau|tới)(?![\wà-ỹ]))", mth, s)
    s = re.sub(r"(quý|quí) (một|hai|ba|bốn|tư)(?![\wà-ỹ])", lambda m: "quý " + str(_NUMW[strip_acc(m.group(2))]), s)
    s = re.sub(r"(?<![\wà-ỹ])(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười lăm|mười|hai mươi|ba mươi) (ngày|tuần|tháng) "
              r"(?=gần nhất|vừa qua|vừa rồi|gần đây|qua|tới|nữa|sắp tới|kể từ|đầu|cuối|liền trước|trước đó)",
              lambda m: f"{_NUMW[strip_acc(m.group(1))]} {m.group(2)} ", s)
    return s

def normalize(q):
    s = unicodedata.normalize("NFC", q.lower())
    s = _words_to_digits(s)
    s = re.sub(r"(?<![\wà-ỹ])tét(?![\wà-ỹ])", "tetx", s)             # 'bánh tét' không phải Tết
    s = strip_acc(s)
    s = fix_telex_tokens(s)
    for p, r in ABBR: s = re.sub(p, r, s)
    return re.sub(r"\s+", " ", s).strip()

def month_word(tok_accented):
    """Giữ lại cho tương thích: 'tháng tư' -> 4 (chỉ dùng trên bản CÒN DẤU)."""
    m = re.search(r"tháng (một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười một|mười hai|mười)(?![\wà-ỹ])(?!\s*(?:ngoái|nay|trước|qua|rồi)\b)", tok_accented)
    return _NUMW.get(strip_acc(m.group(1))) if m else None

# ───────────────────────── cấu trúc kết quả ─────────────────────────
@dataclass
class Period:
    start: date
    end: date                         # gồm cả end
    label: str = ""
    role: str = "main"                # main | baseline | anchor
    kind: str = ""                    # day|week|month|quarter|year|half|range|rolling|forward|tet|stock
    target: str = ""                  # 'expiry' nếu kỳ áp lên HẠN DÙNG (không phải ngày đặt đơn)
    todate: bool = False              # kỳ "đến hôm nay" (tháng này, tuần này, từ đầu năm...)
    def days(self): return (self.end - self.start).days + 1
    def __iter__(self): return iter((self.start, self.end))

@dataclass
class TimeSpec:
    periods: list = field(default_factory=list)
    stock: bool = False               # "đến hết ngày X / cuối ngày / tính đến hôm nay": trạng thái còn lại, cộng dồn TỪ ĐẦU DỮ LIỆU
    stock_date: Optional[date] = None
    shift: Optional[str] = None       # 'sang' (06:00–13:59:59) | 'chieu' (14:00–21:59:59) — ca của kỳ main
    baseline_shift: Optional[str] = None   # "ca chiều so với ca sáng": mốc = ca sáng
    shifts: list = field(default_factory=list)
    needs_data: list = field(default_factory=list)   # 'price_raise_date': phải tra bảng giá
    price_effect: Optional[dict] = None              # {'n_after':7,'n_before':7,'anchor':date|None}
    flags: list = field(default_factory=list)        # mơ hồ / suy luận -> nên hạ độ tin cậy
    series: bool = False              # nhiều tháng/quý, trả list theo thứ tự thời gian
    time_cue: bool = False            # câu có chữ chỉ thời gian (dù giải được hay không)
    def main(self): return [p for p in self.periods if p.role == "main"]
    def baseline(self): return next((p for p in self.periods if p.role == "baseline"), None)
    def anchor(self): return next((p for p in self.periods if p.role == "anchor"), None)
    def overall(self):
        m = self.main()
        return (min(p.start for p in m), max(p.end for p in m)) if m else None
    @property
    def status(self):
        if self.periods or self.stock: return "resolved"
        if self.needs_data: return "needs_data"
        return "unresolved" if self.time_cue else "no_time_expression"
    def to_dict(self):
        iso = lambda d: d.isoformat() if d else None
        return {"status": self.status, "stock": self.stock, "stock_date": iso(self.stock_date), "shift": self.shift,
                "baseline_shift": self.baseline_shift, "series": self.series, "needs_data": self.needs_data,
                "price_effect": ({**self.price_effect, "anchor": iso(self.price_effect.get("anchor"))} if self.price_effect else None),
                "flags": self.flags,
                "periods": [{"start": iso(p.start), "end": iso(p.end), "role": p.role, "kind": p.kind, "target": p.target,
                             "label": p.label, "days": p.days()} for p in self.periods]}

# ───────────────────────── lịch âm / Tết ─────────────────────────
_TET_FALLBACK = {2024: (date(2024, 2, 2), date(2024, 2, 19)), 2025: (date(2025, 1, 22), date(2025, 2, 7)),
                 2026: (date(2026, 2, 10), date(2026, 2, 26)), 2027: (date(2027, 1, 30), date(2027, 2, 15))}
_CAN_CHI = {"giap thin": 2024, "at ty": 2025, "binh ngo": 2026, "dinh mui": 2027}

class LunarCalendar:
    """Bọc calendar.csv: date, weekday, lunar_day, lunar_month, lunar_year, lunar_leap, holiday."""
    def __init__(self, rows):
        self.by_date, self.by_lunar, self.holidays = {}, {}, {}
        for d, ld, lm, ly, leap, hol in rows:
            self.by_date[d] = (ld, lm, ly, leap)
            self.by_lunar[(ly, lm, ld, leap)] = d
            if hol: self.holidays.setdefault(hol, []).append(d)
    @staticmethod
    def _b(x): return str(x).strip().lower() in ("true", "1", "1.0", "yes")
    @classmethod
    def from_csv(cls, path):
        rows = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                d = date.fromisoformat(r["date"][:10])
                hol = (r.get("holiday") or "").strip()
                rows.append((d, int(r["lunar_day"]), int(r["lunar_month"]), int(r["lunar_year"]), cls._b(r["lunar_leap"]), hol))
        return cls(rows)
    @classmethod
    def from_df(cls, df):
        rows = []
        for r in df.itertuples(index=False):
            d = r.date; d = d.date() if hasattr(d, "date") else d
            hol = getattr(r, "holiday", None); hol = "" if hol is None or hol != hol else str(hol).strip()
            rows.append((d, int(r.lunar_day), int(r.lunar_month), int(r.lunar_year), cls._b(r.lunar_leap), hol))
        return cls(rows)
    def lunar(self, ly, lm, ld, leap=False): return self.by_lunar.get((ly, lm, ld, leap))
    def lunar_month_span(self, ly, lm, leap=False):
        ds = sorted(d for (y, m, _, lp), d in self.by_lunar.items() if (y, m, lp) == (ly, lm, leap))
        return (ds[0], ds[-1]) if ds else None
    def holiday(self, name, year):
        return next((d for d in sorted(self.holidays.get(name, [])) if d.year == year), None)

_CAL_CACHE = {}
def _candidate_calendar_paths():
    env = os.environ.get("DATAMATE_CALENDAR")
    c = [env] if env else []
    here = os.path.dirname(os.path.abspath(__file__))
    c += ["/data/calendar.csv", "calendar.csv", os.path.join(here, "calendar.csv"), os.path.join(here, "..", "calendar.csv"),
          os.path.join(os.getcwd(), "train", "calendar.csv")]
    c += glob.glob("/data/*/calendar.csv")
    return c

def load_calendar(cal=None):
    """cal: None | LunarCalendar | pandas.DataFrame | đường dẫn csv  ->  LunarCalendar hoặc None."""
    if isinstance(cal, LunarCalendar): return cal
    if cal is not None and hasattr(cal, "itertuples"):
        k = id(cal)
        if k not in _CAL_CACHE: _CAL_CACHE[k] = LunarCalendar.from_df(cal)
        return _CAL_CACHE[k]
    paths = [cal] if isinstance(cal, (str, os.PathLike)) else _candidate_calendar_paths()
    for p in paths:
        try:
            if p and os.path.isfile(p):
                if p not in _CAL_CACHE: _CAL_CACHE[p] = LunarCalendar.from_csv(p)
                return _CAL_CACHE[p]
        except Exception: continue
    return None

def tet_range(cal, Y):
    """23 tháng Chạp (năm âm Y-1) -> HẾT mùng 10 tháng Giêng (năm âm Y). Bỏ tháng nhuận. Không có lịch -> bảng dự phòng."""
    lc = load_calendar(cal)
    if lc:
        a, b = lc.lunar(Y - 1, 12, 23), lc.lunar(Y, 1, 10)
        if a and b: return a, b
    return _TET_FALLBACK.get(Y)

def before_tet(cal, Y, n=None):
    t = tet_range(cal, Y)
    if not t: return None
    n = n or (t[1] - t[0]).days + 1
    return t[0] - timedelta(days=n), t[0] - timedelta(days=1)

def after_tet(cal, Y, n=None):
    t = tet_range(cal, Y)
    if not t: return None
    n = n or (t[1] - t[0]).days + 1
    return t[1] + timedelta(days=1), t[1] + timedelta(days=n)

def price_effect_periods(d, n_after, n_before=None):
    """rulebook §5: N ngày kể từ D (tính cả D)  vs  N ngày ngay trước D."""
    n_before = n_before or n_after
    return (d, d + timedelta(n_after - 1)), (d - timedelta(n_before), d - timedelta(1))

# ───────────────────────── phân loại / dịch kỳ (kỳ so sánh) ─────────────────────────
def _classify(p, a):
    s, e = p.start, p.end
    if p.kind in ("day", "stock"): return "day"
    if s.day == 1 and s.year == e.year and s.month == e.month:
        return "month" if e == month_end(e) else "mtd"
    if s.day == 1 and s.month in (1, 4, 7, 10) and (e.year, (e.month - 1) // 3) == (s.year, (s.month - 1) // 3):
        return "quarter" if e == month_end(e) and e.month == s.month + 2 else "qtd"
    if (s.month, s.day) == (1, 1) and s.year == e.year:
        return "year" if (e.month, e.day) == (12, 31) else "ytd"
    if s.weekday() == 0 and e - s <= timedelta(6) and p.kind == "week":
        return "week" if e.weekday() == 6 else "wtd"
    return "range"

def _shift_months(s, e, k):
    if s.day == 1 and e == month_end(e):
        y1, m1 = add_months(s.year, s.month, k); y2, m2 = add_months(e.year, e.month, k)
        return date(y1, m1, 1), month_end(date(y2, m2, 1))
    return shift_month(s, k), shift_month(e, k)

def derive_baseline(p, mode, a, flags=None):
    """Kỳ so sánh của p theo mode: prev (kỳ liền trước) | yoy (cùng kỳ năm trước) | prev_month | prev_week | prev_quarter."""
    flags = flags if flags is not None else []
    c = _classify(p, a); s, e = p.start, p.end
    if mode == "yoy":
        if c in ("month", "quarter", "year", "half"): ns, ne = _shift_months(s, e, -12)
        else: ns, ne = shift_year(s, -1), shift_year(e, -1)
        return Period(ns, ne, "cùng kỳ năm trước", "baseline", p.kind, p.target)
    if mode == "prev_month":
        ns, ne = _shift_months(s, e, -1)
        return Period(ns, ne, "cùng kỳ tháng trước (cùng số ngày, cắt ở cuối tháng)" if c in ("mtd", "range", "day") else "tháng liền trước", "baseline", p.kind)
    if mode == "prev_quarter":
        ns, ne = _shift_months(s, e, -3); return Period(ns, ne, "quý trước", "baseline", p.kind)
    if mode == "prev_week":
        return Period(s - timedelta(7), e - timedelta(7), "tuần trước (cùng số ngày)", "baseline", p.kind)
    # prev = kỳ liền trước cùng loại
    if p.kind == "tet":
        n = p.days(); return Period(s - timedelta(n), s - timedelta(1), "ngay trước Tết (cùng số ngày)", "baseline", "range")
    if c in ("month", "mtd"):
        ns, ne = _shift_months(s, e, -1)
        if c == "mtd": ne = min(clamp_day(ns.year, ns.month, e.day), month_end(ns))   # rulebook §2: cùng số ngày
        return Period(ns, ne, "tháng liền trước" if c == "month" else "cùng kỳ tháng trước (cùng số ngày, cắt ở cuối tháng)", "baseline", p.kind)
    if c in ("quarter", "qtd"):
        ns, ne = _shift_months(s, e, -3)
        if c == "qtd": ne = ns + (e - s); flags.append("baseline_by_analogy")
        return Period(ns, ne, "quý trước", "baseline", p.kind)
    if c in ("year", "ytd"):
        ns, ne = shift_year(s, -1), (date(s.year - 1, 12, 31) if c == "year" else shift_year(e, -1))
        if c == "ytd": flags.append("baseline_by_analogy")
        return Period(ns, ne, "năm trước", "baseline", p.kind)
    if c in ("week", "wtd"):
        if c == "wtd": flags.append("baseline_by_analogy")
        return Period(s - timedelta(7), e - timedelta(7), "tuần trước (cùng số ngày)", "baseline", p.kind)
    if c == "day":
        flags.append("baseline_by_analogy"); return Period(s - timedelta(1), s - timedelta(1), "ngày liền trước", "baseline", "day")
    n = p.days(); return Period(s - timedelta(n), s - timedelta(1), f"{n} ngày liền trước", "baseline", "range")

# ───────────────────────── A. giải câu hỏi ─────────────────────────
_WD = {"hai": 0, "ba": 1, "tu": 2, "nam": 3, "sau": 4, "bay": 5, "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5}
_TOK = r"@[A-Z]\d+@"
_STATE = r"tong no|so no|khach no|ai no|\bno bao nhieu|con ton|ton kho|ton cuoi|con no|dang no|con lai (?!ban)|trong kho|het hang|con (?:may|bao nhieu|mau)|hien con|con hang|ton lai|so du"
_TIMECUE = re.compile(r"\b(thang|tuan|quy (?:\d|nay|truoc|roi)|hom (?:nay|qua|kia)|ngay (?:\d|hom|mai|do)|nam (?:nay|ngoai|truoc|20\d\d)|tet|\d+ ngay|ca sang|ca chieu|cung ky|"
                      r"dau nam|dau thang|cuoi thang|cuoi nam|ke tu|tinh den|cuoi ngay|chu nhat|thu (?:hai|ba|tu|nam|sau|bay)|20\d\d|q[1-4])\b|\d+/\d+")

def _to_date(a):
    if isinstance(a, datetime): return a.date()
    if isinstance(a, date): return a
    return date.fromisoformat(str(a)[:10])

class _Resolver:
    def __init__(self, q, a, cal):
        self.q, self.a, self.cal = q, a, load_calendar(cal)
        self.raw = unicodedata.normalize("NFC", q.lower())
        self.s = normalize(q)
        self.toks = []
        self.sp = TimeSpec()
        self.flags = self.sp.flags
        self.used = set()

    # ---- cơ chế token ----
    def new(self, kind, **kw):
        i = len(self.toks); self.toks.append(dict(kind=kind, **kw)); return f" @{kind}{i}@ "
    def sub(self, rx, fn):
        def rep(m):
            r = fn(m); return m.group(0) if r is None else r
        self.s = re.sub(rx, rep, self.s); self.s = re.sub(r"\s+", " ", self.s)
    def tk(self, mk):  # "@D3@" -> dict
        return self.toks[int(re.search(r"\d+", mk).group())]
    def span_tok(self, kind, s, e, label, tkind, **kw):
        return self.new(kind, p=Period(s, e, label, "main", tkind, **{k: v for k, v in kw.items() if k in ("target", "todate")}),
                        **{k: v for k, v in kw.items() if k not in ("target", "todate")})
    def year_of(self, m_year, m_rel=None):
        a = self.a
        if m_year: return int(m_year)
        if m_rel in ("nay",): return a.year
        if m_rel in ("ngoai", "truoc", "qua"): return a.year - 1
        return None

    # ---- 1. lexer ----
    def lex(self):
        a = self.a
        # ca / buổi
        for m, nm in ((r"\bca sang\b", "sang"), (r"\bca chieu\b", "chieu")):
            for mm in re.finditer(m, self.s):
                self.sp.shifts.append((mm.start(), nm))
        self.sp.shifts.sort(); order = [n for _, n in self.sp.shifts]; self.sp.shifts = []
        for n in order:
            if n not in self.sp.shifts: self.sp.shifts.append(n)
        # phần trong ngày: "sáng nay", "chiều qua"
        def pod(m):
            nm = "sang" if m.group(1) == "sang" else "chieu"; self.sp.shifts.append(nm) if nm not in self.sp.shifts else None
            self.flags.append("part_of_day_as_shift")
            d = a if m.group(2) == "nay" else a - timedelta(1)
            return self.span_tok("R", d, d, "hôm nay" if d == a else "hôm qua", "day", rel=True)
        self.sub(r"\b(sang|chieu) (nay|qua|hom qua)\b(?! nay)", pod)

        # --- lịch âm / lễ (trước ngày dương) ---
        self.lex_lunar()
        # --- ngày dương ---
        self.lex_dates()
        # --- chuỗi tháng đặc biệt ---
        def ser_end(m):
            n, M, Y = int(m[1]), int(m[2]), int(m[3]); ps = []
            for k in range(-(n - 1), 1):
                y, mo = add_months(Y, M, k); ps.append(Period(*month_span(y, mo), f"{mo}/{y}", "main", "month"))
            return self.new("L", ps=ps)
        self.sub(r"(\d+) thang lien (?:nhau|tiep) ket thuc (?:o|tai|vao) thang (\d{1,2}) ?(?:/|nam) ?(20\d\d)", ser_end)
        def ser_start(m):
            n, M, Y = int(m[1]), int(m[2]), int(m[3]); ps = []
            for k in range(n):
                y, mo = add_months(Y, M, k); ps.append(Period(*month_span(y, mo), f"{mo}/{y}", "main", "month"))
            return self.new("L", ps=ps)
        self.sub(r"(\d+) thang lien (?:nhau|tiep) (?:tu|bat dau tu) thang (\d{1,2}) ?(?:/|nam) ?(20\d\d)(?! ?(?:den|toi|qua|-))", ser_start)

        # --- Tết & trước/sau Tết (trước 'nam 20xx') ---
        self.lex_tet()
        # --- quý ---
        def q_my(m):
            q = m[1] or m[3]; q = {"i": 1, "ii": 2, "iii": 3, "iv": 4}.get(q) or int(q); Y = int(m[2] or m[4])
            if not 1 <= q <= 4: return None
            return self.span_tok("Q", *quarter_span(Y, q), f"Q{q}/{Y}", "quarter")
        self.sub(r"\bquy (\d|i{1,3}|iv)\s*(?:/|nam|-|\.)?\s*(20\d\d)(?!\d)|\bq([1-4])\s*[/ -]?\s*(20\d\d)(?!\d)", q_my)
        def q_rel(m):
            q = int(m[1]); Y = self.year_of(None, m[2])
            return self.span_tok("Q", *quarter_span(Y, q), f"Q{q}/{Y}", "quarter") if 1 <= q <= 4 else None
        self.sub(r"\bquy (\d) nam (nay|ngoai|truoc|qua)\b", q_rel)
        def q_noy(m):
            q = int(m[1])
            if not 1 <= q <= 4: return None
            Y = a.year if date(a.year, 3 * q - 2, 1) <= a else a.year - 1
            self.flags.append("quarter_no_year")
            return self.span_tok("Q", *quarter_span(Y, q), f"Q{q}/{Y} (không ghi năm)", "quarter")
        self.sub(r"\bquy (\d)(?![\d/])(?! ?(?:/|nam|-) ?\d)", q_noy)
        # --- tháng ---
        self.lex_months()
        # --- nửa năm ---
        def half(m):
            Y = self.year_of(m[2], m[3]) if (m[2] or m[3]) else a.year
            first = m[1] in ("nua dau", "6 thang dau", "sau thang dau")
            s, e = (date(Y, 1, 1), date(Y, 6, 30)) if first else (date(Y, 7, 1), date(Y, 12, 31))
            return self.span_tok("H", s, e, f"{'nửa đầu' if first else 'nửa cuối'} năm {Y}", "half")
        self.sub(r"\b(nua dau|nua cuoi|6 thang dau|6 thang cuoi|sau thang dau|sau thang cuoi) nam(?: (20\d\d)| (nay|ngoai|truoc))?", half)
        # --- năm ---
        self.sub(r"\bnam (20\d\d)(?!\d)", lambda m: self.span_tok("Y", date(int(m[1]), 1, 1), date(int(m[1]), 12, 31), f"năm {m[1]}", "year"))

        # --- N ngày kể từ / trước (hiệu ứng đổi giá) ---
        self.lex_price()

        # --- mô tả kỳ so sánh ---
        self.lex_baseline_desc()
        # --- cửa sổ trượt ---
        def roll(m):
            n, u = int(m[1]), m[2]
            if u == "ngay": s = a - timedelta(n - 1); lab = f"{n} ngày gần nhất (gồm hôm nay)"
            elif u == "tuan": s = a - timedelta(7 * n - 1); lab = f"{n} tuần gần nhất (=7n ngày)"; self.flags.append("assume_7n_days")
            else:
                y, mo = add_months(a.year, a.month, -n); s = clamp_day(y, mo, a.day) + timedelta(1); lab = f"{n} tháng gần nhất"; self.flags.append("rolling_months_ambiguous")
            if m[3] in ("qua", "vua qua"): self.flags.append("'N ngày qua' coi như 'gần nhất'")
            return self.span_tok("K", s, a, lab, "rolling", todate=True)
        self.sub(r"\b(\d+) (ngay|tuan|thang) (gan nhat|vua qua|vua roi|gan day|qua|tro lai day|vua ket thuc)\b", roll)
        def fwd(m):
            n, u = int(m[1]), m[2]
            e = a + timedelta(n) if u == "ngay" else (a + timedelta(7 * n) if u == "tuan" else add_months(a.year, a.month, n) and clamp_day(*add_months(a.year, a.month, n), a.day))
            return self.span_tok("F", a, e, f"{n} {u} tới = [D, D+N]", "forward")
        self.sub(r"\b(\d+) (ngay|tuan|thang) (?:toi|nua|sap toi|tiep theo|sap den|ke tiep)\b", fwd)
        # --- thứ trong tuần ---
        def wday(m):
            idx = 6 if m[2] else _WD[m[1]]; w = m[3]
            if w in ("tuan nay", "nay"): base = week_monday(a); d = base + timedelta(idx)
            elif w.startswith("tuan"): base = week_monday(a) - timedelta(7); d = base + timedelta(idx)
            else:
                d = a - timedelta(((a.weekday() - idx) % 7) or 7); self.flags.append("weekday_last_ambiguous")
            return self.span_tok("D", d, d, f"{m[0]}", "day")
        self.sub(r"(?<!doanh )(?<!tong )(?<!tien )(?<!khoan )(?<!nhap )\b(?:thu (hai|ba|tu|nam|sau|bay|[2-7])|(chu nhat|cn)) (tuan nay|nay|tuan (?:roi|truoc|vua roi|qua)|vua roi|vua qua|gan nhat)\b",
                 lambda m: None if (m[1] == "nam" and m[3] == "nay") or (m[1] == "tu" and m[3] == "nay") else wday(m))
        # --- từ đầu tháng/quý/năm/tuần ---
        def start_of(m):
            u = m[1]
            s = {"thang": a.replace(day=1), "nam": date(a.year, 1, 1), "tuan": week_monday(a), "quy": quarter_span(a.year, (a.month - 1) // 3 + 1)[0]}[u]
            return self.new("S", s=s, unit=u)
        self.sub(r"\bdau (thang|quy|nam|tuan)(?: nay)?(?= |$)(?! \d)", lambda m: start_of(m) if re.search(r"\b(?:tu|tinh tu|ke tu)\s*$", self.s[:m.start()]) or re.match(r"\s*(?:den|toi|cho den)\b", self.s[m.end():]) else None)
        # --- 'từ nay' ---
        self.sub(r"\b(?:ke )?tu (?:hom )?nay\b(?= )", lambda m: self.new("N"))
        # --- tương đối ---
        self.lex_relative()
        # --- 'đến ...' ---
        def now_end(m):
            return self.new("U", inner=self.new_today_idx())
        self.sub(r"\b(?:tinh )?(?:den|toi|cho den|cho toi)(?: het| cuoi| tron)?(?: ngay)? (?:nay|gio|bay gio|hien tai)\b", now_end)
        def until(m):
            return self.new("U", inner=int(re.search(r"\d+", m[1]).group()))
        self.sub(rf"\b(?:tinh )?(?:den|toi|cho den|cho toi)(?: het| cuoi| tron)?(?: ngay)? ({_TOK})", until)
        # --- 'cuối ngày ...' ---
        def cutoff(m): return self.new("C", inner=int(re.search(r"\d+", m[1]).group()))
        self.sub(rf"\b(?:vao |luc )?(?:cuoi|het|chot) (?:ngay )?({_TOK})", lambda m: cutoff(m) if self.tk(m[1])["kind"] in "DR" and self.tk(m[1])["p"].kind == "day" else None)
        self.sub(r"\b(?:luc |vao )?(?:cuoi ngay|het ngay)\b(?! @)", lambda m: self.new("C", inner=None))

    def new_today_idx(self):
        i = len(self.toks); self.toks.append(dict(kind="R", p=Period(self.a, self.a, "hôm nay", "main", "day", todate=False), rel=True)); return i

    def lex_dates(self):
        a = self.a
        def day(d, label=None, noyr=False):
            if noyr: self.flags.append("day_month_no_year")
            return self.span_tok("D", d, d, label or f"ngày {d}", "day")
        def iso(m):
            d = _mk(int(m[1]), int(m[2]), int(m[3])); return day(d) if d else None
        self.sub(r"(?<![\d-])(20\d\d)-(\d{2})-(\d{2})(?![\d-])", iso)
        def full(m):
            d = _mk(int(m[3]), int(m[2]), int(m[1])); return day(d) if d else None
        self.sub(r"(?<![\d/.,-])(\d{1,2}) ?[/.\-] ?(\d{1,2}) ?[/.\-] ?(20\d\d)(?![\d/])", full)
        def word(m):
            dd, mm = int(m[1]), int(m[2]); y = m[3] or None
            if y: d = _mk(int(y), mm, dd); return day(d) if d else None
            if m[4]: d = _mk(self.year_of(None, m[4]), mm, dd); return day(d) if d else None
            y = a.year if (_mk(a.year, mm, dd) or date.max) <= a else a.year - 1
            d = _mk(y, mm, dd); return day(d, noyr=True) if d else None
        self.sub(r"\bngay (\d{1,2}) thang (\d{1,2})(?: nam (20\d\d)| nam (nay|ngoai|truoc))?(?! ?/ ?\d)", lambda m: word(m) if not re.match(r"\d", self.s[m.end():m.end() + 1] or "x") else None)
        def dm(m):
            dd, mm = int(m[1]), int(m[2])
            pre = self.s[max(0, m.start() - 14):m.start()]
            ok = re.search(r"(?:ngay|tu|den|toi|het|vao|hom|truoc|sau|dip|cuoi|dau)\s*$", pre) or (dd, mm) in ((30, 4), (1, 5), (2, 9), (1, 1))
            if not ok or not (1 <= mm <= 12 and 1 <= dd <= 31): return None
            if m[3]: y = self.year_of(None, m[3]); d = _mk(y, mm, dd); return day(d) if d else None
            y = a.year if (_mk(a.year, mm, dd) or date.max) <= a else a.year - 1
            d = _mk(y, mm, dd); return day(d, noyr=True) if d else None
        self.sub(r"(?<![\d/.,-])(\d{1,2}) ?/ ?(\d{1,2})(?![\d/])(?: nam (nay|ngoai|truoc))?", dm)

    def lex_lunar(self):
        a, cal = self.a, self.cal
        MW = r"(\d{1,2}|mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi)"
        def lm(x):
            x = x.strip()
            return {"gieng": 1, "chap": 12}.get(x) or (int(x) if x.isdigit() else None)
        def lunar_date(ly, lmn, ld, label):
            if not cal: self.flags.append("need_calendar_for_lunar"); return None
            d = cal.lunar(ly, lmn, ld)
            return self.span_tok("D", d, d, label, "day") if d else None
        def infer_ly(lmn, ld):
            if not cal: return a.year
            d = cal.lunar(a.year, lmn, ld); return a.year if (d and d <= a) else a.year - 1
        def ld_m(m):
            dd, mm = int(m[1]), int(m[2]); ly = int(m[3]) if m[3] else infer_ly(mm, dd)
            self.flags.append("lunar_expression: tra calendar.csv (bỏ tháng nhuận trừ khi nói rõ)")
            return lunar_date(ly, mm, dd, f"{dd}/{mm} âm lịch")
        self.sub(r"(?<![\d/.,-])(\d{1,2}) ?/ ?(\d{1,2})(?: ?/ ?(20\d\d))? (?:am lich|al)\b", ld_m)
        def mung(m):
            w = m[1]; dd = _NUMW.get(w) if not w.isdigit() else int(w)
            mm = lm(m[2]) if m[2] else 1
            ly = self.year_of(m[3], m[4]) if (m[3] or m[4]) else infer_ly(mm, dd)
            self.flags.append("lunar_expression: tra calendar.csv (bỏ tháng nhuận trừ khi nói rõ)")
            return lunar_date(ly, mm, dd, f"mùng {dd} tháng {mm} âm")
        self.sub(rf"\bmung {MW}(?: tet| thang ({MW[1:-1]}|gieng|chap))?(?: am lich)?(?: nam (20\d\d)| nam (nay|ngoai|truoc))?", lambda m: mung(m) if m[1] else None)
        def ram(m):
            mm = lm(m[1]) if m[1] else 1
            ly = self.year_of(m[2], m[3]) if (m[2] or m[3]) else infer_ly(mm, 15)
            self.flags.append("lunar_expression: tra calendar.csv (bỏ tháng nhuận trừ khi nói rõ)")
            return lunar_date(ly, mm, 15, f"rằm tháng {mm} âm")
        self.sub(r"\bram(?: thang (\d{1,2}|gieng|chap))?(?: am lich)?(?: nam (20\d\d)| nam (nay|ngoai|truoc))?\b", ram)
        def lmonth(m):
            if not cal: self.flags.append("need_calendar_for_lunar"); return None
            mm = lm(m[1].split()[0])
            if m[2] or m[3]: ly = self.year_of(m[2], m[3])
            else:
                f = cal.lunar(a.year, mm, 1); ly = a.year if (f and f <= a) else a.year - 1
            sp = cal.lunar_month_span(ly, mm)
            self.flags.append("lunar_expression: tra calendar.csv (bỏ tháng nhuận trừ khi nói rõ)")
            return self.span_tok("M", sp[0], sp[1], f"tháng {mm} âm lịch {ly}", "month") if sp else None
        self.sub(r"\bthang (gieng|chap|\d{1,2} am lich)(?: am lich)?(?: nam (20\d\d)| nam (nay|ngoai|truoc))?", lmonth)
        # 30 Tết / giao thừa
        def giaothua(m):
            if not cal: self.flags.append("need_calendar_for_lunar"); return None
            Y = self.year_of(m[1], m[2]) if (m[1] or m[2]) else (a.year if (tet_range(cal, a.year) and tet_range(cal, a.year)[0] <= a) else a.year - 1)
            d = cal.lunar(Y, 1, 1)
            if not d: return None
            return self.span_tok("D", d - timedelta(1), d - timedelta(1), f"giao thừa Tết {Y}", "day")
        self.sub(r"\b(?:dem )?(?:giao thua|30 tet|29 tet)(?: tet)?(?: nam (20\d\d)| nam (nay|ngoai|truoc))?", giaothua)
        # lễ trong calendar.csv
        HOL = [(r"tet duong lich", "Tết Dương lịch"), (r"gio to(?: hung vuong)?", "Giỗ tổ"), (r"trung thu", "Trung thu"),
               (r"quoc khanh", "2/9"), (r"giai phong(?: mien nam)?", "30/4"), (r"quoc te lao dong", "1/5")]
        for rx, name in HOL:
            def hol(m, name=name):
                Y = self.year_of(m[1], m[2]) if (m[1] or m[2]) else None
                if Y is None:
                    Y = a.year; h = cal.holiday(name, Y) if cal else None
                    if h is None and name == "Tết Dương lịch": h = date(Y, 1, 1)
                    if h and h > a: Y -= 1
                    self.flags.append("holiday_year_inferred")
                d = cal.holiday(name, Y) if cal else None
                if d is None and name == "Tết Dương lịch": d = date(Y, 1, 1)
                if d is None: self.flags.append("need_calendar_for_lunar"); return None
                return self.span_tok("D", d, d, f"{name} {Y}", "day")
            self.sub(rf"\b{rx}(?: nam (20\d\d)| nam (nay|ngoai|truoc))?", hol)

    def lex_months(self):
        a = self.a
        def mon(M, Y, label=None, noyr=False):
            if not 1 <= M <= 12: return None
            if noyr: self.flags.append("month_no_year")
            return self.span_tok("M", *month_span(Y, M), label or f"{M}/{Y}", "month")
        self.sub(r"\bthang (\d{1,2}) ?(?:/|nam|-|\.) ?(20\d\d)(?!\d)", lambda m: mon(int(m[1]), int(m[2])))
        self.sub(r"(?<![\d/.])(\d{1,2}) ?/ ?(20\d\d)(?![\d/])", lambda m: mon(int(m[1]), int(m[2])))
        self.sub(r"\bthang (\d{1,2}) nam (nay|ngoai|truoc|qua)\b", lambda m: mon(int(m[1]), self.year_of(None, m[2])))
        def noyr(m):
            M = int(m[1])
            if not 1 <= M <= 12: return None
            Y = a.year if date(a.year, M, 1) <= a else a.year - 1
            return mon(M, Y, f"{M}/{Y} (không ghi năm)", True)
        self.sub(r"\bthang (\d{1,2})(?![\d/])(?! ?(?:/|nam|-) ?\d)(?! am lich)", noyr)

    def lex_tet(self):
        a, cal = self.a, self.cal
        def Y_of(ys):
            if ys is None:
                tr = tet_range(cal, a.year); self.flags.append("tet_year_inferred")
                return a.year if (tr and tr[0] <= a) else a.year - 1
            if ys.isdigit(): return int(ys)
            if ys in ("nay",): return a.year
            if ys in ("ngoai", "truoc", "qua"): return a.year - 1
            if ys in ("sau", "toi"): return a.year + 1
            if ys in _CAN_CHI: return _CAN_CHI[ys]
            # vừa rồi / gần nhất
            tr = tet_range(cal, a.year); self.flags.append("tet_most_recent")
            return a.year if (tr and tr[0] <= a) else a.year - 1
        # trước / sau Tết
        def pre(m):
            n = int(m[1]) if m[1] else None
            Y = Y_of(m[2]) if m[2] else None
            return self.new("B", mode="before_tet", n=n, year=Y if m[2] else None)
        self.sub(r"(?:so (?:voi|vs) )?(?:(\d+) ngay )?truoc tet(?: nam (20\d\d|nay|ngoai|truoc))?(?![\w])", pre)
        def post(m):
            n = int(m[1]) if m[1] else None
            return self.new("B", mode="after_tet", n=n, year=Y_of(m[2]) if m[2] else None)
        self.sub(r"(?:so (?:voi|vs) )?(?:(\d+) ngay )?sau tet(?: nam (20\d\d|nay|ngoai|truoc))?(?![\w])", post)
        # Tết chính
        def tet(m):
            ys = m[1] if m.lastindex else None
            ys = ys.strip() if ys else None
            Y = Y_of(ys)
            tr = tet_range(cal, Y)
            if not tr: self.flags.append("need_calendar_for_tet"); return None
            self.flags.append("lunar_expression: tra calendar.csv (bỏ tháng nhuận trừ khi nói rõ)") if False else None
            return self.span_tok("T", tr[0], tr[1], f"Tết {Y}", "tet")
        cc = "|".join(_CAN_CHI)
        self.sub(rf"(?:\b(?:dip|mua|ky nghi|thoi diem|luc) )?\btet(?: am lich)?(?: nam)? (20\d\d|nay|ngoai|truoc|qua|sau|toi|vua roi|vua qua|gan nhat|{cc})(?![\w])", tet)
        self.sub(rf"(?:\b(?:dip|mua|ky nghi|thoi diem|luc) )?\btet(?: nguyen dan| am lich)?(?![\w])(?! duong lich)", lambda m: tet(type("M", (), {"lastindex": None, "__getitem__": lambda s, i: None})()) if not re.search(r"banh $", self.s[:m.start()]) else None)

    def lex_price(self):
        s = self.s
        has_cue = re.search(r"\b(?:tang gia|doi gia|len gia|dieu chinh gia|thay doi gia)\b", s)
        n_after = re.search(r"(\d+) ngay (?:ke tu|dau|sau|tiep theo|tiep sau)", s)
        n_before = re.search(r"(\d+) ngay (?:ngay |lien |sat )?truoc(?: do| day| ngay do| ngay tang gia| ngay doi gia| ngay len gia)?", s)
        if not (n_after and n_before) and not (has_cue and (n_after or n_before)): return
        # anchor: ngày tường minh đứng cạnh cụm giá (ưu tiên ngày sau 'vao|tu|ke tu ngay'); các token ngày còn lại = cùng một ngày
        dts = [(m.start(), self.tk(m[0])["p"].start) for m in re.finditer(_TOK, s) if m[0][1] == "D"]
        uniq = sorted({d for _, d in dts})
        anchor = None
        if len(uniq) == 1: anchor = uniq[0]
        elif len(uniq) > 1:
            m = re.search(rf"(?:vao|tu|ke tu|khi)(?: ngay)? ({_TOK})", s)
            anchor = self.tk(m[1])["p"].start if m and m[1][1] == "D" else uniq[0]
            self.flags.append("price_anchor_ambiguous")
        na = int(n_after[1]) if n_after else int(n_before[1]); nb = int(n_before[1]) if n_before else na
        self.sp.price_effect = {"n_after": na, "n_before": nb, "anchor": anchor}
        # xoá cụm N ngày và token ngày khỏi chuỗi để không bị coi là kỳ độc lập
        s = re.sub(r"(\d+) ngay (?:ke tu|dau|sau|tiep theo|tiep sau)(?: (?:ngay )?(?:do|tang gia|doi gia|len gia|@D\d+@))?", " ", s)
        s = re.sub(r"(\d+) ngay (?:ngay |lien |sat )?truoc(?: do| day| ngay do| ngay tang gia| ngay doi gia| ngay len gia| (?:ngay )?@D\d+@)?", " ", s)
        s = re.sub(r"(?:vao |tu |ke tu )?(?:ngay )?@D\d+@", " ", s)
        self.s = re.sub(r"\s+", " ", s)
        if anchor:
            (m0, m1), (b0, b1) = price_effect_periods(anchor, na, nb)
            self.sp.periods += [Period(m0, m1, f"{na} ngày kể từ {anchor}", "main", "range"),
                                Period(b0, b1, f"{nb} ngày ngay trước {anchor}", "baseline", "range"),
                                Period(anchor, anchor, f"ngày {anchor}", "anchor", "day")]
        else:
            self.sp.needs_data.append("price_raise_date")

    def lex_baseline_desc(self):
        P = r"(?:so (?:voi|vs|sanh voi) )?"
        self.sub(P + r"cung (?:ky|thang|quy|tuan|ngay|thoi diem) (?:cua )?nam (?:truoc|ngoai|qua)\b", lambda m: self.new("B", mode="yoy"))
        self.sub(r"so (?:voi|vs) (?:thang|quy) nay nam (?:truoc|ngoai)\b", lambda m: self.new("B", mode="yoy"))
        self.sub(P + r"cung (?:ky|ngay|thoi diem) (?:cua )?thang (?:truoc|roi|ngoai|qua|lien truoc|truoc do)(?: do)?\b", lambda m: self.new("B", mode="prev_month"))
        self.sub(P + r"cung (?:ky|ngay|thu|thoi diem) (?:cua )?tuan (?:truoc|roi|qua)(?: do)?\b", lambda m: self.new("B", mode="prev_week"))
        self.sub(P + r"cung (?:ky|thoi diem) (?:cua )?quy (?:truoc|roi)\b", lambda m: self.new("B", mode="prev_quarter"))
        self.sub(r"so (?:voi|vs) cung ky(?! (?:thang|nam|quy|tuan))", lambda m: (self.flags.append("cung_ky_unit_unknown") or self.new("B", mode="prev")))
        # 'N ngày liền trước (đó)' ngoài ngữ cảnh đổi giá = kỳ N ngày ngay trước kỳ main
        self.sub(r"(?:so (?:voi|vs) )?(\d+) ngay (?:lien |ngay |sat )?truoc(?: do| day)?\b", lambda m: self.new("B", mode="nbefore", n=int(m[1])))

    def lex_relative(self):
        a = self.a
        def R(s, e, label, kind, todate=False, mode=None, flag=None):
            if flag: self.flags.append(flag)
            return self.new("R", p=Period(s, e, label, "main", kind, todate=todate), rel=True, mode=mode)
        if sum(1 for t in self.toks if t["kind"] in "ML") >= 2 and re.search(r"\bthang (?:sau|truoc)\b", self.s):
            self.s = re.sub(r"\bthang (?:sau|truoc)\b", " ", self.s); self.flags.append("series_trend_relative_months_ignored")
        def prevm():
            y, mo = add_months(a.year, a.month, -1); return month_span(y, mo)
        # tổ hợp 'năm ngoái'
        self.sub(r"\bthang nay nam (?:ngoai|truoc)\b", lambda m: R(*month_span(a.year - 1, a.month), "tháng này năm ngoái", "month", flag="month_this_year_last_full"))
        self.sub(r"\bquy nay nam (?:ngoai|truoc)\b", lambda m: R(*quarter_span(a.year - 1, (a.month - 1) // 3 + 1), "quý này năm ngoái", "quarter"))
        # ngày
        self.sub(r"\b(?:ngay )?hom nay\b|(?<!nhung )(?<!cac )(?<!may )\bngay nay\b", lambda m: R(a, a, "hôm nay", "day"))
        self.sub(r"\b(?:ngay )?(?:hom qua|bua qua)\b", lambda m: R(a - timedelta(1), a - timedelta(1), "hôm qua", "day"))
        self.sub(r"\b(?:ngay )?hom truoc\b", lambda m: R(a - timedelta(1), a - timedelta(1), "hôm trước (=hôm qua)", "day", flag="hom_truoc_as_yesterday"))
        self.sub(r"\bhom kia\b", lambda m: R(a - timedelta(2), a - timedelta(2), "hôm kia", "day"))
        self.sub(r"\b(?:ngay mai|hom sau)\b", lambda m: R(a + timedelta(1), a + timedelta(1), "ngày mai", "day", flag="future"))
        # tuần
        self.sub(r"\btuan nay\b", lambda m: R(week_monday(a), a, "tuần này (T2→hôm nay)", "week", todate=True))
        def wk_prev(m):
            m0 = week_monday(a); fl = "tuan_qua_ambiguous" if m[1] in ("qua", "vua qua") else None
            return R(m0 - timedelta(7), m0 - timedelta(1), "tuần trước (T2→CN)", "week", mode="prev", flag=fl)
        self.sub(r"\btuan (truoc|roi|vua roi|vua qua|qua)\b", wk_prev)
        self.sub(r"\btuan (?:sau|toi|ke tiep)\b", lambda m: R(week_monday(a) + timedelta(7), week_monday(a) + timedelta(13), "tuần sau", "week", flag="future"))
        # tháng
        self.sub(r"\bthang nay\b", lambda m: R(a.replace(day=1), a, "tháng này (1→hôm nay)", "month", todate=True))
        def mo_prev(m):
            fl = "thang_qua_ambiguous" if m[1] in ("qua", "vua qua") else None
            return R(*prevm(), "tháng trước", "month", mode="prev", flag=fl)
        self.sub(r"\bthang (truoc|roi|vua roi|vua qua|qua|lien truoc|truoc do|ngay truoc)\b", mo_prev)
        self.sub(r"\bthang (?:sau|toi|ke tiep)\b", lambda m: R(*month_span(*add_months(a.year, a.month, 1)), "tháng sau", "month", flag="future"))
        # quý
        self.sub(r"\bquy nay\b", lambda m: R(quarter_span(a.year, (a.month - 1) // 3 + 1)[0], a, "quý này (đầu quý→hôm nay)", "quarter", todate=True))
        def q_prev(m):
            q = (a.month - 1) // 3 + 1; y, q = (a.year, q - 1) if q > 1 else (a.year - 1, 4)
            return R(*quarter_span(y, q), "quý trước", "quarter", mode="prev")
        self.sub(r"\bquy (?:truoc|roi|vua roi|qua)\b", q_prev)
        self.sub(r"\bquy (?:sau|toi)\b", lambda m: R(*quarter_span(*((a.year, (a.month - 1) // 3 + 2) if (a.month - 1) // 3 < 3 else (a.year + 1, 1))), "quý sau", "quarter", flag="future"))
        # năm
        self.sub(r"\bnam nay\b", lambda m: R(date(a.year, 1, 1), a, "năm nay (1/1→hôm nay)", "year", todate=True))
        self.sub(r"\bnam (?:ngoai|truoc|qua|vua roi)\b", lambda m: R(date(a.year - 1, 1, 1), date(a.year - 1, 12, 31), "năm ngoái", "year", mode="prev"))
        self.sub(r"\bnam (?:sau|toi)\b", lambda m: R(date(a.year + 1, 1, 1), date(a.year + 1, 12, 31), "năm sau", "year", flag="future"))

    # ---- 2. ngữ pháp ----
    def base_span(self, mk, side):
        """Lấy ngày đầu/cuối của một token (D,M,Q,Y,H,R,T,K,S,N)."""
        t = self.tk(mk); k = t["kind"]
        if k == "S": return t["s"]
        if k == "N": return self.a
        p = t["p"]; return p.start if side == "start" else p.end

    def grammar(self):
        s, a = self.s, self.a
        periods = []     # (pos, Period, tokenkind, tokidx)
        consumed = set()
        def mark(mk): consumed.add(int(re.search(r"\d+", mk).group()))
        # (a) từ X đến Y
        for m in list(re.finditer(rf"\b(?:ke |tinh )?tu (?:ngay |hom )?({_TOK}) ?(@U\d+@)", s)):
            X, U = m[1], m[2]
            if X[1] not in "DMQYHRSN": continue
            u = self.tk(U); inner = self.toks[u["inner"]]
            st = self.base_span(X, "start"); en = inner["p"].end
            if inner["kind"] == "R" and inner["p"].kind in ("month", "quarter", "year", "week") and inner["p"].todate: en = inner["p"].end
            if en < st: self.flags.append("range_reversed"); st, en = en, st
            kx, ki = self.tk(X)["kind"], inner["kind"]
            per = Period(st, en, f"{st}→{en}", "main", "range", todate=(en == a and inner.get("rel", False)))
            per._ends = (kx, ki)
            periods.append((m.start(), per, "G", None)); mark(X); mark(U)
            if X[1] == "D" and self.tk(X)["p"].label and "day_month_no_year" in self.flags: pass
        s2 = s
        # (b) 'từ đầu năm' không có U: [start, hôm nay]
        for m in re.finditer(r"(@S\d+@)(?! ?@U)", s):
            if int(re.search(r"\d+", m[1]).group()) in consumed: continue
            t = self.tk(m[1]); periods.append((m.start(), Period(t["s"], a, f"đầu {t['unit']} → hôm nay", "main", "range", todate=True), "G", None)); mark(m[1])
        for m in re.finditer(r"(@S\d+@) ?(@U\d+@)", s):  # S đứng riêng + U nhưng thiếu 'tu'
            if int(re.search(r"\d+", m[1]).group()) in consumed: continue
            t = self.tk(m[1]); inner = self.toks[self.tk(m[2])["inner"]]
            periods.append((m.start(), Period(t["s"], inner["p"].end, f"đầu {t['unit']} → {inner['p'].end}", "main", "range", todate=inner["p"].end == a), "G", None)); mark(m[1]); mark(m[2])
        # (c) N ... U : từ nay đến ...
        for m in re.finditer(r"(@N\d+@) ?(@U\d+@)", s):
            inner = self.toks[self.tk(m[2])["inner"]]
            periods.append((m.start(), Period(a, inner["p"].end, f"{a}→{inner['p'].end}", "main", "range"), "G", None)); mark(m[1]); mark(m[2])
        # (d) D - D  /  giữa D và D
        for m in re.finditer(rf"(?:giua )?(@D\d+@) ?(?:-|–|—|va) ?(@D\d+@)", s):
            if any(int(x) in consumed for x in re.findall(r"\d+", m[0])): continue
            d1, d2 = self.base_span(m[1], "start"), self.base_span(m[2], "end")
            if d1 <= d2: periods.append((m.start(), Period(d1, d2, f"{d1}→{d2}", "main", "range"), "G", None)); mark(m[1]); mark(m[2])
        # (e) 'từ D' / 'kể từ D' mở: [D, hôm nay]
        for m in re.finditer(rf"\b(?:ke tu|tinh tu|tu) (?:ngay )?(@D\d+@)", s):
            if int(re.search(r"\d+", m[1]).group()) in consumed: continue
            d = self.tk(m[1])["p"].start
            periods.append((m.start(), Period(d, max(a, d), f"{d}→hôm nay (kỳ mở)", "main", "range", todate=True), "G", None)); mark(m[1]); self.flags.append("open_ended_from")
        # (f) mốc tồn: U lẻ / C
        stock_dates = []
        for m in re.finditer(r"@([UC])(\d+)@", s):
            i = int(m[2])
            if i in consumed: continue
            t = self.toks[i]; consumed.add(i)
            inner = self.toks[t["inner"]] if t.get("inner") is not None else None
            if inner is None:
                dd = [x for x in re.finditer(r"@[DR](\d+)@", s) if self.toks[int(x[1])]["p"].kind == "day" and int(x[1]) not in consumed]
                d = self.toks[int(dd[0][1])]["p"].start if dd else a
                if dd: consumed.add(int(dd[0][1]))
            else:
                d = inner["p"].end
                for j, tt in enumerate(self.toks):
                    if tt is inner: consumed.add(j)
            stock_dates.append(d)
        # (f2) 'tồn kho ngày D', 'còn nợ ngày D' = trạng thái cuối ngày D
        for m in re.finditer(rf"\b(?:ton kho|ton cuoi|ton|con no|dang no|kho)(?: cuoi)?(?: vao| luc| tai| theo)?(?: ngay)? (@D\d+@)", s):
            i = int(re.search(r"\d+", m[1]).group())
            if i in consumed: continue
            consumed.add(i); stock_dates.append(self.toks[i]["p"].start)
        for m in re.finditer(rf"\btheo (?:ton kho|so du)(?: cuoi)?(?: vao| luc| tai)?(?: ngay)? (@D\d+@)", s):
            i = int(re.search(r"\d+", m[1]).group())
            if i in consumed: continue
            consumed.add(i); stock_dates.append(self.toks[i]["p"].start)
        # (g) token đơn lẻ còn lại
        singles = []
        for m in re.finditer(r"@([A-Z])(\d+)@", s):
            k, i = m[1], int(m[2])
            if i in consumed: continue
            t = self.toks[i]
            if k in "DMQYHRKFT": singles.append((m.start(), i, t))
            elif k == "L":
                for p in t["ps"]: periods.append((m.start(), p, "L", i))
                consumed.add(i); self.sp.series = True
        return periods, singles, consumed, stock_dates

    def build(self):
        sp, a, s = self.sp, self.a, self.s
        periods, singles, consumed, stock_dates = self.grammar()
        # --- kỳ so sánh trỏ bằng 'so với <kỳ>' ---
        base_ptr = {}
        for m in re.finditer(rf"(?:so (?:voi|vs|sanh voi|sanh)|\bvs\b|\bvoi\b|hon hay kem|hon kem|hay kem|(?:nhieu|it|cao|thap) hon hay (?:it|nhieu|thap|cao) hon|(?:nhieu|it|cao|thap) hon|hay kem|tang hay giam so voi) ?(?:cung ky )?({_TOK})", s):
            if m[1][1] in "MQYDRHTKF": base_ptr[int(re.search(r"\d+", m[1]).group())] = m
        cands = []   # (pos, Period, tokidx, tok)
        for pos, i, t in singles:
            p = t["p"]; p = Period(p.start, p.end, p.label, "main", p.kind, p.target, p.todate); cands.append((pos, p, i, t))
        for pos, p, k, i in periods: cands.append((pos, p, i, None))
        cands.sort(key=lambda x: x[0])
        distinct = {(p.start, p.end) for _, p, _, _ in cands}
        if len(distinct) >= 3 and all(p.kind in ("month", "quarter") for _, p, _, _ in cands) and not re.search(r"@B\d+@", s):
            base_ptr = {}; sp.series = True
            seen_d = set(); cands = [c for c in cands if not ((c[1].start, c[1].end) in seen_d or seen_d.add((c[1].start, c[1].end)))]
        mains, baseline_explicit = [], []
        for pos, p, i, t in cands:
            if i is not None and i in base_ptr and t is not None:
                baseline_explicit.append((p, t))
            else: mains.append((pos, p, t))
        # nếu chỉ có 1 kỳ và nó bị coi là baseline (vd 'tháng trước' đứng một mình sau 'so với') -> main
        if not mains and baseline_explicit:
            p, t = baseline_explicit.pop(0); mains.append((0, p, t))
        main_ps = [p for _, p, _ in mains]
        # --- series ---
        month_like = [p for p in main_ps if p.kind in ("month", "quarter")]
        if len(main_ps) >= 2 and len(month_like) == len(main_ps) and not baseline_explicit and not any(m for m in re.findall(r"@B\d+@", s)):
            if re.search(r"lien tuc|tung thang|tung quy|moi thang|theo thu tu|cac thang|lien tiep|lien nhau|thang cu den thang moi|tang lien|giam lien", s) or sp.series:
                sp.series = True
        # ranges M..M -> series (trừ khi hỏi tổng)
        newm = []
        for p in main_ps:
            e = getattr(p, "_ends", None)
            if e and e[0] == "M" and e[1] == "M" and p.start.day == 1 and p.end == month_end(p.end) and (p.end.year, p.end.month) != (p.start.year, p.start.month):
                if not re.search(r"\btong\b|cong lai|tat ca|ca giai doan|gop|toan bo|tich luy", s):
                    sp.series = True; y, mo = p.start.year, p.start.month
                    while (y, mo) <= (p.end.year, p.end.month): newm.append(Period(*month_span(y, mo), f"{mo}/{y}", "main", "month")); y, mo = add_months(y, mo, 1)
                    continue
            newm.append(p)
        main_ps = newm
        if sp.series: main_ps.sort(key=lambda p: p.start)
        for p in main_ps: sp.periods.append(p)
        # --- baseline từ trỏ tường minh ---
        mainp = main_ps[0] if main_ps else None
        for p, t in baseline_explicit:
            rel_prev = t["kind"] == "R" and t.get("mode") == "prev"
            if rel_prev and mainp is not None:
                unit = p.kind
                if unit == "year" and mainp.kind not in ("year",) and _classify(mainp, a) not in ("year", "ytd"): mode = "yoy"
                else: mode = "prev"
                b = derive_baseline(mainp, mode, a, self.flags)
            else: b = Period(p.start, p.end, p.label, "baseline", p.kind)
            sp.periods.append(b)
        # --- mô tả kỳ so sánh B ---
        for m in re.finditer(r"@B(\d+)@", s):
            t = self.toks[int(m[1])]; mode = t["mode"]
            if mode in ("before_tet", "after_tet"):
                tets = [p for p in main_ps if p.kind == "tet"]
                Y = t.get("year")
                if tets:
                    tp = tets[0]; n = t.get("n") or tp.days()
                    st = (tp.start - timedelta(n), tp.start - timedelta(1)) if mode == "before_tet" else (tp.end + timedelta(1), tp.end + timedelta(n))
                    sp.periods.append(Period(*st, f"{n} ngày {'trước' if mode == 'before_tet' else 'sau'} Tết", "baseline", "range"))
                else:
                    Y = Y or (a.year if (tet_range(self.cal, a.year) and tet_range(self.cal, a.year)[0] <= a) else a.year - 1)
                    r = (before_tet if mode == "before_tet" else after_tet)(self.cal, Y, t.get("n"))
                    if r:
                        sp.periods.append(Period(*r, f"{'trước' if mode == 'before_tet' else 'sau'} Tết {Y}", "main", "range"))
                        if not t.get("n"): self.flags.append("tet_window_length_assumed_same_as_tet")
                    else: self.flags.append("need_calendar_for_tet")
                continue
            if not mainp: self.flags.append("baseline_without_main"); continue
            if mode == "nbefore":
                n = t["n"]; sp.periods.append(Period(mainp.start - timedelta(n), mainp.start - timedelta(1), f"{n} ngày liền trước", "baseline", "range")); continue
            sp.periods.append(derive_baseline(mainp, mode, a, self.flags))
        # --- tăng trưởng ngầm (không nói so với gì) ---
        if mainp and not sp.baseline() and not sp.price_effect and len(main_ps) == 1 and not sp.series \
           and re.search(r"tang truong|tang hay giam|tang/giam|giam hay tang|thay doi|hon hay kem|chenh", s) and re.search(r"%|phan tram|\bpt\b|\bbn %", s) \
           and not (len(sp.shifts) >= 2) and mainp.kind not in ("day",) or (mainp and not sp.baseline() and not sp.price_effect and len(main_ps) == 1 and mainp.kind == "day" and False):
            if not (len(sp.shifts) >= 2):
                sp.periods.append(derive_baseline(mainp, "prev", a, self.flags)); self.flags.append("baseline_implicit_prev_period")
        # --- stock ---
        st_date = stock_dates[0] if stock_dates else None
        state = re.search(_STATE, s)
        if st_date is None and state:
            ds = [p for p in main_ps if p.kind == "day"]
            if ds and len(main_ps) == 1:
                st_date = ds[0].start; sp.periods.remove(ds[0]); self.flags.append("state_question_on_day")
        expiry = re.search(r"het han|han dung|han su dung|\bdate\b|can date|sap het han", s)
        if st_date is None and ((state and (expiry or not (main_ps or sp.periods))) or (expiry and not main_ps_non_expiry(sp))):
            st_date = a; self.flags.append("implicit_today_stock")
        if st_date is not None:
            sp.stock, sp.stock_date = True, st_date
            sp.periods = [p for p in sp.periods if not (p.kind == "day" and p.start == st_date and p.role == "main" and len(main_ps) == 1 and not expiry)] if False else sp.periods
            if stock_dates and not any(p.kind == "stock" for p in sp.periods):
                sp.periods.insert(0, Period(st_date, st_date, f"đến hết {st_date}", "main", "stock"))
            # cộng dồn cũng ghi nhận nếu không có từ khoá trạng thái
            if not state and not stock_dates == []: self.flags.append("cumulative_until_date")
        if expiry:
            for p in sp.periods:
                if p.kind in ("range", "forward") and p.role == "main": p.target = "expiry"
        # --- ca ---
        if sp.shifts:
            sp.shift = sp.shifts[0]
            if len(sp.shifts) >= 2 and re.search(r"so voi|\bvs\b|chenh|hon hay kem|hon kem|nhieu hon|it hon|cao hon|thap hon|so sanh", s):
                sp.baseline_shift = sp.shifts[1]
        # --- dọn ---
        seen, out = set(), []
        for p in sp.periods:
            k = (p.start, p.end, p.role)
            if k not in seen: seen.add(k); out.append(p)
        sp.periods = out
        for p in sp.periods:
            if p.role == "main" and p.kind not in ("stock", "forward") and p.target != "expiry":
                if p.start > a: sp.flags.append(f"period_in_future:{p.label}")
                elif p.start <= a < p.end and p.kind not in ("tet",): sp.flags.append(f"period_extends_past_as_of:{p.label}")
            elif p.role == "main" and p.kind == "tet" and p.start <= a < p.end: sp.flags.append(f"tet_in_progress:{p.label}")
        if re.search(r"dau thang|cuoi thang|giua thang|nua (?:dau|cuoi)(?! nam)|cuoi tuan|dau tuan|dau quy|cuoi quy|cuoi nam|dau nam(?! nay)", self.s) and not any(p.kind == "range" and "đầu" in p.label for p in sp.periods):
            sp.flags.append("vague_part_of_period (rulebook không định nghĩa -> abstain nếu không chắc)")
        sp.time_cue = bool(_TIMECUE.search(self.s)) and not re.fullmatch(r"[\d\W]*", self.s)
        # dedupe flags
        sp.flags = list(dict.fromkeys(sp.flags))
        return sp

def main_ps_non_expiry(sp):
    """kỳ main KHÔNG phải cửa sổ hạn dùng (ngày đặt đơn...): nếu có thì câu hạn dùng không tự suy ra tồn hôm nay."""
    return [p for p in sp.periods if p.role == "main" and p.kind not in ("forward", "range", "stock")]

def resolve(question, as_of, cal=None):
    """as_of: date | 'YYYY-MM-DD'. Trả TimeSpec. Không bao giờ đoán thầm: điều gì suy luận / mơ hồ thì ghi vào .flags."""
    r = _Resolver(question, _to_date(as_of), cal)
    r.lex()
    return r.build()

# ───────────────────────── phạm vi dữ liệu ─────────────────────────
def availability(p, data_start, data_end, as_of):
    """full | partial | outside | future.  'outside' => unanswerable (không phải 0).  partial => tính trên phần có."""
    hi = min(data_end, as_of) if as_of else data_end
    if p.end < data_start or p.start > hi: return "future" if p.start > as_of else "outside"
    return "full" if p.start >= data_start and p.end <= hi else "partial"

def effective(p, data_start, data_end, as_of):
    return max(p.start, data_start), min(p.end, data_end, as_of)

# ───────────────────────── B. ô ngày trong dữ liệu ─────────────────────────
_ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$")
_DMY = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2}|\d{4})(?:[ T,]+\d{1,2}[:h]\d{2}(?::\d{2})?)?$")
_THG = re.compile(r"^(?:th[uứ]\s*\w+,?\s*)?(\d{1,2})\s*(?:thg|th[áa]ng|tháng)\s*(\d{1,2}),?\s*(\d{4})$", re.I)
_DM = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})$")
_YMD8 = re.compile(r"^(20\d\d)(\d{2})(\d{2})$")
_SER = re.compile(r"^\d{5}(?:\.\d+)?$")

def excel_serial(x, base1904=False):
    return (datetime(1904, 1, 1) if base1904 else datetime(1899, 12, 30)) + timedelta(days=float(x))

def parse_cell_date(v, sheet_year=None, dayfirst=True):
    """Trả date hoặc None. KHÔNG bao giờ để pandas đoán cả cột: gọi từng ô.
    - ISO (kể cả có giờ) không bao giờ bị đảo ngày/tháng.
    - dd/mm/yyyy: mặc định dayfirst (kiểu VN); nếu phần đầu > 12 thì chắc chắn là ngày; nếu phần giữa > 12 thì là mm/dd.
    - 'dd/mm' không năm: lấy năm của SHEET chứa dòng đó (rulebook §8).
    - serial Excel: phần thập phân là giờ -> bỏ; chỉ nhận 30000..60000 (1982..2064).
    - số nguyên 10 chữ số / 13 chữ số: epoch giây / mili-giây, đổi sang UTC+7 (phòng thủ)."""
    if v is None: return None
    if isinstance(v, datetime): return v.date()
    if isinstance(v, date): return v
    s = str(v).strip().replace("\xa0", " ")
    if s in ("", "nan", "NaT", "None", "-"): return None
    try:
        m = _ISO.match(s)
        if m: return _mk(int(m[1]), int(m[2]), int(m[3]))
        m = _THG.match(s)
        if m: return _mk(int(m[3]), int(m[2]), int(m[1]))
        m = _DMY.match(s)
        if m:
            a_, b_, y = int(m[1]), int(m[2]), int(m[3]); y = y + 2000 if y < 100 else y
            if a_ > 12 and b_ <= 12: return _mk(y, b_, a_)
            if b_ > 12 and a_ <= 12: return _mk(y, a_, b_)
            return _mk(y, b_, a_) if dayfirst else _mk(y, a_, b_)
        m = _DM.match(s)
        if m and sheet_year: return _mk(int(sheet_year), int(m[2]), int(m[1]))
        m = _YMD8.match(s)
        if m: return _mk(int(m[1]), int(m[2]), int(m[3]))
        if _SER.match(s) and 30000 <= float(s) <= 60000: return excel_serial(s).date()
        if re.fullmatch(r"\d{10}|\d{13}", s):
            x = int(s) / (1000 if len(s) == 13 else 1); return (datetime(1970, 1, 1) + timedelta(seconds=x, hours=7)).date()
    except (ValueError, OverflowError):
        return None
    return None

def parse_time_of_day(v):
    """-> số giây từ 00:00 (hoặc None). Nhận 'HH:MM[:SS]', ISO có giờ, phần thập phân serial Excel."""
    s = str(v).strip()
    m = re.search(r"(\d{1,2})[:h](\d{2})(?::(\d{2}))?", s)
    if m: return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3] or 0)
    if _SER.match(s) and "." in s: return round((float(s) % 1) * 86400)
    return None

def in_shift(sec, shift):
    """ca sáng 06:00–13:59(:59), ca chiều 14:00–21:59(:59); 22:00–05:59 không thuộc ca nào."""
    if sec is None: return False
    return (6 * 3600 <= sec < 14 * 3600) if shift == "sang" else (14 * 3600 <= sec < 22 * 3600)
