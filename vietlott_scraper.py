#!/usr/bin/env python3
"""
Lấy toàn bộ kết quả xổ số Vietlott (tất cả các kỳ) từ https://vietlott.vn/
và lưu ra CSV, mỗi sản phẩm một file.

Sản phẩm hỗ trợ: 645 (Mega 6/45), 655 (Power 6/55), 535 (Lotto 5/35),
max3d (Max 3D / Max 3D+), max3dpro (Max 3D Pro), keno, bingo18.

Chỉ dùng thư viện chuẩn của Python (>= 3.8).

Ví dụ:
    python3 vietlott_scraper.py                     # tất cả sản phẩm
    python3 vietlott_scraper.py 645 655 max3d       # chỉ một số sản phẩm
    python3 vietlott_scraper.py keno --workers 8    # tăng số request song song
    python3 vietlott_scraper.py --max-pages 3       # chạy thử vài trang

Chạy lại lệnh sẽ tự động tiếp tục phần bị gián đoạn và chỉ lấy thêm các kỳ mới.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://vietlott.vn"
RESULT_PAGE = BASE + "/vi/trung-thuong/ket-qua-trung-thuong/"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

RENDER_INFO = {
    "SiteId": "main.frontend.vi", "SiteAlias": "main.vi", "UserSessionId": "",
    "SiteLang": "vi", "IsPageDesign": False, "ExtraParam1": "", "ExtraParam2": "",
    "ExtraParam3": "", "SiteURL": "", "WebPage": None, "SiteName": "Vietlott",
    "OrgPageAlias": None, "PageAlias": None, "FullPageAlias": None, "RefKey": None,
    "System": 1,
}


# ---------------------------------------------------------------- HTTP

def http(url, body=None, method_name=None, retries=5):
    headers = {"User-Agent": UA}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "text/plain; charset=utf-8"
        headers["X-AjaxPro-Method"] = method_name
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  ! lỗi mạng ({e}), thử lại sau {wait}s", file=sys.stderr)
            time.sleep(wait)


def ajax(webpart, params):
    url = f"{BASE}/ajaxpro/Vietlott.PlugIn.WebParts.{webpart},Vietlott.PlugIn.WebParts.ashx"
    value = json.loads(http(url, params, "ServerSideDrawResult"))["value"]
    if value.get("Error"):
        raise RuntimeError(value.get("InfoMessage"))
    return value


# ---------------------------------------------------------------- parsing helpers

def clean(html):
    return re.sub(r"\s+", " ", html)


def balls(fragment):
    return re.findall(r'<span class="bong_tron[^"]*">\s*(\w+)\s*</span>', fragment)


def rows(html):
    return re.findall(r"<tr[^>]*>(.*?)</tr>", clean(html))


def parse_lotto(html, n_main, has_bonus):
    """Mega 6/45, Power 6/55, Lotto 5/35: | Ngày | Kỳ | Bộ số |"""
    out = []
    for tr in rows(html):
        m = re.search(r"<td>(\d\d/\d\d/\d{4})</td>.*?id=(\d+)", tr)
        if not m:
            continue
        main, _, bonus = tr.partition("bong_tron-sperator")
        nums = balls(main)
        row = {"draw_id": m.group(2), "date": m.group(1)}
        row.update({f"n{i + 1}": v for i, v in enumerate(nums[:n_main])})
        if has_bonus:
            b = balls(bonus)
            row["bonus"] = b[0] if b else ""
        out.append(row)
    return out


MAX3D_PRIZES = [("giai_dac_biet", "đặc biệt"), ("giai_nhat", "nhất"),
                ("giai_nhi", "nhì"), ("giai_ba", "ba")]


def parse_max3d(html):
    """Max 3D / Max 3D Pro: mỗi <tr> là một kỳ, các giải chia theo <h5>."""
    out = []
    for tr in rows(html):
        m = re.search(r"id=(\d+)[^>]*>\d+</a>\s*\|\s*Ngày:\s*(\d\d/\d\d/\d{4})", tr)
        if not m:
            continue
        row = {"draw_id": m.group(1), "date": m.group(2)}
        parts = re.split(r"<h5>([^<]+)</h5>", tr)
        for title, body in zip(parts[1::2], parts[2::2]):
            groups = re.findall(r'class="day_so_ket_qua_v2[^"]*"[^>]*>(.*?)</div>', body)
            numbers = ["".join(balls(g)) for g in groups]
            for col, key in MAX3D_PRIZES:
                if title.strip().lower().endswith(key):
                    row[col] = " ".join(x for x in numbers if x)
        out.append(row)
    return out


def parse_keno(html):
    out = []
    for tr in rows(html):
        m = re.search(r"id=(\d+)[^>]*>(\d\d/\d\d/\d{4})", tr)
        if not m:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr)
        nums = balls(tds[1]) if len(tds) > 1 else []
        row = {"draw_id": m.group(1), "date": m.group(2)}
        row.update({f"n{i + 1}": v for i, v in enumerate(nums)})
        row["chan_le"] = _text(tds[2]) if len(tds) > 2 else ""
        row["lon_nho"] = _text(tds[3]) if len(tds) > 3 else ""
        out.append(row)
    return out


def parse_bingo(html):
    out = []
    for tr in rows(html):
        m = re.search(r"id=(\d+)[^>]*>(\d\d/\d\d/\d{4})", tr)
        if not m:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr)
        nums = re.findall(r'<span class="bong_tron_bingo[^"]*">\s*(\d+)\s*</span>', tds[1]) if len(tds) > 1 else []
        row = {"draw_id": m.group(1), "date": m.group(2)}
        row.update({f"n{i + 1}": v for i, v in enumerate(nums)})
        row["tong"] = _text(tds[2]) if len(tds) > 2 else ""
        row["lon_hoa_nho"] = _text(tds[3]) if len(tds) > 3 else ""
        out.append(row)
    return out


def _text(fragment):
    return clean(re.sub(r"<[^>]+>", " ", fragment)).strip()


# ---------------------------------------------------------------- games

class LottoGame:
    """6/45, 6/55, 5/35: cần 'Key' lấy từ trang dò số."""

    def __init__(self, webpart, slug, n_main, n_bands, has_bonus, band_len):
        self.webpart, self.slug = webpart, slug
        self.n_main, self.n_bands, self.has_bonus, self.band_len = n_main, n_bands, has_bonus, band_len
        self.columns = ["draw_id", "date"] + [f"n{i + 1}" for i in range(n_main)] + (["bonus"] if has_bonus else [])
        self._key = None

    def key(self, refresh=False):
        if self._key is None or refresh:
            page = http(RESULT_PAGE + self.slug)
            self._key = re.search(r"ServerSideDrawResult\(RenderInfo, '([0-9a-f]+)'", page).group(1)
        return self._key

    def fetch(self, page):
        bands = [[""] * self.band_len for _ in range(self.n_bands)]
        params = {"ORenderInfo": RENDER_INFO, "GameDrawId": "", "ArrayNumbers": bands,
                  "CheckMulti": False, "PageIndex": page}
        try:
            value = ajax(self.webpart, dict(params, Key=self.key()))
        except RuntimeError:
            value = ajax(self.webpart, dict(params, Key=self.key(refresh=True)))
        return parse_lotto(value["HtmlContent"] or "", self.n_main, self.has_bonus)


class SimpleGame:
    def __init__(self, webpart, params, parser, columns):
        self.webpart, self.params, self.parser, self.columns = webpart, params, parser, columns

    def prepare(self):
        # Keno/Bingo18 chỉ trả các trang sau trang đầu khi gửi kèm đúng TotalRow.
        if "TotalRow" in self.params:
            value = ajax(self.webpart, dict(self.params, ORenderInfo=RENDER_INFO, PageIndex=0, TotalRow=0))
            self.params = dict(self.params, TotalRow=value["RetNumber"])

    def fetch(self, page):
        value = ajax(self.webpart, dict(self.params, ORenderInfo=RENDER_INFO, PageIndex=page))
        return self.parser(value["HtmlContent"] or "")


GAMES = {
    "645": LottoGame("Game645CompareWebPart", "winning-number-645", 6, 6, False, 18),
    "655": LottoGame("Game655CompareWebPart", "winning-number-655", 6, 5, True, 18),
    "535": LottoGame("Game535CompareWebPart", "winning-number-535", 5, 5, True, 35),
    "max3d": SimpleGame(
        "GameMax3DCompareWebPart",
        {"GameId": "5", "GameDrawId": "", "number01": "", "number02": "", "CheckMulti": 0},
        parse_max3d, ["draw_id", "date"] + [c for c, _ in MAX3D_PRIZES]),
    "max3dpro": SimpleGame(
        "GameMax3DProCompareWebPart",
        {"GameId": "7", "GameDrawId": "", "number01": "", "number02": ""},
        parse_max3d, ["draw_id", "date"] + [c for c, _ in MAX3D_PRIZES]),
    "keno": SimpleGame(
        "GameKenoCompareWebPart",
        {"GameId": "6", "GameDrawNo": "", "number": "", "DrawDate": "", "ProcessType": 0,
         "OddEven": 2, "UpperLower": 2, "TotalRow": 0},
        parse_keno, ["draw_id", "date"] + [f"n{i + 1}" for i in range(20)] + ["chan_le", "lon_nho"]),
    "bingo18": SimpleGame(
        "GameBingoCompareWebPart",
        {"GameId": "8", "GameDrawNo": "", "number": "", "DrawDate": "", "TotalRow": 0},
        parse_bingo, ["draw_id", "date", "n1", "n2", "n3", "tong", "lon_hoa_nho"]),
}


# ---------------------------------------------------------------- crawling

class Store:
    """CSV + file trạng thái để chạy tiếp khi bị ngắt và cập nhật kỳ mới."""

    def __init__(self, out_dir, name, columns):
        self.columns = columns
        self.csv_path = os.path.join(out_dir, f"{name}.csv")
        self.state_path = os.path.join(out_dir, f".{name}.state.json")
        self.rows = {}
        if os.path.exists(self.csv_path):
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.rows[r["draw_id"]] = r
        self.state = {"complete": False, "next_page": 0}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state = json.load(f)
        self._fh = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=columns, extrasaction="ignore")
        if not self.rows:
            self._fh.seek(0)
            self._fh.truncate()
            self._w.writeheader()

    def add(self, items):
        new = 0
        for r in items:
            if r["draw_id"] not in self.rows:
                self.rows[r["draw_id"]] = r
                self._w.writerow(r)
                new += 1
        self._fh.flush()
        return new

    def save_state(self, **kw):
        self.state.update(kw)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f)

    def finalize(self):
        """Ghi lại CSV đã sắp xếp theo kỳ quay, tăng dần."""
        self._fh.close()
        tmp = self.csv_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            w.writeheader()
            for k in sorted(self.rows, key=int):
                w.writerow(self.rows[k])
        os.replace(tmp, self.csv_path)


def crawl(game, start_page, pool, workers, stop_when_known, store, max_pages, delay):
    """Duyệt trang từ start_page; trả về trang tiếp theo, hoặc None nếu đã hết dữ liệu."""
    page = start_page
    while True:
        if max_pages is not None and page - start_page >= max_pages:
            return page
        pages = list(range(page, page + workers))
        results = list(pool.map(game.fetch, pages))
        for p, items in zip(pages, results):
            if not items:
                return None
            known = all(r["draw_id"] in store.rows for r in items)
            store.add(items)
            if stop_when_known and known:
                return None
        page = pages[-1] + 1
        if not stop_when_known:
            store.save_state(next_page=page)
        last = results[-1][-1]
        print(f"  trang {page:>6} | tổng {len(store.rows):>7} kỳ | tới kỳ #{last['draw_id']} ({last['date']})", flush=True)
        if delay:
            time.sleep(delay)


def run(name, out_dir, workers, max_pages, delay):
    game = GAMES[name]
    store = Store(out_dir, name, game.columns)
    print(f"== {name}: đã có {len(store.rows)} kỳ trong {store.csv_path}")
    if hasattr(game, "prepare"):
        game.prepare()
    try:
        with ThreadPoolExecutor(workers) as pool:
            # 1) Lấy các kỳ mới (trang đầu) cho tới khi gặp trang toàn kỳ đã có.
            if store.rows:
                crawl(game, 0, pool, workers, True, store, max_pages, delay)
            # 2) Lấy (tiếp) lịch sử cũ cho tới trang cuối cùng.
            if not store.state.get("complete"):
                # Lùi lại vài trang vì kỳ mới sẽ đẩy các kỳ cũ xuống trang sau.
                start = max(0, store.state.get("next_page", 0) - workers)
                nxt = crawl(game, start, pool, workers, False, store, max_pages, delay)
                if nxt is None:
                    store.save_state(complete=True, next_page=0)
    finally:
        store.finalize()
    print(f"== {name}: xong, {len(store.rows)} kỳ -> {store.csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Lấy toàn bộ kết quả xổ số Vietlott ra CSV.")
    ap.add_argument("games", nargs="*", metavar="GAME",
                    help="Sản phẩm: " + ", ".join(GAMES) + " (mặc định: tất cả)")
    ap.add_argument("-o", "--out", default="data", help="Thư mục lưu CSV (mặc định: data)")
    ap.add_argument("-w", "--workers", type=int, default=4, help="Số request song song (mặc định: 4)")
    ap.add_argument("--delay", type=float, default=0.2, help="Nghỉ giữa các lượt (giây, mặc định: 0.2)")
    ap.add_argument("--max-pages", type=int, help="Giới hạn số trang mỗi sản phẩm (để chạy thử)")
    args = ap.parse_args()

    unknown = [g for g in args.games if g not in GAMES]
    if unknown:
        ap.error(f"không có sản phẩm {', '.join(unknown)}; chọn trong: {', '.join(GAMES)}")
    os.makedirs(args.out, exist_ok=True)
    for name in args.games or list(GAMES):
        run(name, args.out, max(1, args.workers), args.max_pages, args.delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng. Chạy lại lệnh để tiếp tục.", file=sys.stderr)
        sys.exit(130)
