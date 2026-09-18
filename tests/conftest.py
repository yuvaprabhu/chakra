"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from screener.store import Store

# The real UDiFF header, in order.
UDIFF_HEADER = [
    "TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN",
    "TckrSymb", "SctySrs", "XpryDt", "FininstrmActlXpryDt", "StrkPric", "OptnTp",
    "FinInstrmNm", "OpnPric", "HghPric", "LwPric", "ClsPric", "LastPric",
    "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst", "ChngInOpnIntrst",
    "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd", "SsnId", "NewBrdLotQty",
    "Rmks", "Rsvd01", "Rsvd02", "Rsvd03", "Rsvd04",
]


def udiff_row(
    *,
    trad_dt: str,
    isin: str,
    symbol: str,
    series: str = "EQ",
    instr_type: str = "STK",
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 99.0,
    close: float = 104.0,
    prev_close: float = 100.0,
    volume: int = 10_000,
    turnover: float = 1_040_000.0,
    trades: int = 500,
) -> dict:
    row = {col: "" for col in UDIFF_HEADER}
    row.update(
        TradDt=trad_dt, BizDt=trad_dt, Sgmt="CM", Src="NSE",
        FinInstrmTp=instr_type, FinInstrmId="1001", ISIN=isin, TckrSymb=symbol,
        SctySrs=series, FinInstrmNm=f"{symbol} LIMITED",
        OpnPric=f"{open_}", HghPric=f"{high}", LwPric=f"{low}", ClsPric=f"{close}",
        LastPric=f"{close}", PrvsClsgPric=f"{prev_close}", SttlmPric=f"{close}",
        OpnIntrst="0", ChngInOpnIntrst="0", TtlTradgVol=str(volume),
        TtlTrfVal=f"{turnover}", TtlNbOfTxsExctd=str(trades), SsnId="F1",
        NewBrdLotQty="1",
    )
    return row


def udiff_csv(rows: list[dict]) -> str:
    return pd.DataFrame(rows, columns=UDIFF_HEADER).to_csv(index=False)


def udiff_zip(rows: list[dict], *, inner_name: str | None = None) -> bytes:
    buf = io.BytesIO()
    name = inner_name or "BhavCopy_NSE_CM_0_0_0_20240801_F_0000.csv"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, udiff_csv(rows))
    return buf.getvalue()


def default_rows(trad_dt: str = "2024-08-01") -> list[dict]:
    """Two equities, plus the non-equity rows UDiFF really carries."""
    return [
        udiff_row(trad_dt=trad_dt, isin="INE002A01018", symbol="RELIANCE",
                  open_=2900, high=2950, low=2880, close=2940, prev_close=2895,
                  volume=5_000_000, turnover=1.47e10, trades=120_000),
        udiff_row(trad_dt=trad_dt, isin="INE009A01021", symbol="INFY",
                  open_=1800, high=1820, low=1790, close=1810, prev_close=1795,
                  volume=3_000_000, turnover=5.43e9, trades=90_000),
        udiff_row(trad_dt=trad_dt, isin="INE123X01019", symbol="SMALLCO",
                  series="BE", open_=45, high=46, low=44, close=45.5,
                  prev_close=45, volume=12_000, turnover=546_000, trades=300),
        # Noise that must be filtered out:
        udiff_row(trad_dt=trad_dt, isin="INE002A01018", symbol="RELIANCE",
                  instr_type="FUTSTK", series="", close=2945),
        udiff_row(trad_dt=trad_dt, isin="INE002A01018", symbol="NIFTY",
                  instr_type="IDX", series="", close=24_500),
        udiff_row(trad_dt=trad_dt, isin="INF204KB14I2", symbol="NIFTYBEES",
                  instr_type="STK", series="ETF", close=250),
        udiff_row(trad_dt=trad_dt, isin="", symbol="NOISIN",
                  instr_type="STK", series="EQ", close=10),
    ]


class StubCookieJar:
    def __init__(self):
        self._jar = {}

    def clear(self):
        self._jar.clear()

    def set(self, k, v):
        self._jar[k] = v

    def __len__(self):
        return len(self._jar)


class StubResponse:
    def __init__(self, status_code=200, content=b"", url=""):
        self.status_code = status_code
        self.content = content
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status at {self.status_code}")


class StubTransport:
    """Stands in for ``requests.Session``. Records calls, replays queued responses."""

    def __init__(self, responses: dict[str, list] | None = None, default=None):
        self.headers = {}
        self.cookies = StubCookieJar()
        self.responses = responses or {}
        self.default = default
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, dict(headers or {})))
        # Any homepage hit sets a cookie, like the real warm-up.
        if url.startswith("https://www.nseindia.com"):
            self.cookies.set("nsit", f"tok{len(self.calls)}")
            return StubResponse(200, b"<html>ok</html>", url)
        for prefix, queue in self.responses.items():
            if url.startswith(prefix):
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(item, Exception):
                    raise item
                return item
        if self.default is not None:
            return self.default
        return StubResponse(404, b"", url)

    @property
    def archive_calls(self) -> list[str]:
        return [u for u, _ in self.calls if "nsearchives" in u]


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "data")


@pytest.fixture
def no_sleep():
    """Collapse backoff so retry tests stay fast; records the delays asked for."""
    delays: list[float] = []
    return delays, delays.append


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, request):
    """Hard-block outbound sockets for the whole suite.

    A stub injected on the calling thread is invisible to thread-pool workers,
    which is how an earlier version of the Upstox tests silently hit the live
    API and passed for the wrong reason. Failing loudly beats a green test that
    measured the internet.
    """
    if request.node.get_closest_marker("allow_network"):
        return
    import socket

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "network access attempted in a test — inject a stub session instead"
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
