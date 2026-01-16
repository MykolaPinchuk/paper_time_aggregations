from __future__ import annotations

import bisect
from array import array
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


@dataclass(frozen=True)
class TimeAggConfig:
    entities: tuple[str, ...] = ("device_ip", "device_id", "app_id", "site_id")
    windows: tuple[int, ...] = (1, 6, 24)
    alpha: float = 1.0
    beta: float = 10.0
    gap_hours: int = 0
    bucket_edges: tuple[int, ...] = ()
    add_calendar_windows: bool = False
    event_windows: tuple[int, ...] = ()


class _EventWindowState:
    __slots__ = ("max_imps", "blocks", "imps", "clicks")

    def __init__(self, max_imps: int) -> None:
        if max_imps <= 0:
            raise ValueError(f"event window size must be >0, got {max_imps}")
        self.max_imps = int(max_imps)
        self.blocks: deque[tuple[float, float]] = deque()
        self.imps: float = 0.0
        self.clicks: float = 0.0

    def add_block(self, imps: float, clicks: float) -> None:
        if imps <= 0:
            return
        self.blocks.append((imps, clicks))
        self.imps += imps
        self.clicks += clicks
        self.drop_excess()

    def drop_excess(self) -> None:
        while self.blocks and self.imps > self.max_imps:
            blk_imps, blk_clicks = self.blocks[0]
            excess = self.imps - self.max_imps
            if blk_imps <= excess + 1e-9:
                self.blocks.popleft()
                self.imps -= blk_imps
                self.clicks -= blk_clicks
                continue
            keep_imps = blk_imps - excess
            keep_clicks = blk_clicks * (keep_imps / blk_imps)
            self.blocks[0] = (keep_imps, keep_clicks)
            self.imps -= excess
            self.clicks -= (blk_clicks - keep_clicks)
            break

    def snapshot(self) -> tuple[float, float]:
        self.drop_excess()
        return self.imps, self.clicks


@dataclass
class _EntityState:
    max_window: int
    hours: array
    imps: array
    clicks: array
    cum_imps: array
    cum_clicks: array
    start_idx: int = 0
    last_seen_hour: int | None = None
    day_idx: int | None = None
    day_imps: float = 0.0
    day_clicks: float = 0.0
    prev_day_idx: int | None = None
    prev_day_imps: float = 0.0
    prev_day_clicks: float = 0.0
    event_windows: dict[int, _EventWindowState] | None = None

    def advance(self, hour_idx: int) -> None:
        cutoff = int(hour_idx) - int(self.max_window)
        # We never delete list elements (small max length ~= #hours with activity),
        # but moving start_idx keeps bisect work bounded.
        while self.start_idx < len(self.hours) and self.hours[self.start_idx] < cutoff:
            self.start_idx += 1

    def _ensure_day(self, current_day: int) -> None:
        if self.day_idx is None:
            self.day_idx = current_day
            return
        if current_day == self.day_idx:
            return
        if current_day == self.day_idx + 1:
            self.prev_day_idx = self.day_idx
            self.prev_day_imps = self.day_imps
            self.prev_day_clicks = self.day_clicks
        else:
            self.prev_day_idx = None
            self.prev_day_imps = 0.0
            self.prev_day_clicks = 0.0
        self.day_idx = current_day
        self.day_imps = 0.0
        self.day_clicks = 0.0

    def _trailing_sum(self, hour_idx: int, window: int) -> tuple[float, float]:
        if not self.hours:
            return 0.0, 0.0
        cutoff = int(hour_idx) - int(window)
        i = bisect.bisect_left(self.hours, cutoff, lo=self.start_idx)
        if i >= len(self.hours):
            return 0.0, 0.0
        total_imps = self.cum_imps[-1]
        total_clicks = self.cum_clicks[-1]
        if i == 0:
            return total_imps, total_clicks
        return total_imps - self.cum_imps[i - 1], total_clicks - self.cum_clicks[i - 1]

    def snapshot(
        self, hour_idx: int, day_idx: int, needed_trailing: tuple[int, ...]
    ) -> tuple[dict[int, tuple[float, float]], float, tuple[float, float], tuple[float, float], dict[int, tuple[float, float]]]:
        self.advance(hour_idx)
        self._ensure_day(day_idx)

        trailing: dict[int, tuple[float, float]] = {}
        for window in needed_trailing:
            trailing[int(window)] = self._trailing_sum(hour_idx, int(window))
        recency = np.nan if self.last_seen_hour is None else float(hour_idx - self.last_seen_hour)
        day = (self.day_imps, self.day_clicks)
        yday = (self.prev_day_imps, self.prev_day_clicks)
        event: dict[int, tuple[float, float]] = {}
        if self.event_windows:
            for n, state in self.event_windows.items():
                event[n] = state.snapshot()
        return trailing, recency, day, yday, event

    def add_counts(self, hour_idx: int, day_idx: int, imps: float, clicks: float) -> None:
        self._ensure_day(day_idx)
        self.hours.append(int(hour_idx))
        self.imps.append(float(imps))
        self.clicks.append(float(clicks))
        if self.cum_imps:
            self.cum_imps.append(float(self.cum_imps[-1]) + float(imps))
            self.cum_clicks.append(float(self.cum_clicks[-1]) + float(clicks))
        else:
            self.cum_imps.append(float(imps))
            self.cum_clicks.append(float(clicks))
        self.day_imps += imps
        self.day_clicks += clicks
        if self.event_windows:
            for state in self.event_windows.values():
                state.add_block(imps, clicks)
        self.last_seen_hour = hour_idx


def _hour_index(series: pd.Series) -> np.ndarray:
    base = series.min()
    if pd.isna(base):
        raise ValueError("Cannot compute hour index on empty series.")
    diffs = series.values.astype("datetime64[ns]") - base.to_datetime64()
    return (diffs / np.timedelta64(1, "h")).astype(np.int64)


def _day_index(series: pd.Series) -> np.ndarray:
    day = series.dt.floor("D")
    base = day.min()
    if pd.isna(base):
        raise ValueError("Cannot compute day index on empty series.")
    diffs = day.values.astype("datetime64[ns]") - base.to_datetime64()
    return (diffs / np.timedelta64(1, "D")).astype(np.int64)


def _feature_names(cfg: TimeAggConfig) -> dict[str, list[str]]:
    names: dict[str, list[str]] = defaultdict(list)
    gap = int(cfg.gap_hours)

    for entity in cfg.entities:
        names["recency"].append(f"{entity}__recency_hours")

        for w in cfg.windows:
            if gap > 0:
                names["trailing"].append(f"{entity}__imps_log_w{w}h_g{gap}h")
                names["trailing"].append(f"{entity}__ctr_w{w}h_g{gap}h")
            else:
                names["trailing"].append(f"{entity}__imps_log_w{w}h")
                names["trailing"].append(f"{entity}__ctr_w{w}h")

        if cfg.bucket_edges:
            edges = sorted({int(x) for x in cfg.bucket_edges})
            prev = 0
            for end in edges:
                names["bucket"].append(f"{entity}__imps_log_b{prev}to{end}h")
                names["bucket"].append(f"{entity}__ctr_b{prev}to{end}h")
                prev = end

        if cfg.add_calendar_windows:
            names["calendar"].append(f"{entity}__imps_log_today")
            names["calendar"].append(f"{entity}__ctr_today")
            names["calendar"].append(f"{entity}__imps_log_yday")
            names["calendar"].append(f"{entity}__ctr_yday")

        for n in cfg.event_windows:
            n = int(n)
            names["event"].append(f"{entity}__imps_log_last{n}imps")
            names["event"].append(f"{entity}__ctr_last{n}imps")

    return names


def _needed_trailing_windows(cfg: TimeAggConfig) -> set[int]:
    needed: set[int] = set()
    gap = int(cfg.gap_hours)

    for w in cfg.windows:
        w = int(w)
        if w <= 0:
            raise ValueError(f"Window sizes must be >0, got {w}")
        if gap > 0:
            needed.add(w + gap)
        else:
            needed.add(w)

    if gap > 0:
        needed.add(gap)

    for edge in cfg.bucket_edges:
        edge = int(edge)
        if edge <= 0:
            raise ValueError(f"bucket_edges must be >0, got {edge}")
        needed.add(edge)

    return needed


def add_time_agg_features(
    df: pd.DataFrame, cfg: TimeAggConfig | None = None, inplace: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    if cfg is None:
        cfg = TimeAggConfig()
    if not cfg.entities:
        return df, []
    for col in cfg.entities:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    if "hour_dt" not in df.columns:
        raise ValueError("DataFrame must contain 'hour_dt'.")

    if not df["hour_dt"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by hour_dt in ascending order.")
    hour_idx = _hour_index(df["hour_dt"])
    day_idx = _day_index(df["hour_dt"])

    if any(w <= 0 for w in cfg.windows):
        raise ValueError(f"All cfg.windows must be >0, got {cfg.windows}")
    if cfg.gap_hours < 0:
        raise ValueError(f"gap_hours must be >=0, got {cfg.gap_hours}")
    if cfg.bucket_edges and sorted(cfg.bucket_edges) != list(cfg.bucket_edges):
        raise ValueError("bucket_edges must be sorted ascending (e.g. 1 6 24 168).")
    if len(set(cfg.bucket_edges)) != len(cfg.bucket_edges):
        raise ValueError(f"bucket_edges must not contain duplicates, got {cfg.bucket_edges}")

    n = len(df)
    click_arr = df["click"].to_numpy(dtype=np.float32, copy=False)
    entity_arrays: dict[str, np.ndarray] = {}
    for col in cfg.entities:
        s = df[col]
        if ptypes.is_categorical_dtype(s):
            entity_arrays[col] = s.cat.codes.to_numpy(dtype=np.int32, copy=False)
        elif ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s):
            codes, _ = pd.factorize(s, sort=False)
            entity_arrays[col] = codes.astype(np.int32, copy=False)
        elif ptypes.is_integer_dtype(s):
            entity_arrays[col] = s.to_numpy(dtype=np.int32, copy=False)
        else:
            codes, _ = pd.factorize(s, sort=False)
            entity_arrays[col] = codes.astype(np.int32, copy=False)

    feature_arrays: dict[str, np.ndarray] = {}
    default_ctr = float(cfg.alpha / (cfg.alpha + cfg.beta))
    feature_names = _feature_names(cfg)
    for group in feature_names.values():
        for name in group:
            if name.endswith("__recency_hours"):
                feature_arrays[name] = np.full(n, np.nan, dtype=np.float32)
            elif "__ctr_" in name:
                feature_arrays[name] = np.full(n, default_ctr, dtype=np.float32)
            else:
                feature_arrays[name] = np.zeros(n, dtype=np.float32)

    states: dict[str, dict[Any, _EntityState]] = {col: {} for col in cfg.entities}
    pending: dict[str, dict[Any, list[float]]] = {col: {} for col in cfg.entities}

    needed_trailing = _needed_trailing_windows(cfg)
    max_window = int(max(needed_trailing)) if needed_trailing else 1
    needed_trailing_sorted = tuple(sorted(int(w) for w in needed_trailing))

    def _new_state() -> _EntityState:
        event_windows = {int(n): _EventWindowState(int(n)) for n in cfg.event_windows} if cfg.event_windows else None
        return _EntityState(
            max_window=max_window,
            hours=array("i"),
            imps=array("f"),
            clicks=array("f"),
            cum_imps=array("f"),
            cum_clicks=array("f"),
            event_windows=event_windows,
        )

    def flush(hour: int, day: int) -> None:
        for col in cfg.entities:
            col_pending = pending[col]
            if not col_pending:
                continue
            col_states = states[col]
            for key, (imps, clicks) in col_pending.items():
                state = col_states.get(key)
                if state is None:
                    state = _new_state()
                    col_states[key] = state
                state.add_counts(hour, day, imps, clicks)
            pending[col] = {}

    current_hour = None
    current_day = None
    for idx in range(n):
        hour = int(hour_idx[idx])
        day = int(day_idx[idx])
        if current_hour is None:
            current_hour = hour
            current_day = day
        elif hour != current_hour:
            flush(current_hour, int(current_day))
            current_hour = hour
            current_day = day

        click = float(click_arr[idx])

        for col in cfg.entities:
            key = entity_arrays[col][idx]
            col_states = states[col]
            state = col_states.get(key)
            if state is None:
                state = _new_state()
                col_states[key] = state

            trailing, recency, day_stats, yday_stats, event_stats = state.snapshot(hour, day, needed_trailing_sorted)
            feature_arrays[f"{col}__recency_hours"][idx] = np.float32(recency)

            gap = int(cfg.gap_hours)
            for w in cfg.windows:
                w = int(w)
                if gap > 0:
                    big_imps, big_clicks = trailing[w + gap]
                    gap_imps, gap_clicks = trailing[gap]
                    imps = max(0.0, big_imps - gap_imps)
                    clicks = max(0.0, big_clicks - gap_clicks)
                    imps_name = f"{col}__imps_log_w{w}h_g{gap}h"
                    ctr_name = f"{col}__ctr_w{w}h_g{gap}h"
                else:
                    imps, clicks = trailing[w]
                    imps_name = f"{col}__imps_log_w{w}h"
                    ctr_name = f"{col}__ctr_w{w}h"
                feature_arrays[imps_name][idx] = np.float32(np.log1p(imps))
                feature_arrays[ctr_name][idx] = np.float32((clicks + cfg.alpha) / (imps + cfg.alpha + cfg.beta))

            if cfg.bucket_edges:
                edges = list(cfg.bucket_edges)
                prev = 0
                prev_imps, prev_clicks = 0.0, 0.0
                for end in edges:
                    end = int(end)
                    end_imps, end_clicks = trailing[end]
                    imps = max(0.0, end_imps - prev_imps)
                    clicks = max(0.0, end_clicks - prev_clicks)
                    feature_arrays[f"{col}__imps_log_b{prev}to{end}h"][idx] = np.float32(np.log1p(imps))
                    feature_arrays[f"{col}__ctr_b{prev}to{end}h"][idx] = np.float32(
                        (clicks + cfg.alpha) / (imps + cfg.alpha + cfg.beta)
                    )
                    prev = end
                    prev_imps, prev_clicks = end_imps, end_clicks

            if cfg.add_calendar_windows:
                today_imps, today_clicks = day_stats
                yday_imps, yday_clicks = yday_stats
                feature_arrays[f"{col}__imps_log_today"][idx] = np.float32(np.log1p(today_imps))
                feature_arrays[f"{col}__ctr_today"][idx] = np.float32(
                    (today_clicks + cfg.alpha) / (today_imps + cfg.alpha + cfg.beta)
                )
                feature_arrays[f"{col}__imps_log_yday"][idx] = np.float32(np.log1p(yday_imps))
                feature_arrays[f"{col}__ctr_yday"][idx] = np.float32(
                    (yday_clicks + cfg.alpha) / (yday_imps + cfg.alpha + cfg.beta)
                )

            if cfg.event_windows:
                for n in cfg.event_windows:
                    n = int(n)
                    imps, clicks = event_stats.get(n, (0.0, 0.0))
                    feature_arrays[f"{col}__imps_log_last{n}imps"][idx] = np.float32(np.log1p(imps))
                    feature_arrays[f"{col}__ctr_last{n}imps"][idx] = np.float32(
                        (clicks + cfg.alpha) / (imps + cfg.alpha + cfg.beta)
                    )

            pend = pending[col].get(key)
            if pend is None:
                pending[col][key] = [1.0, click]
            else:
                pend[0] += 1.0
                pend[1] += click

    if current_hour is not None and current_day is not None:
        flush(int(current_hour), int(current_day))

    new_cols_df = pd.DataFrame(feature_arrays)
    new_cols = list(new_cols_df.columns)
    output = pd.concat([df if inplace else df.copy(), new_cols_df], axis=1)
    return output, new_cols
