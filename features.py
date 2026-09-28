from collections import Counter
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config


META = ['cookie_id', 'cookie_created_at', 'window_start_ts', 'window_end_ts']
EVENT_COLUMNS = [
    'cookie_id',
    'event_ts',
    'eid',
    'event_name',
    'platform',
    'user_agent',
    'item_id',
    'item_category',
    'item_location',
    'seller_type',
    'search_query',
    'search_page',
    'pointer_x',
    'pointer_y',
]
MISSING = '__MISSING__'


def prepare_events(events, meta, deduplicate=True):
    events = events[EVENT_COLUMNS].copy()
    events['cookie_id'] = events.cookie_id.astype('string')
    events['event_ts'] = pd.to_datetime(events.event_ts, utc=True, format='mixed', errors='coerce')
    events = events.merge(meta[META], on='cookie_id', how='inner')
    inside = (events.event_ts >= events.window_start_ts) & (events.event_ts < events.window_end_ts)
    events = events.loc[inside].copy()
    if deduplicate:
        events = events.drop_duplicates(EVENT_COLUMNS).copy()
    for column in ['search_page', 'pointer_x', 'pointer_y']:
        events[column] = pd.to_numeric(events[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    text_columns = [
        'eid',
        'event_name',
        'platform',
        'user_agent',
        'item_id',
        'item_category',
        'item_location',
        'seller_type',
        'search_query',
    ]
    for column in text_columns:
        events[column] = events[column].astype('string').fillna(MISSING)
        events.loc[events[column].str.strip().eq(''), column] = MISSING
    events['search_query'] = events.search_query.str.casefold().str.replace('\\s+', ' ', regex=True).str.strip()
    events.loc[events.search_query.eq(MISSING.casefold()), 'search_query'] = MISSING
    return events.sort_values(['cookie_id', 'event_ts'], kind='stable').reset_index(drop=True)


def get_event_names(train, events):
    observed = events.loc[events.cookie_id.isin(train.cookie_id), 'event_name']
    counts = observed.value_counts()
    return sorted(counts.index, key=lambda name: (-int(counts[name]), str(name)))[:32]


def metadata_features(meta):
    age_end = (meta.window_end_ts - meta.cookie_created_at).dt.total_seconds() / 86400
    return pd.DataFrame(
        {
            'meta_age_end_days': age_end.to_numpy(),
            'meta_created_inside_window': meta.cookie_created_at.ge(meta.window_start_ts).astype(float).to_numpy(),
        },
        index=pd.Index(meta.cookie_id, name='cookie_id'),
    )


def basic_features(meta, raw_events):
    events = raw_events.groupby('cookie_id', sort=False)
    counts = events.size()
    items = raw_events.loc[raw_events.item_id.ne(MISSING)].groupby('cookie_id').item_id.nunique()
    x = pd.DataFrame({'n_events': counts, 'item_nunique': items})
    return x.reindex(meta.cookie_id).fillna(0).astype(float)


def add_stats(out, name, values, quantiles=True):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    keys = ['mean', 'std', 'min', 'max', 'median'] + (['p10', 'p90'] if quantiles else [])
    if a.size == 0:
        out.update({f'{name}_{k}': np.nan for k in keys})
        return
    vals = [a.mean(), a.std() if a.size > 1 else np.nan, a.min(), a.max(), np.median(a)]
    if quantiles:
        vals += list(np.quantile(a, [0.1, 0.9]))
    out.update({f'{name}_{k}': float(v) for k, v in zip(keys, vals)})


def add_distribution(out, name, values):
    vals = [str(v) for v in values if str(v) != MISSING]
    counts = Counter(vals)
    n = len(vals)
    out[f'{name}_n_observed'] = n
    out[f'{name}_nunique'] = len(counts)
    if not n:
        for k in ['unique_share', 'top_share', 'entropy', 'hhi']:
            out[f'{name}_{k}'] = np.nan
    else:
        p = np.array(list(counts.values()), dtype=float) / n
        out[f'{name}_unique_share'] = len(counts) / n
        out[f'{name}_top_share'] = float(p.max())
        out[f'{name}_entropy'] = float(-(p * np.log(p)).sum())
        out[f'{name}_hhi'] = float((p * p).sum())
    return counts


def parse_user_agent(ua):
    if ua == MISSING:
        return (MISSING, MISSING, np.nan)
    u = ua.lower()
    auto = float(
        bool(
            re.search(
                'headless|selenium|puppeteer|playwright|python-requests|'
                'python-urllib|aiohttp|curl/|wget/|scrapy|httpx/|go-http-client',
                u,
            ),
        ),
    )
    browser = 'other'
    for pattern, label in [
        ('edg[eaios]*/', 'edge'),
        ('firefox/|fxios/', 'firefox'),
        ('chrome/|crios/', 'chrome'),
        ('safari/', 'safari'),
    ]:
        if re.search(pattern, u):
            browser = label
            break
    os_family = 'other'
    for pattern, label in [
        ('android', 'android'),
        ('iphone|ipad|ipod', 'ios'),
        ('windows', 'windows'),
        ('macintosh|mac os', 'mac'),
        ('linux', 'linux'),
    ]:
        if re.search(pattern, u):
            os_family = label
            break
    return (browser, os_family, auto)


def activity_features(events, event_names):
    n = len(events)
    out = {'n_events': n, 'has_events': int(n > 0)}
    event = events.event_name.to_numpy(dtype=str)
    evc = add_distribution(out, 'event', event)
    for j, name in enumerate(event_names or []):
        out[f'event_type_{j:02d}_count'] = evc.get(name, 0)
        out[f'event_type_{j:02d}_share'] = evc.get(name, 0) / n if n else np.nan
    known = sum((evc.get(k, 0) for k in event_names or []))
    out['event_other_share'] = (n - known) / n if n else np.nan
    for col, prefix in [
        ('item_id', 'item'),
        ('item_category', 'category'),
        ('item_location', 'location'),
        ('search_query', 'query'),
        ('seller_type', 'seller'),
    ]:
        add_distribution(out, prefix, events[col].to_numpy(dtype=str))
        out[f'{prefix}_missing_share'] = float(events[col].eq(MISSING).mean()) if n else np.nan
    out['events_per_item'] = n / out['item_nunique'] if out['item_nunique'] else np.nan
    t = events.event_ts.astype('int64').to_numpy(dtype=float) / 1000000000.0
    dt = np.diff(t)
    positive = dt[dt > 0]
    add_stats(out, 'gap', positive)
    for threshold in [1, 2, 5, 10, 60, 300, 1800]:
        out[f'gap_le_{threshold}_share'] = float((dt <= threshold).mean()) if len(dt) else np.nan
    out['gap_zero_share'] = float((dt == 0).mean()) if len(dt) else np.nan
    out['gap_cv'] = float(positive.std() / positive.mean()) if len(positive) > 1 else np.nan
    out['gap_burstiness'] = (
        float((positive.std() - positive.mean()) / (positive.std() + positive.mean()))
        if len(positive) > 1 else np.nan
    )
    out['gap_mad_over_median'] = (
        float(np.median(np.abs(positive - np.median(positive))) / np.median(positive))
        if len(positive) else np.nan
    )
    out['gap_lag1_corr'] = (
        float(np.corrcoef(dt[:-1], dt[1:])[0, 1])
        if len(dt) >= 4 and dt[:-1].std() > 0 and dt[1:].std() > 0 else np.nan
    )
    span = t[-1] - t[0] if n else np.nan
    out['activity_span_seconds'] = span
    out['events_per_span_second'] = n / (span + 1) if n else np.nan
    out['timestamp_nunique'] = len(np.unique(t))
    if n:
        start = events.window_start_ts.iloc[0].timestamp()
        end = events.window_end_ts.iloc[0].timestamp()
        relative = t - start
        hours = np.floor(relative / 3600).astype(int)
        hc = np.bincount(hours, minlength=24)[:24].astype(float)
        out['first_event_offset_seconds'] = t[0] - start
        out['last_event_recency_seconds'] = end - t[-1]
        out['active_minutes'] = len(np.unique(np.floor(relative / 60)))
        out['active_hours'] = len(np.unique(hours))
        p = hc[hc > 0] / n
        out['hour_entropy'] = float(-(p * np.log(p)).sum())
        out['hour_top_share'] = float(hc.max() / n)
    else:
        hc = np.zeros(24)
        out.update(
            first_event_offset_seconds=np.nan,
            last_event_recency_seconds=np.nan,
            active_minutes=0,
            active_hours=0,
            hour_entropy=np.nan,
            hour_top_share=np.nan,
        )
    for j in range(24):
        out[f'hour_{j:02d}_share'] = hc[j] / n if n else np.nan
    for seconds in [1, 10, 60, 300]:
        count = np.arange(1, n + 1) - np.searchsorted(t, t - seconds, side='left') if n else np.array([])
        out[f'max_events_in_{seconds}s'] = int(count.max()) if n else 0
    for gap in [300, 1800]:
        bounds = np.r_[0, np.flatnonzero(dt > gap) + 1, n] if n else np.array([0])
        sizes = np.diff(bounds)
        durations = np.array([t[b - 1] - t[a] for a, b in zip(bounds[:-1], bounds[1:])])
        out[f'session_{gap}_count'] = len(sizes)
        add_stats(out, f'session_{gap}_events', sizes, False)
        add_stats(out, f'session_{gap}_duration', durations, False)
    seq = events.loc[~events.event_ts.duplicated(keep=False)]
    seq_ev = seq.event_name.to_numpy(dtype=str)
    pairs = [f'{a}\x1f{b}' for a, b in zip(seq_ev[:-1], seq_ev[1:])]
    add_distribution(out, 'transition', pairs)
    out['sequence_usable_share'] = len(seq) / n if n else np.nan
    out['event_repeat_share'] = float((seq_ev[1:] == seq_ev[:-1]).mean()) if len(seq_ev) > 1 else np.nan
    out['event_repeat_lag2_share'] = float((seq_ev[2:] == seq_ev[:-2]).mean()) if len(seq_ev) > 2 else np.nan
    it = seq.loc[seq.item_id.ne(MISSING), 'item_id'].to_numpy(dtype=str)
    out['item_consecutive_repeat_share'] = float((it[1:] == it[:-1]).mean()) if len(it) > 1 else np.nan
    pages = events.search_page.to_numpy(dtype=float)
    pages = pages[np.isfinite(pages)]
    add_stats(out, 'search_page', pages)
    out['search_page_nunique'] = len(np.unique(pages))
    out['search_page_le1_share'] = float((pages <= 1).mean()) if len(pages) else np.nan
    pg = seq.loc[seq.search_page.notna() & seq.search_query.ne(MISSING)]
    same_query = pg.search_query.to_numpy(dtype=str)[1:] == pg.search_query.to_numpy(dtype=str)[:-1]
    steps = np.diff(pg.search_page.to_numpy(dtype=float))[same_query]
    out['search_page_step1_share'] = float((steps == 1).mean()) if len(steps) else np.nan
    out['search_page_forward_share'] = float((steps > 0).mean()) if len(steps) else np.nan
    queries = events.loc[events.search_query.ne(MISSING), 'search_query'].tolist()
    add_stats(out, 'query_length', [len(q) for q in queries], False)
    add_stats(out, 'query_words', [len(q.split()) for q in queries], False)
    valid_ptr = events.pointer_x.notna() & events.pointer_y.notna()
    out['ptr_observed_share'] = float(valid_ptr.mean()) if n else np.nan
    pts = events.loc[valid_ptr, ['pointer_x', 'pointer_y']].to_numpy(dtype=float)
    out['ptr_nunique'] = len(np.unique(pts, axis=0)) if len(pts) else 0
    out['ptr_unique_share'] = out['ptr_nunique'] / len(pts) if len(pts) else np.nan
    ps = seq.loc[seq.pointer_x.notna() & seq.pointer_y.notna()]
    xy = ps[['pointer_x', 'pointer_y']].to_numpy(dtype=float)
    times = ps.event_ts.astype('int64').to_numpy(dtype=float) / 1000000000.0
    if len(xy) > 1:
        delta = np.diff(xy, axis=0)
        distance = np.linalg.norm(delta, axis=1)
        scale = max(float(np.linalg.norm(np.ptp(xy, axis=0))), 1.0)
        add_stats(out, 'ptr_normalized_step', distance / scale)
        add_stats(out, 'ptr_normalized_speed', distance / scale / np.diff(times))
        out['ptr_stationary_share'] = float((distance == 0).mean())
        out['ptr_axis_aligned_share'] = float(((delta[:, 0] == 0) | (delta[:, 1] == 0)).mean())
        out['ptr_path_efficiency'] = float(np.linalg.norm(xy[-1] - xy[0]) / distance.sum()) if distance.sum() else np.nan
    else:
        add_stats(out, 'ptr_normalized_step', [])
        add_stats(out, 'ptr_normalized_speed', [])
        out.update(
            ptr_stationary_share=np.nan,
            ptr_axis_aligned_share=np.nan,
            ptr_path_efficiency=np.nan,
        )
    return out


def canonical_platform(value):
    s = str(value).strip().casefold()
    if s in ('web', 'desktop'):
        return 'web'
    if s in ('android',):
        return 'android'
    if s in ('ios', 'iphone', 'ipad'):
        return 'ios'
    return MISSING if s == MISSING.casefold() else s


@lru_cache(maxsize=32768)
def ua_family(value):
    browser, os, auto = parse_user_agent(value)
    if re.search('yabrowser/', str(value), flags=re.I):
        browser = 'yandex'
    return (browser, os, auto)


def most_common(a):
    cnt = Counter((v for v in a if v != MISSING))
    return min(cnt, key=lambda x: (-cnt[x], x)) if cnt else MISSING


def divide(a, b):
    return float(a / b) if b else np.nan


def add_gap_stats(out, prefix, a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    for name in ['mean', 'std', 'p25', 'p50', 'p75', 'p95', 'cv', 'mad_ratio', 'log_std']:
        out[prefix + '_' + name] = np.nan
    if not len(a):
        return
    q25, med, q75, q95 = np.quantile(a, [0.25, 0.5, 0.75, 0.95])
    out.update(
        {
            prefix + '_mean': a.mean(),
            prefix + '_std': a.std() if len(a) > 1 else np.nan,
            prefix + '_p25': q25,
            prefix + '_p50': med,
            prefix + '_p75': q75,
            prefix + '_p95': q95,
            prefix + '_cv': divide(a.std(), a.mean()) if len(a) > 1 else np.nan,
            prefix + '_mad_ratio': divide(np.median(np.abs(a - med)), med),
            prefix + '_log_std': float(np.log1p(a).std()) if len(a) > 1 else np.nan,
        },
    )


def add_diversity(out, prefix, a):
    a = [str(x) for x in a if str(x) != MISSING]
    counts = np.asarray(list(Counter(a).values()), dtype=float)
    n = len(a)
    out[prefix + '_nunique'] = len(counts)
    out[prefix + '_unique_share'] = divide(len(counts), n)
    out[prefix + '_top_share'] = divide(counts.max(), n) if n else np.nan
    out[prefix + '_entropy'] = float(-np.sum(counts / n * np.log(counts / n))) if n else np.nan


def behavior_features(events):
    n = len(events)
    out = {}
    ev = events.event_name.to_numpy(dtype=str)
    ns = events.event_ts.astype('int64').to_numpy()
    t = (ns - ns[0]) / 1000000000.0 if n else np.array([], dtype=float)
    dt = np.diff(t)
    pos = dt[dt > 0]
    out['ex_gap_count'] = len(pos)
    add_gap_stats(out, 'ex_gap', pos)
    for seconds in (30, 60, 120, 300, 1800):
        within = pos[pos <= seconds]
        add_gap_stats(out, f'ex_within{seconds}', within)
        out[f'ex_within{seconds}_count'] = len(within)
    for scale in (0.5, 2.0, 5.0):
        out[f"ex_gap_relative_{str(scale).replace('.', '_')}_share"] = (
            float((pos <= scale * np.median(pos)).mean()) if len(pos) else np.nan
        )
    for digits in (0, 1, 2):
        vals = np.round(pos, digits)
        counts = np.asarray(list(Counter(vals).values()))
        out[f'ex_gap_round{digits}_top_share'] = divide(counts.max(), len(pos)) if len(pos) else np.nan
        out[f'ex_gap_round{digits}_unique_share'] = divide(len(counts), len(pos))
    if len(pos):
        p = np.histogram(pos, bins=[0, 1, 2, 5, 10, 30, 60, 120, 300, 1800, np.inf])[0] / len(pos)
        out['ex_gap_hist_entropy'] = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    else:
        out['ex_gap_hist_entropy'] = np.nan
    cnt = Counter(ev)
    for a, b, label in [
        ('item_view', 'search_results_view', 'views_per_search'),
        ('photo_swipe', 'item_view', 'photos_per_view'),
        ('favorite_add', 'item_view', 'favorites_per_view'),
        ('contact_phone_show', 'item_view', 'phones_per_view'),
        ('seller_page_view', 'item_view', 'sellers_per_view'),
        ('contact_message_sent', 'contact_chat_open', 'messages_per_chat'),
    ]:
        out['ex_' + label] = divide(cnt[a], cnt[b])
    contacts = sum((cnt[x] for x in ['contact_phone_show', 'contact_chat_open', 'contact_message_sent']))
    out['ex_contacts_per_view'] = divide(contacts, cnt['item_view'])
    out['ex_intent_per_view'] = divide(contacts + cnt['favorite_add'] + cnt['login'], cnt['item_view'])
    items = events.item_id.to_numpy(dtype=str)
    cats = events.item_category.to_numpy(dtype=str)
    locs = events.item_location.to_numpy(dtype=str)
    qry = events.search_query.to_numpy(dtype=str)
    for name in ('item_view', 'search_results_view'):
        mask = ev == name
        out[f'ex_{name}_count'] = int(mask.sum())
        add_diversity(out, f'ex_{name}_item', items[mask])
        add_diversity(out, f'ex_{name}_category', cats[mask])
        add_diversity(out, f'ex_{name}_location', locs[mask])
        add_diversity(out, f'ex_{name}_query', qry[mask])
        gaps = np.diff(t[mask])
        add_gap_stats(out, f'ex_{name}_gap', gaps[gaps > 0])
    validitem = items != MISSING
    itemcnt = Counter(items[validitem])
    a = np.asarray(list(itemcnt.values()), dtype=float)
    out['ex_item_singleton_share'] = float((a == 1).mean()) if len(a) else np.nan
    out['ex_item_repeated_share'] = float((a >= 3).mean()) if len(a) else np.nan
    viewitems = set(items[(ev == 'item_view') & validitem])
    searchcnt = cnt['search_results_view']
    out['ex_view_unique_per_search'] = divide(len(viewitems), searchcnt)
    for name, mask in [
        ('contact', np.isin(ev, ['contact_phone_show', 'contact_chat_open', 'contact_message_sent'])),
        ('favorite', ev == 'favorite_add'),
        ('photo', ev == 'photo_swipe'),
    ]:
        actitems = set(items[mask & validitem])
        out[f'ex_view_items_with_{name}_share'] = divide(len(viewitems & actitems), len(viewitems))
        out[f'ex_{name}_without_view_share'] = divide(len(actitems - viewitems), len(actitems))
    unique = ~events.event_ts.duplicated(keep=False).to_numpy()
    pair = unique[:-1] & unique[1:] & (dt > 0) & (dt <= 1800)
    usable = int(pair.sum())
    out['ex_pair_count'] = usable
    out['ex_pair_usable_share'] = divide(usable, max(n - 1, 0))
    for a_name in ('search_results_view', 'item_view', 'photo_swipe'):
        left = ev[:-1] == a_name
        denom = int((pair & left).sum())
        for b_name in ('search_results_view', 'item_view', 'photo_swipe', 'favorite_add', 'contact_phone_show'):
            mask = pair & left & (ev[1:] == b_name)
            out[f'ex_next_{a_name}__{b_name}'] = divide(mask.sum(), denom)
        add_gap_stats(out, f'ex_after_{a_name}_gap', dt[pair & left])
    for values, name in [(items, 'item'), (cats, 'category'), (locs, 'location'), (qry, 'query')]:
        valid = pair & (values[:-1] != MISSING) & (values[1:] != MISSING)
        out[f'ex_adjacent_{name}_change'] = divide(((values[1:] != values[:-1]) & valid).sum(), valid.sum())
    for gap in (300, 1800):
        starts = np.r_[0, np.flatnonzero(dt > gap) + 1] if n else np.array([], dtype=int)
        ends = np.r_[starts[1:], n] if n else np.array([], dtype=int)
        size = ends - starts
        out[f'ex_session{gap}_singleton_share'] = float((size == 1).mean()) if len(size) else np.nan
        rates = []
        unique_rates = []
        concentration = []
        for a, b in zip(starts, ends):
            span = t[b - 1] - t[a]
            if b - a >= 3:
                rates.append((b - a - 1) / (span + 1))
            its = items[a:b]
            obs = its[its != MISSING]
            if len(obs):
                unique_rates.append(len(set(obs)) / len(obs))
            cs = cats[a:b]
            cs = cs[cs != MISSING]
            if len(cs):
                concentration.append(max(Counter(cs).values()) / len(cs))
        add_gap_stats(out, f'ex_session{gap}_rate', rates)
        out[f'ex_session{gap}_item_unique_mean'] = float(np.mean(unique_rates)) if unique_rates else np.nan
        out[f'ex_session{gap}_category_top_mean'] = float(np.mean(concentration)) if concentration else np.nan
    plats = [canonical_platform(x) for x in events.platform.to_numpy(dtype=str)]
    out['norm_platform_mode'] = most_common(plats)
    out['norm_platform_nunique'] = len(set(plats) - {MISSING})
    for name in ('web', 'android', 'ios'):
        out['norm_platform_' + name + '_share'] = divide(plats.count(name), n)
    ua = [ua_family(x) for x in events.user_agent.to_numpy(dtype=str)]
    out['norm_browser_mode'] = most_common([x[0] for x in ua])
    out['norm_os_mode'] = most_common([x[1] for x in ua])
    out['norm_browser_nunique'] = len({x[0] for x in ua} - {MISSING})
    autos = np.array([x[2] for x in ua])
    autos = autos[np.isfinite(autos)]
    out['norm_ua_automation_share'] = float(autos.mean()) if len(autos) else np.nan
    return out


def positive_ratio(a, b):
    return float(a / b) if b > 0 else np.nan


def add_session_stats(out, prefix, values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a):
        vals = [a.mean(), *np.quantile(a, [0.1, 0.5, 0.9])]
    else:
        vals = [np.nan] * 4
    out.update(zip([prefix + s for s in ('_mean', '_q10', '_median', '_q90')], vals))


def add_repetition_stats(out, prefix, values):
    values = [v for v in values if v != MISSING]
    counts = np.array(list(Counter(values).values()), dtype=float)
    n = len(values)
    out[prefix + '_collision'] = positive_ratio(np.sum(counts * (counts - 1)), n * (n - 1))
    out[prefix + '_repeat_mass'] = positive_ratio(np.sum(np.maximum(counts - 1, 0)), n)
    out[prefix + '_singleton_mass'] = positive_ratio(np.sum(counts == 1), n)
    out[prefix + '_entropy_corrected'] = (
        float(-np.sum(counts / n * np.log(counts / n))) + (len(counts) - 1) / (2 * n)
        if n else np.nan
    )


def session_features(events):
    out = {}
    n = len(events)
    times = events.event_ts.astype('int64').to_numpy()
    t = (times - times[0]) / 1000000000.0 if n else np.empty(0)
    dt = np.diff(t)
    positive = dt[dt > 0]
    ev = events.event_name.to_numpy(dtype=str)
    values = {name: events[col].to_numpy(dtype=str) for name, col in [
        ('item', 'item_id'),
        ('category', 'item_category'),
        ('location', 'item_location'),
        ('query', 'search_query'),
    ]}
    unique = ~events.event_ts.duplicated(keep=False).to_numpy()
    adjacent = unique[:-1] & unique[1:] & (dt > 0)
    for cutoff in (120, 300, 900, 1800):
        key = f'v3_s{cutoff}'
        bounds = np.r_[0, np.flatnonzero(dt > cutoff) + 1, n] if n else np.array([0])
        sizes = np.diff(bounds)
        gap_cvs, log_stds, lvs, residuals, medians, ranges = ([], [], [], [], [], [])
        regular_events = 0
        supported_events = 0
        largest_cv = np.nan
        largest_med = np.nan
        for a, b in zip(bounds[:-1], bounds[1:]):
            gaps = dt[a:b - 1]
            gaps = gaps[gaps > 0]
            if len(gaps) < 3:
                continue
            mean = gaps.mean()
            cv = gaps.std() / mean
            gap_cvs.append(cv)
            log_stds.append(np.log1p(gaps).std())
            medians.append(np.median(gaps))
            ranges.append((np.max(gaps) - np.min(gaps)) / mean)
            z = gaps[:-1] + gaps[1:]
            lv = float(np.mean(3 * ((gaps[1:] - gaps[:-1]) / z) ** 2))
            lvs.append(lv)
            seq = np.r_[0.0, np.cumsum(gaps)]
            idx = np.arange(len(seq), dtype=float)
            slope = np.dot(idx - idx.mean(), seq - seq.mean()) / np.sum((idx - idx.mean()) ** 2)
            residuals.append(np.std(seq - (seq.mean() + slope * (idx - idx.mean()))) / mean)
            supported_events += b - a
            regular_events += (b - a) * (cv < 0.25)
            if b - a == sizes.max():
                largest_cv = cv
                largest_med = np.median(gaps)
        for name, arr in [
            ('cv', gap_cvs),
            ('log_std', log_stds),
            ('local_variation', lvs),
            ('fit_residual', residuals),
            ('median_gap', medians),
        ]:
            add_session_stats(out, key + '_' + name, arr)
        out[key + '_supported_events_share'] = positive_ratio(supported_events, n)
        out[key + '_regular_events_share'] = positive_ratio(regular_events, supported_events)
        out[key + '_largest_cv'] = largest_cv
        out[key + '_largest_median_gap'] = largest_med
        local = (dt[:-1] > 0) & (dt[1:] > 0) & (dt[:-1] <= cutoff) & (dt[1:] <= cutoff)
        a, b = (dt[:-1][local], dt[1:][local])
        out[key + '_local_variation'] = float(np.mean(3 * ((a - b) / (a + b)) ** 2)) if len(a) else np.nan
        out[key + '_adjacent_log_diff'] = float(np.mean(np.abs(np.log1p(a) - np.log1p(b)))) if len(a) else np.nan
        out[key + '_median_rate_cv'] = positive_ratio(np.std(medians), np.mean(medians)) if len(medians) > 1 else np.nan
    if len(positive):
        q = np.quantile(positive, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        scale = q[2]
        for label, val in zip(['q10', 'q25', 'q50', 'q75', 'q90', 'q95'], q):
            out['v3_gap_' + label + '_over_median'] = val / scale
        out['v3_gap_trimmed_cv'] = positive_ratio(np.std(positive[positive <= q[4]]), np.mean(positive[positive <= q[4]]))
        ordered = np.sort(positive)
        out['v3_gap_maximum_fraction'] = positive_ratio(ordered[-1], ordered.sum())
        out['v3_gap_gini'] = float(
            2 * np.dot(np.arange(1, len(ordered) + 1), ordered)
            / (len(ordered) * ordered.sum()) - (len(ordered) + 1) / len(ordered),
        )
    else:
        for label in ['q10', 'q25', 'q50', 'q75', 'q90', 'q95']:
            out['v3_gap_' + label + '_over_median'] = np.nan
        out.update(v3_gap_trimmed_cv=np.nan, v3_gap_maximum_fraction=np.nan, v3_gap_gini=np.nan)
    for name, a in values.items():
        add_repetition_stats(out, 'v3_' + name, a)
        for cutoff in (60, 300, 1800):
            usable = adjacent & (dt <= cutoff) & (a[:-1] != MISSING) & (a[1:] != MISSING)
            out[f'v3_{name}_same_within{cutoff}'] = positive_ratio(np.sum((a[:-1] == a[1:]) & usable), usable.sum())
    for same in (True, False):
        a = values['category']
        mask = adjacent & (dt <= 1800) & (a[:-1] != MISSING) & (a[1:] != MISSING) & ((a[:-1] == a[1:]) == same)
        add_session_stats(out, 'v3_category_' + ('same' if same else 'change') + '_gap', dt[mask])
    pts = events[['pointer_x', 'pointer_y']].to_numpy(dtype=float)
    observed = np.isfinite(pts).all(axis=1)
    xy = pts[observed]
    if len(xy) >= 3:
        span = np.ptp(xy, axis=0)
        out['v3_ptr_axis_ratio'] = positive_ratio(min(span), max(span))
        out['v3_ptr_unique_x_share'] = len(np.unique(xy[:, 0])) / len(xy)
        out['v3_ptr_unique_y_share'] = len(np.unique(xy[:, 1])) / len(xy)
        out['v3_ptr_xy_corr'] = float(np.corrcoef(xy.T)[0, 1]) if (np.std(xy, axis=0) > 0).all() else np.nan
    else:
        out.update(
            v3_ptr_axis_ratio=np.nan,
            v3_ptr_unique_x_share=np.nan,
            v3_ptr_unique_y_share=np.nan,
            v3_ptr_xy_corr=np.nan,
        )
    valid = adjacent & (dt <= 300) & observed[:-1] & observed[1:]
    delta = np.diff(pts, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    scale = float(np.linalg.norm(np.ptp(xy, axis=0))) if len(xy) else 0.0
    for suffix, vals in [
        ('step', lengths[valid] / scale if scale else []),
        ('speed', lengths[valid] / dt[valid] / scale if scale else []),
    ]:
        add_session_stats(out, 'v3_ptr_local_' + suffix, vals)
    triple = valid[:-1] & valid[1:] & (lengths[:-1] > 0) & (lengths[1:] > 0)
    cos = np.sum(delta[:-1][triple] * delta[1:][triple], axis=1) / (lengths[:-1][triple] * lengths[1:][triple])
    add_session_stats(out, 'v3_ptr_turn_cos', cos)
    plats = [canonical_platform(x) for x in events.platform.to_numpy(dtype=str)]
    ua = [ua_family(x) for x in events.user_agent.to_numpy(dtype=str)]
    out['v3_ua_os_nunique'] = len({x[1] for x in ua} - {MISSING})
    add_repetition_stats(out, 'v3_ua_profile', [x[0] + '|' + x[1] for x in ua])
    out['v3_ua_platform_mismatch'] = np.mean([p in ('android', 'ios') and p != u[1] for p, u in zip(plats, ua)]) if n else np.nan
    pages = events.search_page.to_numpy(dtype=float)
    qry = values['query']
    mask = (qry != MISSING) & np.isfinite(pages)
    coverage = []
    unique_page = []
    for q in sorted(set(qry[mask])):
        z = pages[mask & (qry == q)]
        coverage.append(len(np.unique(z)) / (np.max(z) - np.min(z) + 1))
        unique_page.append(len(np.unique(z)) / len(z))
    add_session_stats(out, 'v3_query_page_coverage', coverage)
    add_session_stats(out, 'v3_query_page_unique', unique_page)
    return out


def feature_batch(events, event_names):
    rows = {}
    for cookie_id, history in events.groupby('cookie_id', sort=False):
        row = activity_features(history, event_names)
        row.update(behavior_features(history))
        row.update(session_features(history))
        rows[str(cookie_id)] = row
    return pd.DataFrame.from_dict(rows, orient='index')


def make_features(meta, events, event_names, n_jobs=4):
    events = events.loc[events.cookie_id.isin(meta.cookie_id)]
    events = events.sort_values(['cookie_id', 'event_ts'], kind='stable')
    groups = np.array_split(meta.cookie_id.to_numpy(), min(len(meta), n_jobs * 8))
    chunks = (events.loc[events.cookie_id.isin(ids)].copy() for ids in groups)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        batches = Parallel(n_jobs=n_jobs, pre_dispatch=n_jobs * 2)(
            (delayed(feature_batch)(chunk, event_names) for chunk in chunks),
        )
    empty = feature_batch(events.iloc[:0], event_names)
    default = activity_features(events.iloc[:0], event_names)
    default.update(behavior_features(events.iloc[:0]))
    default.update(session_features(events.iloc[:0]))
    ids = pd.Index(meta.cookie_id, name='cookie_id')
    rows = pd.concat([batch for batch in batches if len(batch)]) if len(events) else empty
    absent = ids.difference(rows.index)
    rows = rows.reindex(ids, columns=list(default))
    if len(absent):
        rows.loc[absent] = pd.DataFrame([default] * len(absent), index=absent)
    for column, value in default.items():
        if isinstance(value, str):
            rows[column] = rows[column].fillna(MISSING).astype(str)
        else:
            rows[column] = rows[column].astype(float).replace([np.inf, -np.inf], np.nan)
    return pd.concat([metadata_features(meta), rows], axis=1)
