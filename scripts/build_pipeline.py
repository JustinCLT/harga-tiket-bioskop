"""Feature, temporal validation, baseline and L1 boosting pipeline.

Usage: python scripts/build_pipeline.py [--fetch-metadata]
Fetching requires TMDB_READ_ACCESS_TOKEN (or TMDB_API_KEY).
"""
from __future__ import annotations
import argparse, ast, json, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
OUT = ROOT / "data"
EXT = ROOT / "external_data"
SEED = 2026


def load(name):
    return pd.read_csv(DATA / f"{name}.csv")


def fetch_metadata():
    import requests

    def safe_get(url, **kwargs):
        try:
            response = requests.get(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # Never print a requests exception: its URL may contain the API key.
            raise RuntimeError(
                f"TMDB HTTP request failed (status={status or 'unavailable'}); credential value omitted"
            ) from None

    token = os.getenv("TMDB_API_RAT")
    key = os.getenv("TMDB_API_KEY")
    envfile = ROOT / ".env"
    if (not token or not key) and envfile.exists():
        for line in envfile.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, value = line.split("=", 1)
            name = name.strip().upper()
            value = value.strip().strip("\"'")
            if name == "TMDB_API_RAT" and not token:
                token = value
            if name == "TMDB_API_KEY" and not key:
                key = value
    if not token and not key:
        raise RuntimeError(
            "Set TMDB_API_RAT or TMDB_API_KEY in environment or project .env"
        )
    test = load("test")
    catalog = load("movies")
    missing = sorted(
        set(test.movie_title.dropna()) - set(catalog.original_title.dropna())
    )
    rows = []
    audit = []
    for title in missing:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        query = re.sub(r"\b(2D|3D|IMAX|DUBBING)\b", "", title, flags=re.I).strip()
        params = {"query": query, "language": "en-US"}
        if key:
            params["api_key"] = key
        resp = safe_get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            headers=headers,
            timeout=30,
        )
        results = resp.json().get("results", [])
        norm = lambda x: re.sub(
            r"[^a-z0-9]",
            "",
            re.sub(r"\b(2D|3D|IMAX|DUBBING)\b", "", str(x), flags=re.I).lower(),
        )
        exact = [x for x in results if norm(x.get("title")) == norm(title)]
        if not exact:
            audit.append((title, "not_found_or_title_mismatch"))
            continue
        best = max(exact, key=lambda x: x.get("popularity", 0))
        mid = best["id"]
        detail_params = {
            "append_to_response": "credits,release_dates",
            "language": "en-US",
        }
        if key:
            detail_params["api_key"] = key
        detail = safe_get(
            f"https://api.themoviedb.org/3/movie/{mid}",
            params=detail_params,
            headers=headers,
            timeout=30,
        )
        m = detail.json()
        credits = m.get("credits", {})
        crew = credits.get("crew", [])
        cast = credits.get("cast", [])
        directors = [x["name"] for x in crew if x.get("job") == "Director"]
        producers = [x["name"] for x in crew if x.get("job") == "Producer"]
        writers = [x["name"] for x in crew if x.get("department") == "Writing"]
        certs = []
        for country in m.get("release_dates", {}).get("results", []):
            if country.get("iso_3166_1") == "ID":
                certs += [
                    x.get("certification")
                    for x in country.get("release_dates", [])
                    if x.get("certification")
                ]
        rows.append(
            {
                "original_title": title,
                "age_rating": certs[0] if certs else np.nan,
                "genre": "|".join(x["name"] for x in m.get("genres", [])),
                "producer": "|".join(producers),
                "director": "|".join(directors),
                "writer": "|".join(writers),
                "casts": "|".join(x["name"] for x in cast[:10]),
                "overview": m.get("overview"),
                "popularity": m.get("popularity"),
                "runtime": m.get("runtime"),
                "original_language": m.get("original_language"),
                "release_year": str(m.get("release_date", ""))[:4],
                "released": m.get("release_date"),
                "imdb_id": m.get("imdb_id"),
                "imdb_rating": np.nan,
                "tmdb_id": mid,
            }
        )
        audit.append((title, "matched:" + str(mid)))
        time.sleep(0.25)
    EXT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(EXT / "missing_movies_metadata.csv", index=False)
    (EXT / "README.md").write_text(
        """# External movie metadata\n\nSource: TMDB API v3. Endpoints: `/3/search/movie` and `/3/movie/{movie_id}?append_to_response=credits,release_dates`. Retrieval date: 2026-09-29. API authentication and attribution/terms: https://developer.themoviedb.org/docs/authentication-application and https://www.themoviedb.org/api-terms-of-use. Rows are accepted only on normalized title match after removing 2D/3D/IMAX/DUBBING format markers.\n\n`age_rating` uses Indonesian (`ID`) release certification when available and remains blank when TMDB has no certification. Results and unmatched titles are recorded in `acquisition_audit.csv`. Titles attempted are test titles absent from `movies.csv`; see the audit for completed titles.\n\nSet `TMDB_API_RAT` (API Read Access Token) or `TMDB_API_KEY` in the environment or project `.env` before running `python scripts/build_pipeline.py --fetch-metadata`. Credentials are read locally and never written to project outputs.\n""",
        encoding="utf-8",
    )
    pd.DataFrame(audit, columns=["test_title", "result"]).to_csv(
        EXT / "acquisition_audit.csv", index=False
    )


def history_stats(hist, n_days=3):
    # Load the exact reusable implementation from main.ipynb, avoiding a second
    # definition that could drift from the competition's D1-D3 logic.
    nb = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
    fn = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            continue
        fn = next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "compute_history_stats"
            ),
            None,
        )
        if fn is not None:
            break
    if fn is None:
        raise RuntimeError("compute_history_stats() not found in main.ipynb")
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    scope = {"pd": pd, "np": np}
    exec(compile(mod, "main.ipynb", "exec"), scope)
    return scope["compute_history_stats"](hist, n_days=n_days)


def mase(y, pred, scale):
    return float(
        np.mean(
            np.abs(np.asarray(y) - np.asarray(pred)) / np.maximum(1, np.asarray(scale))
        )
    )


def load_movie_metadata():
    catalog = load("movies")
    extfile = EXT / "missing_movies_metadata.csv"
    if not extfile.exists():
        return catalog
    external = pd.read_csv(extfile)
    if external.empty:
        return catalog
    combined = catalog.merge(
        external, on="original_title", how="outer", suffixes=("_catalog", "_external")
    )
    for col in ["age_rating", "genre", "producer", "director", "writer", "casts"]:
        a = combined.get(f"{col}_catalog")
        b = combined.get(f"{col}_external")
        if a is not None and b is not None:
            combined[col] = a.combine_first(b)
        elif a is not None:
            combined[col] = a
        elif b is not None:
            combined[col] = b
    return combined


def make_features():
    train = load("train")
    hist = load("test_history")
    test = load("test")
    movies = load_movie_metadata()
    holidays = load("holidays")
    prices = load("ticket_prices")
    for d, col in [
        (train, "date_show"),
        (hist, "date_show"),
        (test, "date_show"),
        (holidays, "date"),
    ]:
        d[col] = pd.to_datetime(d[col])
    # Historical cluster profile and capacity, computed from train only.
    tr = train.copy()
    tr["capacity_est"] = np.where(
        (tr.total_show > 0) & (tr.occupation_rate > 0),
        tr.total_ticket / (tr.total_show * tr.occupation_rate / 100),
        np.nan,
    )
    clusters = tr.groupby("cinema_ids", as_index=False).agg(
        cluster_mean_ticket=("total_ticket", "mean"),
        cluster_mean_occupation=("occupation_rate", "mean"),
        cluster_mean_show=("total_show", "mean"),
        capacity_per_show=("capacity_est", "median"),
    )
    hstats = history_stats(hist).rename(columns={"mean_ticket": "scale_raw"})
    feat = test.merge(hstats, on=["movie_title", "cinema_ids"], how="left").merge(
        clusters, on="cinema_ids", how="left"
    )
    feat["scale"] = feat.scale_raw.clip(lower=1)
    feat = feat.merge(
        movies[["original_title", "genre", "age_rating"]].drop_duplicates(
            "original_title"
        ),
        left_on="movie_title",
        right_on="original_title",
        how="left",
    )
    # Create calendar features and nearest holiday distance; holiday dates not listed are normal days.
    cal = holidays.rename(columns={"date": "date_show"})[
        ["date_show", "day_tipe", "holiday_tipe", "holiday_name"]
    ]
    feat = feat.merge(cal, on="date_show", how="left")
    feat["day_tipe"] = feat.day_tipe.fillna(
        feat.date_show.dt.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday")
    )
    feat["holiday_tipe"] = feat.holiday_tipe.fillna("normal")
    hol_dates = (
        holidays.loc[holidays.holiday_tipe.eq("holiday"), "date"]
        .dropna()
        .sort_values()
        .values.astype("datetime64[ns]")
    )
    dates = feat.date_show.values.astype("datetime64[ns]")
    if len(hol_dates):
        delta = (
            (dates[:, None] - hol_dates[None, :]).astype("timedelta64[D]").astype(int)
        )
        nearest = np.abs(delta).argmin(axis=1)
        feat["days_to_holiday"] = delta[
            np.arange(len(delta)), nearest
        ]  # negative = before (H-), positive = after (H+)
    else:
        feat["days_to_holiday"] = 999
    feat["days_since_d1"] = (
        feat.groupby(["movie_title", "cinema_ids"]).date_show.transform("rank") + 3
    )
    # Attach price by city and calendar bucket.
    daymap = {"weekday": "Weekday", "friday": "Friday", "weekend": "Weekend"}
    feat["price_day"] = feat.day_tipe.map(daymap)
    feat = feat.merge(prices, on=["city_name", "price_day"], how="left")
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-metadata", action="store_true")
    args = ap.parse_args()
    if args.fetch_metadata:
        fetch_metadata()
    OUT.mkdir(exist_ok=True)
    f = make_features()
    # parquet if engine installed; retain CSV fallback for portability.
    try:
        f.to_parquet(OUT / "feature_table.parquet", index=False)
    except ImportError:
        f.to_csv(OUT / "feature_table.csv", index=False)
    print(
        f"Feature rows: {len(f):,}; test titles missing genre/rating: {f.loc[f.genre.isna(),'movie_title'].nunique():,}/{f.loc[f.age_rating.isna(),'movie_title'].nunique():,}"
    )
    history_floor = history_stats(load("test_history"))
    history_floor = history_floor[history_floor.mean_ticket.eq(1)][
        ["movie_title", "cinema_ids"]
    ]
    target_pairs = f[["movie_title", "cinema_ids"]].drop_duplicates()
    target_floor = history_floor.merge(target_pairs, on=["movie_title", "cinema_ids"])
    print(
        f"History pairs at scale=1: {len(history_floor)}; overlapping forecast test pairs: {len(target_floor)} (test labels unavailable)"
    )
    # Rolling temporal forecast: each validation pair uses its final 7 train
    # observations as D4-D10 and the immediately preceding three as D1-D3.
    tr = load("train")
    tr["date_show"] = pd.to_datetime(tr.date_show)
    metadata = load_movie_metadata()[
        ["original_title", "genre", "age_rating"]
    ].drop_duplicates("original_title")
    tr = tr.merge(
        metadata, left_on="movie_title", right_on="original_title", how="left"
    )
    tr = tr.sort_values(["movie_title", "cinema_ids", "date_show"])
    grp = tr.groupby(["movie_title", "cinema_ids"], sort=False)
    tr["row_in_pair"] = grp.cumcount()
    pair_sizes = (
        tr.groupby(["movie_title", "cinema_ids"]).row_in_pair.transform("max") + 1
    )
    tr["val_start"] = pair_sizes - 7
    valid = tr[(tr.row_in_pair >= tr.val_start) & (tr.val_start >= 3)].copy()
    history = tr[tr.row_in_pair < tr.val_start].copy()
    history["tail_rank"] = history.groupby(
        ["movie_title", "cinema_ids"]
    ).row_in_pair.transform(lambda x: x.max() - x)
    prior = history[history.tail_rank < 3]
    stats = history_stats(prior).rename(columns={"mean_ticket": "scale_raw"})
    valid = valid.merge(
        stats[["movie_title", "cinema_ids", "scale_raw"]],
        on=["movie_title", "cinema_ids"],
        how="left",
    )
    valid["scale"] = valid.scale_raw.clip(lower=1)
    valid["target_norm"] = valid.total_ticket / valid.scale
    valid["day_tipe"] = valid.date_show.dt.dayofweek.map(
        lambda d: "weekend" if d >= 5 else ("friday" if d == 4 else "weekday")
    )
    daymap = {"weekday": "Weekday", "friday": "Friday", "weekend": "Weekend"}
    multip = {"weekday": 1.0, "friday": 1.2, "weekend": 1.46}
    dayrank = valid.row_in_pair - valid.val_start + 1
    decay = np.power(0.9, np.maximum(0, dayrank - 1))
    baseline = valid.scale.to_numpy() * valid.day_tipe.map(multip).to_numpy() * decay
    print(
        "Temporal holdout rows:",
        len(valid),
        "baseline MASE:",
        mase(valid.total_ticket, baseline, valid.scale),
    )
    low = valid[valid.scale.eq(1)]
    print(
        "scale=1 rows:",
        len(low),
        "MASE:",
        mase(low.total_ticket, low.total_ticket.mean() if len(low) else [], low.scale)
        if len(low)
        else "n/a",
    )
    # Train on earlier observations only. Per-pair preceding three tickets define
    # each sample's scale; validation rows stay outside model fitting.
    try:
        # Avoid joblib probing Windows physical cores through deprecated WMIC.
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
        from lightgbm import LGBMRegressor

        prices = load("ticket_prices")
        holidays = load("holidays")
        holidays["date"] = pd.to_datetime(holidays.date)
        cats = [
            "cinema_ids",
            "city_name",
            "genre",
            "age_rating",
            "day_tipe",
            "holiday_tipe",
            "price_day",
        ]
        # Attach train-only cluster features to the historic training examples.
        train_rows = tr[tr.row_in_pair < tr.val_start].copy()
        # Recompute cluster aggregates from the training window only, excluding
        # each pair's held-out D4-D10 observations to prevent feature leakage.
        train_rows["capacity_est"] = np.where(
            (train_rows.total_show > 0) & (train_rows.occupation_rate > 0),
            train_rows.total_ticket
            / (train_rows.total_show * train_rows.occupation_rate / 100),
            np.nan,
        )
        temporal_clusters = train_rows.groupby("cinema_ids", as_index=False).agg(
            cluster_mean_ticket=("total_ticket", "mean"),
            cluster_mean_occupation=("occupation_rate", "mean"),
            cluster_mean_show=("total_show", "mean"),
            capacity_per_show=("capacity_est", "median"),
        )
        train_rows["past_mean"] = train_rows.groupby(
            ["movie_title", "cinema_ids"]
        ).total_ticket.transform(lambda s: s.shift().rolling(3, min_periods=3).mean())
        train_rows = train_rows.dropna(subset=["past_mean"])
        train_rows = train_rows.merge(temporal_clusters, on="cinema_ids", how="left")
        train_rows["target_norm"] = train_rows.total_ticket / train_rows.past_mean.clip(
            lower=1
        )
        train_rows["scale"] = train_rows.past_mean.clip(lower=1)
        train_rows["day_tipe"] = train_rows.date_show.dt.dayofweek.map(
            lambda d: "weekend" if d >= 5 else ("friday" if d == 4 else "weekday")
        )
        train_rows = train_rows.merge(
            holidays.rename(columns={"date": "date_show"})[
                ["date_show", "holiday_tipe"]
            ],
            on="date_show",
            how="left",
        )
        train_rows["holiday_tipe"] = train_rows.holiday_tipe.fillna("normal")
        train_rows["price_day"] = train_rows.day_tipe.map(daymap)
        train_rows = train_rows.merge(prices, on=["city_name", "price_day"], how="left")
        valid = valid.merge(temporal_clusters, on="cinema_ids", how="left")
        # Holiday and price features for the held-out dates.
        valid = valid.merge(
            holidays.rename(columns={"date": "date_show"})[
                ["date_show", "holiday_tipe"]
            ],
            on="date_show",
            how="left",
            suffixes=("", "_cal"),
        )
        valid["holiday_tipe"] = valid["holiday_tipe"].fillna("normal")
        valid["price_day"] = valid.day_tipe.map(daymap)
        valid = valid.merge(
            prices, on=["city_name", "price_day"], how="left", suffixes=("", "_price")
        )
        modelcols = cats + [
            "scale",
            "cluster_mean_ticket",
            "cluster_mean_occupation",
            "cluster_mean_show",
            "capacity_per_show",
        ]
        modelcols = [c for c in modelcols if c in valid]
        X = valid[modelcols].copy()
        X_train = train_rows[modelcols].copy()
        for c in cats:
            if c in X:
                levels = pd.Index(
                    pd.concat([X[c], X_train[c]]).fillna("Unknown").astype(str).unique()
                )
                X[c] = pd.Categorical(
                    X[c].fillna("Unknown").astype(str), categories=levels
                )
                X_train[c] = pd.Categorical(
                    X_train[c].fillna("Unknown").astype(str), categories=levels
                )
        model = LGBMRegressor(
            objective="mae", random_state=SEED, n_estimators=400, verbosity=-1
        )
        model.fit(
            X_train,
            train_rows.target_norm,
            categorical_feature=[c for c in cats if c in X],
        )
        pred = np.maximum(0, model.predict(X)) * valid.scale
        print("Temporal validation MASE:", mase(valid.total_ticket, pred, valid.scale))
    except ImportError:
        print("LightGBM unavailable; install requirements.txt to run model stage")


if __name__ == "__main__":
    main()
