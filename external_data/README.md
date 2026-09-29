# External movie metadata

Metadata for all 20 distinct test titles absent from `movies.csv` was retrieved on 2026-09-29. The per-title matched TMDB ID and status are recorded in `acquisition_audit.csv`; all 20 title-format variants matched successfully.

Source and endpoints: [TMDB API documentation](https://developer.themoviedb.org/docs/authentication-application), API v3 `/3/search/movie` and `/3/movie/{movie_id}?append_to_response=credits,release_dates`. Attribution and use terms: [TMDB API terms](https://www.themoviedb.org/api-terms-of-use). Results use normalized title matching after removing format markers (2D/3D/IMAX/DUBBING), while retaining each test title as `original_title` for the catalog merge.

The output is `missing_movies_metadata.csv`, with the catalog fields `original_title`, `age_rating`, `genre`, `producer`, `director`, `writer`, and `casts`, plus overview, popularity, runtime, original language, release date/year, and TMDB ID. `age_rating` comes from Indonesian (`ID`) release certification; the 20 matched records had a value at retrieval time.

Acquisition can be repeated with `python scripts/build_pipeline.py --fetch-metadata`. The script reads `TMDB_API_RAT` as a bearer token and/or `TMDB_API_KEY` from environment variables or the project `.env`; it does not log or save credential values. `.env` is ignored by Git.
