import os
import pickle
from typing import List, Optional, Dict, Any, Tuple

import httpx
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing inside .env file"
    )

# =========================================================
# CONSTANTS
# =========================================================
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Movie Recommendation API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================================================
# GLOBAL VARIABLES
# =========================================================
df = None
indices = None
tfidf_matrix = None

TITLE_TO_INDEX = {}

# =========================================================
# MODELS
# =========================================================
class MovieCard(BaseModel):

    tmdb_id: int

    title: str

    poster_url: Optional[str] = None

    release_date: Optional[str] = None

    vote_average: Optional[float] = None


class MovieDetails(BaseModel):

    tmdb_id: int

    title: str

    overview: Optional[str] = None

    poster_url: Optional[str] = None

    backdrop_url: Optional[str] = None

    release_date: Optional[str] = None

    genres: List[dict] = []


class TFIDFMovie(BaseModel):

    title: str

    score: float

    tmdb: Optional[MovieCard] = None


class BundleResponse(BaseModel):

    query: str

    movie_details: MovieDetails

    tfidf_recommendations: List[TFIDFMovie]

    genre_recommendations: List[MovieCard]

# =========================================================
# HELPERS
# =========================================================
def normalize_title(title: str):

    return str(title).strip().lower()


def image_url(path: Optional[str]):

    if not path:
        return None

    return f"{TMDB_IMAGE_BASE}{path}"


async def tmdb_get(endpoint: str, params: Dict[str, Any]):

    params["api_key"] = TMDB_API_KEY

    try:

        async with httpx.AsyncClient(timeout=20) as client:

            response = await client.get(
                f"{TMDB_BASE_URL}{endpoint}",
                params=params
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"TMDB Request Failed: {e}"
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    return response.json()

# =========================================================
# LOAD PICKLES
# =========================================================
@app.on_event("startup")
def startup():

    global df
    global indices
    global tfidf_matrix
    global TITLE_TO_INDEX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    for title, idx in indices.items():

        TITLE_TO_INDEX[
            normalize_title(title)
        ] = int(idx)

# =========================================================
# HOME
# =========================================================
@app.get("/home", response_model=List[MovieCard])
async def home(

    category: str = Query("popular"),

    limit: int = Query(24)

):

    if category == "trending":

        data = await tmdb_get(
            "/trending/movie/day",
            {"language": "en-US"}
        )

    else:

        data = await tmdb_get(
            f"/movie/{category}",
            {
                "language": "en-US",
                "page": 1
            }
        )

    results = data.get("results", [])

    output = []

    for movie in results[:limit]:

        output.append(
            MovieCard(
                tmdb_id=movie["id"],
                title=movie.get("title"),
                poster_url=image_url(
                    movie.get("poster_path")
                ),
                release_date=movie.get("release_date"),
                vote_average=movie.get("vote_average")
            )
        )

    return output

# =========================================================
# SEARCH MOVIES
# =========================================================
@app.get("/tmdb/search")
async def search_movies(

    query: str

):

    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "language": "en-US",
            "include_adult": False,
            "page": 1
        }
    )

# =========================================================
# MOVIE DETAILS
# =========================================================
@app.get("/movie/id/{movie_id}")
async def movie_details(movie_id: int):

    data = await tmdb_get(
        f"/movie/{movie_id}",
        {
            "language": "en-US"
        }
    )

    return MovieDetails(
        tmdb_id=data["id"],
        title=data.get("title"),
        overview=data.get("overview"),
        poster_url=image_url(
            data.get("poster_path")
        ),
        backdrop_url=image_url(
            data.get("backdrop_path")
        ),
        release_date=data.get("release_date"),
        genres=data.get("genres", [])
    )

# =========================================================
# TFIDF RECOMMENDATION
# =========================================================
def get_recommendations(

    title: str,

    top_n: int = 10

):

    key = normalize_title(title)

    if key not in TITLE_TO_INDEX:

        return []

    idx = TITLE_TO_INDEX[key]

    similarity = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    scores = list(enumerate(similarity))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:top_n + 1]

    movies = []

    for i, score in scores:

        movies.append(
            (
                str(df.iloc[i]["title"]),
                float(score)
            )
        )

    return movies


async def tmdb_card(title: str):

    data = await tmdb_get(
        "/search/movie",
        {
            "query": title,
            "language": "en-US",
            "page": 1
        }
    )

    results = data.get("results", [])

    if not results:
        return None

    movie = results[0]

    return MovieCard(
        tmdb_id=movie["id"],
        title=movie.get("title"),
        poster_url=image_url(
            movie.get("poster_path")
        ),
        release_date=movie.get("release_date"),
        vote_average=movie.get("vote_average")
    )

# =========================================================
# TFIDF ROUTE
# =========================================================
@app.get("/recommend/tfidf")
async def recommend_tfidf(

    title: str,

    top_n: int = 10

):

    recs = get_recommendations(title, top_n)

    return [
        {
            "title": t,
            "score": s
        }
        for t, s in recs
    ]

# =========================================================
# GENRE RECOMMENDATION
# =========================================================
@app.get("/recommend/genre")
async def recommend_genre(

    tmdb_id: int,

    limit: int = 12

):

    details = await tmdb_get(
        f"/movie/{tmdb_id}",
        {"language": "en-US"}
    )

    genres = details.get("genres", [])

    if not genres:
        return []

    genre_id = genres[0]["id"]

    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "language": "en-US",
            "page": 1
        }
    )

    results = discover.get("results", [])

    output = []

    for movie in results[:limit]:

        if movie["id"] == tmdb_id:
            continue

        output.append(
            MovieCard(
                tmdb_id=movie["id"],
                title=movie.get("title"),
                poster_url=image_url(
                    movie.get("poster_path")
                ),
                release_date=movie.get("release_date"),
                vote_average=movie.get("vote_average")
            )
        )

    return output

# =========================================================
# FULL SEARCH
# =========================================================
@app.get("/movie/search")
async def full_search(

    query: str,

    tfidf_top_n: int = 10,

    genre_limit: int = 10

):

    data = await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "language": "en-US",
            "page": 1
        }
    )

    results = data.get("results", [])

    if not results:

        raise HTTPException(
            status_code=404,
            detail="Movie not found"
        )

    movie = results[0]

    movie_id = movie["id"]

    details = await movie_details(movie_id)

    # TFIDF
    tfidf_items = []

    recommendations = get_recommendations(
        details.title,
        tfidf_top_n
    )

    for title, score in recommendations:

        card = await tmdb_card(title)

        tfidf_items.append(
            TFIDFMovie(
                title=title,
                score=score,
                tmdb=card
            )
        )

    # Genre
    genre_movies = await recommend_genre(
        movie_id,
        genre_limit
    )

    return BundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_movies
    )