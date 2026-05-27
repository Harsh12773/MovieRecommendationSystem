import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
API_BASE = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="MovieVerse",
    page_icon="🎬",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
<style>

html, body, [class*="css"]  {
    background-color: #0e1117;
    color: white;
}

.stApp {
    background-color: #0e1117;
}

h1,h2,h3,h4,h5,h6,p,span,label {
    color: white !important;
}

.movie-card {
    background-color: #161b22;
    border-radius: 14px;
    padding: 10px;
    margin-bottom: 10px;
    transition: 0.3s;
}

.movie-card:hover {
    transform: scale(1.02);
}

.poster-title {
    text-align: center;
    font-size: 15px;
    margin-top: 8px;
    font-weight: 600;
}

.block-container {
    padding-top: 1rem;
}

</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# API
# =========================================================
def api_get(endpoint, params=None):

    try:

        response = requests.get(
            f"{API_BASE}{endpoint}",
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            return None

        return response.json()

    except:
        return None

# =========================================================
# TITLE
# =========================================================
st.title("🎬 MovieVerse")

st.caption(
    "AI Powered Movie Recommendation System"
)

# =========================================================
# SEARCH BAR
# =========================================================
query = st.text_input(
    "Search Movie"
)

# =========================================================
# SEARCH RESULTS
# =========================================================
if query:

    search_data = api_get(
        "/tmdb/search",
        {"query": query}
    )

    results = search_data.get("results", [])

    st.subheader("Search Results")

    cols = st.columns(5)

    for index, movie in enumerate(results[:15]):

        with cols[index % 5]:

            poster = movie.get("poster_path")

            if poster:

                st.image(
                    f"https://image.tmdb.org/t/p/w500{poster}"
                )

            st.markdown(
                f"""
                <div class="poster-title">
                    {movie.get("title")}
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "View",
                key=movie["id"]
            ):

                st.session_state.movie_id = movie["id"]

# =========================================================
# HOME PAGE
# =========================================================
else:

    st.subheader("🔥 Trending Movies")

    movies = api_get(
        "/home",
        {
            "category": "trending",
            "limit": 20
        }
    )

    cols = st.columns(5)

    for index, movie in enumerate(movies):

        with cols[index % 5]:

            st.markdown(
                '<div class="movie-card">',
                unsafe_allow_html=True
            )

            if movie["poster_url"]:

                st.image(
                    movie["poster_url"]
                )

            st.markdown(
                f"""
                <div class="poster-title">
                    {movie["title"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Open",
                key=f"home_{movie['tmdb_id']}"
            ):

                st.session_state.movie_id = movie["tmdb_id"]

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

# =========================================================
# DETAILS PAGE
# =========================================================
if "movie_id" in st.session_state:

    st.divider()

    movie_id = st.session_state.movie_id

    details = api_get(
        f"/movie/id/{movie_id}"
    )

    if details:

        left, right = st.columns([1, 2])

        with left:

            st.image(details["poster_url"])

        with right:

            st.title(details["title"])

            st.write(
                details["overview"]
            )

            genres = [
                g["name"]
                for g in details["genres"]
            ]

            st.write(
                f"Genres: {', '.join(genres)}"
            )

        st.subheader(
            "🎯 Recommended Movies"
        )

        bundle = api_get(
            "/movie/search",
            {
                "query": details["title"]
            }
        )

        recommendations = bundle.get(
            "tfidf_recommendations",
            []
        )

        cols = st.columns(5)

        for index, item in enumerate(recommendations):

            movie = item.get("tmdb")

            if not movie:
                continue

            with cols[index % 5]:

                if movie["poster_url"]:

                    st.image(
                        movie["poster_url"]
                    )

                st.markdown(
                    f"""
                    <div class="poster-title">
                        {movie["title"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )