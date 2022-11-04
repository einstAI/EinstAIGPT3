from AML.Synthetic.EINSTEINAI4DB.deep_rl.parse_graph import SchemaGraph
from AML.Synthetic.EINSTEINAI4DB.deep_rl.parse_graph import gen_job_light_imdb_schema
from AML.Synthetic.EINSTEINAI4DB.deep_rl.parse_graph import SchemaGraph


def gen_job_light_imdb_schema():
    schema_graph = SchemaGraph()
    schema_graph.add_table("actors", ["id", "name", "age"])
    schema_graph.add_table("directors", ["id", "name", "age"])
    schema_graph.add_table("genres", ["id", "name"])
    schema_graph.add_table("movies", ["id", "title", "year", "score"])
    schema_graph.add_table("movie_genres", ["movie_id", "genre_id"])
    schema_graph.add_table("movie_actors", ["movie_id", "actor_id"])
    schema_graph.add_table("movie_directors", ["movie_id", "director_id"])

    schema_graph.add_relationship("movies", "id", "movie_genres", "movie_id")
    schema_graph.add_relationship("movies", "id", "movie_actors", "movie_id")
    schema_graph.add_relationship("movies", "id", "movie_directors", "movie_id")
    schema_graph.add_relationship("movie_genres", "genre_id", "genres", "id")
    schema_graph.add_relationship("movie_actors", "actor_id", "actors", "id")
    schema_graph.add_relationship("movie_directors", "director_id", "directors", "id")

    return schema_graph