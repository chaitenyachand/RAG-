from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import os, shutil

def create_bm25_index(data, index_dir="bm25_index"):
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.mkdir(index_dir)
    schema = Schema(content=TEXT(stored=True))
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    for item in data:
        writer.add_document(content=item["support"].replace("\n", " "))
    writer.commit()

def retrieve_bm25(query, index_dir="bm25_index", top_k=3):
    from whoosh import index
    ix = index.open_dir(index_dir)
    parser = QueryParser("content", ix.schema)
    q = parser.parse(query)
    results = []
    with ix.searcher() as searcher:
        hits = searcher.search(q, limit=top_k)
        for hit in hits:
            results.append(hit["content"])
    return results
