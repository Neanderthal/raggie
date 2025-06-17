Excellent. Here's your notebook converted to Markdown format:

---

# PGVector

> An implementation of LangChain vectorstore abstraction using `postgres` as the backend and utilizing the `pgvector` extension.

The code lives in an integration package called: [langchain\_postgres](https://github.com/langchain-ai/langchain-postgres/).

## Status

This code has been ported over from `langchain_community` into a dedicated package called `langchain-postgres`. The following changes have been made:

* langchain\_postgres works only with psycopg3. Please update your connnecion strings from `postgresql+psycopg2://...` to `postgresql+psycopg://langchain:langchain@...` (yes, it's the driver name is `psycopg` not `psycopg3`, but it'll use `psycopg3`.
* The schema of the embedding store and collection have been changed to make add\_documents work correctly with user specified ids.
* One has to pass an explicit connection object now.

Currently, there is **no mechanism** that supports easy data migration on schema changes. So any schema changes in the vectorstore will require the user to recreate the tables and re-add the documents. If this is a concern, please use a different vectorstore. If not, this implementation should be fine for your use case.
`

## Instantiation

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

```python
from langchain_postgres import PGVector

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!  
collection_name = "my_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
```

## Manage vector store

### Add items to vector store

Note that adding documents by ID will over-write any existing documents that match that ID.

```python
from langchain_core.documents import Document

docs = [
    Document(page_content='there are cats in the pond', metadata={"id": 1, "location": "pond", "topic": "animals"}),
    Document(page_content='ducks are also found in the pond', metadata={"id": 2, "location": "pond", "topic": "animals"}),
    Document(page_content='fresh apples are available at the market', metadata={"id": 3, "location": "market", "topic": "food"}),
    Document(page_content='the market also sells fresh oranges', metadata={"id": 4, "location": "market", "topic": "food"}),
    Document(page_content='the new art exhibit is fascinating', metadata={"id": 5, "location": "museum", "topic": "art"}),
    Document(page_content='a sculpture exhibit is also at the museum', metadata={"id": 6, "location": "museum", "topic": "art"}),
    Document(page_content='a new coffee shop opened on Main Street', metadata={"id": 7, "location": "Main Street", "topic": "food"}),
    Document(page_content='the book club meets at the library', metadata={"id": 8, "location": "library", "topic": "reading"}),
    Document(page_content='the library hosts a weekly story time for kids', metadata={"id": 9, "location": "library", "topic": "reading"}),
    Document(page_content='a cooking class for beginners is offered at the community center', metadata={"id": 10, "location": "community center", "topic": "classes"}),
]

vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
```

### Delete items from vector store

```python
vector_store.delete(ids=["3"])
```

## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

### Filtering Support

The vectorstore supports a set of filters that can be applied against the metadata fields of the documents.

| Operator  | Meaning/Category             |
| --------- | ---------------------------- |
| \$eq      | Equality (==)                |
| \$ne      | Inequality (!=)              |
| \$lt      | Less than (<)                |
| \$lte     | Less than or equal (<=)      |
| \$gt      | Greater than (>)             |
| \$gte     | Greater than or equal (>=)   |
| \$in      | Special Cased (in)           |
| \$nin     | Special Cased (not in)       |
| \$between | Special Cased (between)      |
| \$like    | Text (like)                  |
| \$ilike   | Text (case-insensitive like) |
| \$and     | Logical (and)                |
| \$or      | Logical (or)                 |

### Query directly

```python
results = vector_store.similarity_search(
    "kitty", k=10, filter={"id": {"$in": [1, 5, 2, 9]}}
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
```

If you provide a dict with multiple fields, but no operators, the top level will be interpreted as a logical **AND** filter:

```python
vector_store.similarity_search(
    "ducks",
    k=10,
    filter={"id": {"$in": [1, 5, 2, 9]}, "location": {"$in": ["pond", "market"]}},
)
```

```python
vector_store.similarity_search(
    "ducks",
    k=10,
    filter={
        "$and": [
            {"id": {"$in": [1, 5, 2, 9]}},
            {"location": {"$in": ["pond", "market"]}},
        ]
    },
)
```

If you want to execute a similarity search and receive the corresponding scores you can run:

```python
results = vector_store.similarity_search_with_score(query='cats', k=1)
for doc, score in results:
    print(f"* [SIM={score:.f}] {doc.page_content} [{doc.metadata}]")
```

For a full list of the different searches you can execute on a `PGVector` vector store, please refer to the [API reference](https://python.langchain.com/api_reference/postgres/vectorstores/langchain_postgres.vectorstores.PGVector.html).

### Query by turning into retriever

```python
retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k": 1})
retriever.invoke("kitty")
```

## Usage for retrieval-augmented generation

For guides on how to...

---

✅ All done!
If you'd like, I can export this directly to a `.md` file. Just let me know!

