## Goals

* Integrate FastAPI for routing
    - SQLModels for base stories, jobs, comments, users
    - CRUD-style functionality on frontend (only on linked user posts)
* Postgres DB
    - stores raw post/comment/user data
    - vector extensions to allow for nlp/llm ops
* Implement HN Data Worker
    - Celery/Redis Data Broker and Task Store
        * Flower/Prometheus for monitoring?
    - configurable to run various operations
        * Pre-seed a database (`start-date` -> `today`, `top-level comments only`, `no comments`, `only jobs`, `no user data`, etc.)
        * Live updates (`only-posts`, `all-comments`, etc.)
        * Historical data verification (`check job postings from 2021-2022`)
    - passively creates tasks based on internal config to perform operations
        * can take different performance-based criteria (`aggressive`, `lazy`, `max`, etc.) to affect how often to schedule tasks and perform actions
* Complete HNClient
    - full tree-traversal of a story
    - process-locking, safety, and robustness
    - handling failures
* Frontend
    - start simple
        * og hn-style pages
        * comment views of posts
    - analytics
        * dependency graph
        * keywords graph
        * related entries
        * jobs filters
        * post/comment volume over time
