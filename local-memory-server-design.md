# Local Memory Server - Design Document

## Overview

A lightweight, local memory server that provides persistent conversation memory for LLM agents. This server eliminates the dependency on external services like Letta while providing similar core functionality.

### Goals

1. **Local-first**: Runs entirely on your machine with no external service dependencies
2. **Simple distribution**: Uses UV with inline script dependencies for easy setup
3. **Pluggable embeddings**: Supports local models, Ollama, or OpenAI embeddings
4. **Minimal but complete**: Captures conversations, retrieves relevant context, manages memory blocks
5. **Drop-in integration**: Provides a context manager that wraps existing OpenAI client code

### Non-Goals (for initial implementation)

- Full Letta API compatibility
- Agent self-management of memory (no `memory_replace` tool calls)
- Automatic compaction/summarization
- Multi-user or authentication support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  with learning(agent="my_agent"):                         │  │
│  │      response = client.chat.completions.create(...)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Interceptor Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Patch OpenAI    │  │ Before Request: │  │ After Response: │  │
│  │ client.chat     │  │ Inject context  │  │ Capture convo   │  │
│  │ .completions    │  │ from memory     │  │ to memory       │  │
│  │ .create()       │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Server (FastAPI)                      │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Embedding       │  │ Search          │  │ Storage         │  │
│  │ Service         │  │ Service         │  │ Service         |  │
│  │ (pluggable)     │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SQLite Database                              │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐    │
│  │ agents      │  │ messages    │  │ memory_blocks         │    │
│  │             │  │ + embeddings│  │                       │    │
│  └─────────────┘  └─────────────┘  └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Descriptions

**User Code**

This is the application code written by the developer. The key insight is that existing OpenAI client code requires minimal changes—just wrap it in a `learning()` context manager. The context manager takes an `agent` parameter that serves as a namespace, isolating memory between different agents. Inside the context, all calls to the OpenAI chat completions API are transparently intercepted.

**Interceptor Layer**

The interceptor is the core of the client library. It uses Python's dynamic nature to monkey-patch the OpenAI client's `create()` method while the context manager is active. The interceptor performs three functions:

1. *Patching*: On context entry, it replaces the real `create()` method with a wrapped version. On context exit, it restores the original method.

2. *Before Request*: When the wrapped `create()` is called, the interceptor first queries the memory server for relevant context (memory blocks and semantically similar past messages). It formats this context and injects it into the messages list as a system message before forwarding the request to OpenAI.

3. *After Response*: Once OpenAI returns a response, the interceptor extracts the user's message and the assistant's reply, then stores both in the memory server for future retrieval. The original response is returned to the caller unchanged.

**Memory Server**

The memory server is a local HTTP service built with FastAPI. It exposes a REST API that the interceptor calls. Internally, it coordinates three services:

- *Embedding Service*: Converts text into vector embeddings for semantic search. This is pluggable—it can use a local sentence-transformers model, call out to a local Ollama server, or use OpenAI's embedding API. The choice is made at server startup via configuration.

- *Search Service*: Given a query, retrieves the most semantically similar past messages. It embeds the query, computes cosine similarity against stored message embeddings, and returns the top matches.

- *Storage Service*: Handles all database operations—creating agents, storing messages, managing memory blocks. It abstracts the SQLite details from the rest of the server.

**SQLite Database**

All persistent state lives in a single SQLite database file. Three tables store the data:

- *agents*: Each named agent is a row. The agent name serves as the namespace key—messages and memory blocks are associated with an agent ID, ensuring complete isolation between agents.

- *messages*: Every conversation turn (user message or assistant response) is stored here, along with its vector embedding as a binary blob. The embeddings enable semantic search.

- *memory_blocks*: Structured, labeled text blocks that are always included in context (not retrieved via search). These are useful for persistent information like user preferences or agent persona that should be present in every interaction.

### Data Flow Example

To illustrate how these layers interact, consider this sequence when a user asks "What's my favorite color?":

1. User code calls `client.chat.completions.create()` inside a `learning()` context
2. Interceptor catches the call and extracts the user's question
3. Interceptor calls `POST /context/my_agent` on the memory server with the query
4. Memory server embeds the query and searches for similar past messages
5. Memory server returns memory blocks + relevant messages (e.g., a past message where the user said "I love blue")
6. Interceptor formats this context and injects it as a system message
7. Interceptor forwards the augmented request to OpenAI
8. OpenAI responds with an answer that references the user's preference for blue
9. Interceptor stores both the user message and assistant response via `POST /messages`
10. Interceptor returns the response to the user code

---

## File Structure

```
local-memory-server/
├── server.py              # FastAPI server (main entry point)
├── client.py              # Client library with learning() context manager
├── embeddings.py          # Pluggable embedding backends
├── database.py            # SQLite database operations
├── config.py              # Configuration management
├── models.py              # Pydantic models for API
└── README.md              # Usage documentation
```

---

## UV Inline Dependencies

All Python files should include PEP 723 inline script metadata so they can be run directly with `uv run`. 

For example, `server.py` should declare its dependencies at the top of the file so that running `uv run server.py` automatically creates an isolated environment and installs what's needed.

Dependencies by file:

| File | Dependencies |
|------|--------------|
| `server.py` | fastapi, uvicorn, numpy, pydantic, sentence-transformers (for local embedding) |
| `client.py` | httpx, openai |
| `embeddings.py` | numpy, sentence-transformers, httpx (for Ollama), openai (optional) |
| `database.py` | numpy (for serializing embeddings) |
| `config.py` | pydantic |
| `models.py` | pydantic |

Note: When using the Ollama or OpenAI embedding backends, `sentence-transformers` is not required. Consider structuring the server to lazily import backends so users only need dependencies for their chosen backend.

---

## Database Schema

Use SQLite for storage. The database file should default to `~/.local-memory/memory.db`.

### Tables

**agents** - Namespaces for memory isolation

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | TEXT | PRIMARY KEY | UUID |
| name | TEXT | UNIQUE NOT NULL | User-provided name like "my_agent" |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |
| metadata | TEXT | | JSON blob for extensibility |

**messages** - Conversation history with embeddings

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | TEXT | PRIMARY KEY | UUID |
| agent_id | TEXT | NOT NULL, FK to agents.id | |
| role | TEXT | NOT NULL | "user", "assistant", or "system" |
| content | TEXT | NOT NULL | The message text |
| embedding | BLOB | | numpy array serialized with np.save() to BytesIO |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |
| metadata | TEXT | | JSON blob (model used, token count, etc.) |

**memory_blocks** - Structured always-in-context memory

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | TEXT | PRIMARY KEY | UUID |
| agent_id | TEXT | NOT NULL, FK to agents.id | |
| label | TEXT | NOT NULL | "human", "persona", "project", etc. |
| value | TEXT | NOT NULL | The block content |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |

Additional constraint on memory_blocks: UNIQUE(agent_id, label) - one block per label per agent.

### Indexes

- `idx_messages_agent_created` on messages(agent_id, created_at DESC)
- `idx_memory_blocks_agent` on memory_blocks(agent_id)

### Embedding Serialization

Store numpy arrays as BLOBs using `np.save()` to a `BytesIO` buffer, then read the bytes. Deserialize with `np.load()` from a `BytesIO` wrapper around the bytes.

---

## Configuration

Configuration should be loadable from environment variables with sensible defaults.

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| host | MEMORY_SERVER_HOST | 127.0.0.1 | Server bind address |
| port | MEMORY_SERVER_PORT | 8283 | Server port (matches Letta default) |
| db_path | MEMORY_SERVER_DB_PATH | ~/.local-memory/memory.db | SQLite database path |
| embedding_backend | MEMORY_EMBEDDING_BACKEND | local | "local", "ollama", or "openai" |
| embedding_model | MEMORY_EMBEDDING_MODEL | (backend default) | Model name override |
| ollama_base_url | OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| default_search_limit | MEMORY_DEFAULT_SEARCH_LIMIT | 5 | Default results for search |
| max_search_limit | MEMORY_MAX_SEARCH_LIMIT | 20 | Maximum allowed search results |
| max_context_messages | MEMORY_MAX_CONTEXT_MESSAGES | 10 | Max messages to inject as context |

---

## Embedding Service

The embedding service should be pluggable with a common interface. Implement as an abstract base class with three concrete implementations.

### Interface

All backends must implement:
- `embed(texts: list[str]) -> np.ndarray` - Returns array of shape (len(texts), dimension)
- `dimension: int` property - Returns the embedding dimension

### Backends

**LocalEmbedding**
- Uses sentence-transformers library
- Default model: `all-MiniLM-L6-v2` (384 dimensions, ~80MB)
- Runs entirely locally, no network calls
- Lazy-load the model on first use to avoid slow startup if not needed

**OllamaEmbedding**
- Calls local Ollama server's `/api/embeddings` endpoint
- Default model: `all-minilm` (or `nomic-embed-text` for better quality)
- Requires Ollama to be running with the model pulled
- Dimension determined on first call

**OpenAIEmbedding**
- Uses OpenAI's embeddings API
- Default model: `text-embedding-3-small` (1536 dimensions)
- Requires OPENAI_API_KEY environment variable
- Adds latency and cost but no local ML dependencies

### Factory Function

Provide a `get_embedding_backend(backend_type: str, **kwargs)` factory function that instantiates the appropriate backend based on configuration.

---

## API Endpoints

Base URL: `http://127.0.0.1:8283`

### Health Check

**GET /health**

Returns server status and configuration info.

Response:
```json
{
  "status": "ok",
  "embedding_backend": "local",
  "embedding_dimension": 384,
  "database_path": "/home/user/.local-memory/memory.db"
}
```

### Agents

**POST /agents**

Create a new agent or return existing one with the same name.

Request:
```json
{
  "name": "my_agent",
  "metadata": {}  // optional
}
```

Response: Agent object
```json
{
  "id": "uuid",
  "name": "my_agent",
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": null
}
```

**GET /agents/{name}**

Get agent by name. Returns 404 if not found.

Response: Agent object

### Messages

**POST /messages**

Store a message. Automatically generates and stores embedding.

Request:
```json
{
  "agent_name": "my_agent",
  "role": "user",  // "user", "assistant", or "system"
  "content": "Hello, world!",
  "metadata": {}  // optional
}
```

Response: Message object
```json
{
  "id": "uuid",
  "agent_id": "agent-uuid",
  "role": "user",
  "content": "Hello, world!",
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": null,
  "similarity": null
}
```

**GET /messages/{agent_name}**

List messages for an agent, most recent first.

Query parameters:
- `limit` (int, default 100): Maximum messages to return

Response: Array of Message objects

**POST /messages/search**

Semantic search over messages.

Request:
```json
{
  "agent_name": "my_agent",
  "query": "What's my name?",
  "limit": 5  // optional, default 5, max 20
}
```

Response: Array of Message objects with `similarity` field populated, sorted by similarity descending.

### Memory Blocks

**POST /memory-blocks**

Create a new memory block. Returns 409 if block with same label already exists.

Request:
```json
{
  "agent_name": "my_agent",
  "label": "human",
  "value": "Name: Alice\nLocation: Boston"
}
```

Response: MemoryBlock object
```json
{
  "id": "uuid",
  "agent_id": "agent-uuid",
  "label": "human",
  "value": "Name: Alice\nLocation: Boston",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

**GET /memory-blocks/{agent_name}**

List all memory blocks for an agent.

Response: Array of MemoryBlock objects

**GET /memory-blocks/{agent_name}/{label}**

Get a specific memory block. Returns 404 if not found.

Response: MemoryBlock object

**PUT /memory-blocks/{agent_name}/{label}**

Update a memory block's value. Returns 404 if not found.

Request:
```json
{
  "value": "Name: Alice\nLocation: New York"
}
```

Response: Updated MemoryBlock object

### Context Retrieval

**POST /context/{agent_name}**

Get context to inject into a prompt. This is the main endpoint used by the client interceptor.

Request:
```json
{
  "agent_name": "my_agent",
  "query": "What's my name?"
}
```

Response:
```json
{
  "memory_blocks": [
    {
      "id": "uuid",
      "agent_id": "agent-uuid",
      "label": "human",
      "value": "Name: Alice",
      "created_at": "...",
      "updated_at": "..."
    }
  ],
  "relevant_messages": [
    {
      "id": "uuid",
      "agent_id": "agent-uuid",
      "role": "user",
      "content": "My name is Alice",
      "created_at": "...",
      "metadata": null,
      "similarity": 0.87
    }
  ]
}
```

---

## Client Interceptor

The client library provides a `learning()` context manager that intercepts OpenAI API calls to inject memory context and capture conversations.

### Public Interface

```python
from client import learning

with learning(agent="my_agent"):
    response = client.chat.completions.create(...)
```

Parameters:
- `agent` (str, required): Agent name for memory namespace
- `server_url` (str, optional): Memory server URL, defaults to env var MEMORY_SERVER_URL or http://127.0.0.1:8283
- `capture_only` (bool, default False): If True, only capture conversations without injecting context

### How the Interceptor Works

The interceptor must perform these steps:

**On context manager entry:**
1. Create an HTTP client for the memory server
2. Verify the server is reachable (call /health)
3. Ensure the agent exists (call POST /agents)
4. Monkey-patch `openai.resources.chat.completions.Completions.create`

**On each patched create() call:**

*Before the request:*
1. Extract the user's query from the messages list (find the last message with role="user")
2. If not in capture_only mode, call POST /context/{agent_name} with the query
3. Format the returned context as a string
4. Inject the context as a system message at the beginning of the messages list (after any existing system message)

*After the response:*
1. Store the user message via POST /messages
2. Extract the assistant's response from the API response
3. Store the assistant message via POST /messages
4. Return the original response unchanged

**On context manager exit:**
1. Restore the original `create` function
2. Clean up any resources

### Context Formatting

Format the retrieved context as a string for injection. Suggested format:

```
## Memory

### human
Name: Alice
Location: Boston

### persona
I am a helpful assistant.

## Relevant Past Conversations

**User**: My name is Alice and I live in Boston.

**Assistant**: Nice to meet you, Alice! How can I help you today?
```

Truncate individual messages if they're very long (e.g., > 500 characters) to avoid bloating the context.

### Message Injection Strategy

Insert the context as a system message. If the messages list already has a system message at the start, insert the context message immediately after it. Otherwise, prepend it to the list.

The injected message should have role="system" and content that clearly delineates it as memory context, e.g.:

```
The following is context from your memory:

[formatted context here]
```

### Extracting User Content

The user's message content might be a string or a list (for multi-modal messages). Handle both cases:
- If string, use directly
- If list, extract text from content blocks with type="text" and concatenate

### Error Handling

- If the memory server is unreachable on context manager entry, raise a clear error telling the user to start the server
- If memory server calls fail during interception, log a warning but don't fail the LLM call—gracefully degrade to no-memory behavior

---

## Semantic Search Implementation

The search endpoint uses cosine similarity to rank messages.

### Algorithm

1. Generate embedding for the query using the configured backend
2. Load all messages with embeddings for the specified agent (limit to most recent N, e.g., 1000, for performance)
3. Compute cosine similarity between query embedding and each message embedding
4. Sort by similarity descending
5. Return top-k results

### Cosine Similarity

For vectors a and b:
```
similarity = dot(a, b) / (norm(a) * norm(b))
```

Use numpy for efficient computation.

### Performance Considerations

For a personal memory server with thousands of messages, loading all embeddings into memory and computing similarities is fast enough. If this becomes a bottleneck, consider:
- Adding a vector index (sqlite-vss, FAISS)
- Caching embeddings in memory
- Limiting search to recent time windows

---

## Testing Strategy

### Unit Tests

**database.py**
- Agent CRUD operations
- Message storage and retrieval
- Embedding serialization/deserialization round-trip
- Memory block CRUD operations
- Agent isolation (messages from one agent don't appear in another's queries)

**embeddings.py**
- Each backend produces arrays of correct shape
- Backends are interchangeable (same interface)
- Mock external calls for Ollama and OpenAI backends

**server.py**
- Use FastAPI's TestClient
- Test each endpoint's happy path
- Test error cases (404s, 409s, validation errors)
- Test context endpoint returns both memory blocks and relevant messages

**client.py**
- Mock the memory server with httpx mock
- Mock OpenAI client
- Verify context is injected into messages
- Verify user and assistant messages are captured
- Verify capture_only mode doesn't inject context
- Verify original create function is restored after context exit

### Integration Test

A manual test script that:
1. Starts with a fresh database
2. Creates an agent
3. Has a conversation where the user states their name
4. In a new `learning()` context, asks "what's my name?"
5. Verifies the LLM's response indicates it knows the name

This requires a real OpenAI API key and a running server.

---

## Usage Examples

### Starting the Server

```bash
# Default configuration (local embeddings)
uv run server.py

# With Ollama embeddings (lighter dependencies)
MEMORY_EMBEDDING_BACKEND=ollama uv run server.py

# With OpenAI embeddings (no local ML dependencies)
MEMORY_EMBEDDING_BACKEND=openai uv run server.py

# Custom port
MEMORY_SERVER_PORT=9000 uv run server.py
```

### Basic Client Usage

```python
from openai import OpenAI
from client import learning

client = OpenAI()

# Store information
with learning(agent="my_agent"):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "My name is Alice and I prefer Python."}]
    )

# Later, in a new session - agent remembers
with learning(agent="my_agent"):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What programming language should I use?"}]
    )
    # Response will reference Python preference

# Capture-only mode for logging
with learning(agent="logger", capture_only=True):
    response = client.chat.completions.create(...)
    # Conversation stored but no context injected
```

### Different Agents Have Isolated Memory

```python
with learning(agent="work_assistant"):
    # This agent knows about work stuff
    ...

with learning(agent="personal_assistant"):
    # This agent has completely separate memory
    ...
```

---

## Implementation Order

Suggested order for implementation:

1. **config.py** and **models.py** - Pure data definitions, no logic
2. **database.py** - Foundation for everything else, can be tested in isolation
3. **embeddings.py** - Start with LocalEmbedding only, add others after basic flow works
4. **server.py** - Build incrementally: health endpoint first, then agents, then messages, then memory blocks, then context
5. **client.py** - The trickiest part; implement last when server is stable

---

## Future Enhancements

Not in scope for initial implementation, but worth considering:

1. **Compaction/Summarization**: Periodically summarize old conversations to keep search space manageable
2. **Anthropic Support**: Add interceptor for Anthropic's client library
3. **Streaming Support**: Handle streaming responses in the interceptor
4. **Memory Block Tools**: Allow agents to self-manage memory via tool calls
5. **Web UI**: Simple dashboard to view and manage memories
6. **Export/Import**: Backup and restore memory databases
