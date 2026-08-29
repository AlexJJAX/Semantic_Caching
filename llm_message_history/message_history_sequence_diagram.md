# RedisVL Message History — Multi-Session Request Sequence

This sequence shows how the demonstration seeds three isolated histories, retrieves each
session by exact tag, sends the same prompt with different context, stores each exchange, and
removes only its owned Redis state at shutdown.

```mermaid
sequenceDiagram
    autonumber

    actor Operator
    participant Main as Application<br/>main() / run_demo()
    participant History as RedisVL<br/>SemanticMessageHistory
    participant Vectorizer as Local Hugging Face<br/>Message Vectorizer
    participant Redis as Redis Hash + Search<br/>Shared Message Index
    participant Adapter as OpenAIClient<br/>Role Adapter
    participant OpenAI as OpenAI Chat Completions<br/>GPT-5.6 Luna

    Note over Operator,OpenAI: Initialization and controlled reset

    Operator->>Main: Run Multiple_sessions.py
    Main->>Main: Load settings and validate OPENAI_API_KEY
    Main->>Redis: Create bounded client and PING
    Redis-->>Main: PONG
    Main->>Adapter: Create OpenAI client with configured model
    Main->>History: Initialize namespaced history and key prefix
    History->>Redis: Create or validate message Search index
    Redis-->>History: Index ready
    Main->>History: clear()
    History->>Redis: Delete stale messages owned by the demo prefix
    Redis-->>History: Controlled namespace is empty

    Note over Main,Redis: Seed three independent session tags

    loop Student, young professional, retired pensioner
        Main->>History: add_messages(4 persona messages, session_tag)
        loop Each seeded message
            History->>Vectorizer: Embed message content locally
            Vectorizer-->>History: Message vector
            History->>Redis: Store Hash record with session_tag,<br/>role, content, timestamp, and vector
            Redis-->>History: Record stored
        end
    end

    Note over Operator,OpenAI: Same prompt, session-specific context

    loop Each session tag, processed sequentially
        Main->>History: get_recent(session_tag, limit=5)
        History->>Redis: Search with exact session_tag filter,<br/>sort timestamp descending, limit 5
        Redis-->>History: Matching records only
        History->>History: Restore chronological order
        History-->>Main: Selected session context

        Main->>Adapter: converse(shared prompt, context)
        Adapter->>Adapter: Preserve user, system, and assistant;<br/>map legacy llm to assistant
        Note right of Adapter: Unknown roles raise ValueError<br/>before the model call
        Adapter->>OpenAI: Context plus new user prompt
        OpenAI-->>Adapter: Session-specific assistant response
        Adapter-->>Main: Response text

        Main->>History: store_exchange(prompt, response, session_tag)
        loop New user and assistant messages
            History->>Vectorizer: Embed message content locally
            Vectorizer-->>History: Message vector
            History->>Redis: Append record under the same session_tag
            Redis-->>History: Record stored
        end
        Main-->>Operator: Print persona label and response
    end

    Note over Main,Redis: Inspect one partition

    Main->>History: get_recent(student, limit=5)
    History->>Redis: Exact student TAG filter and recency sort
    Redis-->>History: Five newest student messages
    History-->>Main: Chronological student history
    Main-->>Operator: Print stored records

    Note over Main,Redis: Nested finally cleanup on success or failure

    Main->>History: delete()
    History->>Redis: Drop owned Search index and message keys
    Redis-->>History: Demo state removed
    Main->>Adapter: close()
    Adapter->>OpenAI: Close API client
    Main->>Redis: Close connection pool
    Main-->>Operator: Process exits
```
