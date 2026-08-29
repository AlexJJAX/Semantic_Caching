"""
LLM Message History - Multiple Sessions

Demonstrates using Redis (via redisvl SemanticMessageHistory) to manage
multiple simultaneous conversation sessions with an OpenAI LLM backend.

Requires local Redis instance running (on default port 6379).
"""

# --- Stdlib ---
from typing import Dict, List, Optional

# --- Third-party ---
from openai import OpenAI
from redis import Redis
from redisvl.extensions.message_history import SemanticMessageHistory
from redisvl.utils.vectorize import BaseVectorizer

from redis_ai_portfolio.config import PortfolioSettings, get_settings
from redis_ai_portfolio.redis import create_redis_client

# --- Redis Connection ---

SETTINGS = get_settings()
REDIS_URL = SETTINGS.redis_url


# --- OpenAI LLM Client ---

class OpenAIClient:
    """Thin wrapper around the OpenAI chat completions API that adapts message history format."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        api_key = api_key or SETTINGS.openai_api_key
        self.client = OpenAI(api_key=api_key)
        self._model = model or SETTINGS.openai_model

    def converse(self, prompt: str, context: List[Dict]) -> str:
        """Send a prompt with conversation context to the OpenAI chat completions endpoint."""
        messages = self.remap(context)
        # Append the new user message after the existing history
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty response")
        return content

    def close(self) -> None:
        self.client.close()

    def remap(self, context: List[Dict]) -> List[Dict]:
        """Convert redisvl message history format to OpenAI messages format.

        Current RedisVL and OpenAI both use 'assistant'. The legacy RedisVL
        'llm' value is still accepted so existing histories remain readable.
        """
        role_map = {
            "user": "user",
            "assistant": "assistant",
            "llm": "assistant",
            "system": "system",
        }
        new_context = []
        for statement in context:
            role = statement["role"]
            if role not in role_map:
                raise ValueError(f"Unknown chat role: {role!r}")
            new_context.append({"role": role_map[role], "content": statement["content"]})
        return new_context


def create_message_history(
    redis_client: Redis,
    *,
    settings: PortfolioSettings = SETTINGS,
    vectorizer: BaseVectorizer | None = None,
) -> SemanticMessageHistory:
    """Create the namespaced RedisVL history with an injectable vectorizer."""
    return SemanticMessageHistory(
        name=settings.redis_name("idx", "message-history", "budgeting"),
        prefix=f"{settings.redis_name('message-history', 'budgeting')}:",
        redis_client=redis_client,
        vectorizer=vectorizer,
    )


def store_exchange(
    history: SemanticMessageHistory,
    prompt: str,
    response: str,
    *,
    session_tag: str,
) -> None:
    """Store one exchange with RedisVL's current assistant-role convention."""
    history.add_messages(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        session_tag=session_tag,
    )


# --- Main Demo ---

def run_demo(history: SemanticMessageHistory, client: OpenAIClient) -> None:
    """Run the budgeting scenario with initialized dependencies."""

    # --- Session Tags (personas) ---
    student = "student"
    yp = "young professional"
    retired = "retired pensioner"

    # --- Seed conversation history for each persona ---

    # Student session
    history.add_messages(
        [
            {"role": "system", "content": "You are a personal assistant helping people create sound financial budgets. Be very brief and concise in your responses."},
            {"role": "user",   "content": "I'm a college student living in Montana and I need help creating a budget. I am a first year accounting student."},
            {"role": "assistant", "content": "Sure, I can help you with that. What is your monthly income and average monthly expenses?"},
            {"role": "user",   "content": "my rent is $500, utilities are $100, and I spend $200 on groceries. I make $1000 a month as a part time tutor."},
        ],
        session_tag=student,
    )

    # Young professional session
    history.add_messages(
        [
            {"role": "system", "content": "You are a personal assistant helping people create sound financial budgets. Be very brief and concise in your responses."},
            {"role": "user",   "content": "I'm a young professional living in New York City and I need help planning for retirement. I already have a sizable emergency fund."},
            {"role": "assistant", "content": "Sure I can help you with that. What is your monthly income and average monthly expenses?"},
            {"role": "user",   "content": "I make $5000 a month as a software engineer. My rent is $2000, utilities are $200, groceries are $300, and I spend $500 on entertainment."},
        ],
        session_tag=yp,
    )

    # Retiree session
    history.add_messages(
        [
            {"role": "system", "content": "You are a personal assistant helping people create sound financial budgets. Be very brief and concise in your responses."},
            {"role": "user",   "content": "I'm a retired pensioner living in Florida and I need help creating a budget."},
            {"role": "assistant", "content": "Sure I can help you with that. What is your monthly income and average monthly expenses?"},
            {"role": "user",   "content": "I make $2000 a month from my pension. I own my home outright, utilities are $100, groceries are $200, and I spend $100 on entertainment."},
        ],
        session_tag=retired,
    )

    # --- Query each session with the same prompt ---
    # Each session_tag retrieves only that persona's conversation context,
    # so the same LLM call produces contextually distinct responses.

    prompt = "What is the single most important thing I should focus on financially?"

    context = history.get_recent(session_tag=student)
    response = client.converse(prompt=prompt, context=context)
    store_exchange(history, prompt, response, session_tag=student)
    print("Student: ", prompt)
    print("\nLLM: ", response)

    context = history.get_recent(session_tag=yp)
    response = client.converse(prompt=prompt, context=context)
    store_exchange(history, prompt, response, session_tag=yp)
    print("Young Professional: ", prompt)
    print("\nLLM: ", response)

    context = history.get_recent(session_tag=retired)
    response = client.converse(prompt=prompt, context=context)
    store_exchange(history, prompt, response, session_tag=retired)
    print("Retiree: ", prompt)
    print("\nLLM: ", response)

    # --- Inspect stored history for the student session ---
    print("\nStudent session history:")
    for ctx in history.get_recent(session_tag=student):
        print(ctx)



def main() -> None:
    """Initialize the demo and guarantee scoped cleanup on success or failure."""
    if not SETTINGS.openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

    redis_client = create_redis_client(REDIS_URL)
    openai_client: OpenAIClient | None = None
    history: SemanticMessageHistory | None = None
    try:
        redis_client.ping()
        openai_client = OpenAIClient()
        history = create_message_history(redis_client, settings=SETTINGS)
        # This is an intentionally ephemeral demo; clear only its owned prefix.
        history.clear()
        run_demo(history, openai_client)
    finally:
        try:
            if history is not None:
                history.delete()
        finally:
            try:
                if openai_client is not None:
                    openai_client.close()
            finally:
                redis_client.close()


if __name__ == "__main__":
    main()
