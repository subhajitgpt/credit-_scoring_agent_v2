from dotenv import load_dotenv

load_dotenv()

import os
import importlib

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from schemas import AgentResponse

def _get_tools():
    try:
        module = importlib.import_module("langchain_tavily")
        TavilySearch = getattr(module, "TavilySearch")
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: 'langchain-tavily'. Install it (or select the correct venv) and try again."
        ) from e
    return [TavilySearch()]


def _get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: 'langchain-ollama'. Install it and ensure Ollama is running."
            ) from e
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return ChatOllama(model=model, temperature=0)

    if provider == "openai":

        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: 'langchain-openai'. Install it (or select the correct venv)."
            ) from e
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}.")


def main():
    tools = _get_tools()
    llm = _get_llm()

    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=AgentResponse,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                }
            ]
        }
    )
    # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
