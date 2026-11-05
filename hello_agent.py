"""
hello_agent.py - Step 3 smoke test.

Creates ONE AutoGen agent, sends it one message, and prints the reply.
If this works, your API key, model and AutoGen setup are all correct.

Run:  python hello_agent.py
"""
import asyncio
import time

from autogen_agentchat.agents import AssistantAgent

from config import MODEL_NAME, get_model_client


async def main() -> None:
    model_client = get_model_client()

    # An agent = a name + a description + a model + a system message (its job)
    agent = AssistantAgent(
        name="support_greeter",
        description="Greets customers of BluePeak Store.",
        model_client=model_client,
        system_message=(
            "You are a friendly customer support assistant for BluePeak Store, "
            "an online store in India. Reply in at most two sentences."
        ),
    )

    print(f"Model: {MODEL_NAME}")
    print("Sending a test message...\n")

    start = time.perf_counter()
    result = await agent.run(
        task="Hi, I just placed my first order. What can you help me with?"
    )
    elapsed = time.perf_counter() - start

    reply = result.messages[-1]
    print("Agent reply:")
    print(reply.content)

    if reply.models_usage:
        print(
            f"\nTokens used: {reply.models_usage.prompt_tokens} in, "
            f"{reply.models_usage.completion_tokens} out"
        )
    print(f"Time taken: {elapsed:.1f} seconds")
    print("\nStep 3 complete: your agent is talking to OpenAI.")

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
