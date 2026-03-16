from langchain.agents import create_agent

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input

from langgraph.checkpoint.memory import MemorySaver

def run():
    # Get predefined attributes
    model = get_model(temperature=0.8, top_p=0.8)

    memory = MemorySaver()
    # Create agent
    agent = create_agent(
        model=model,
        system_prompt=(
            "Du är en hjälpsam assistent som svarar på användarens frågor."
            "Svara alltid på svenska och var koncis men informativ."
        ),
        checkpointer=memory
    )
    print("Agenten är redo! Skriv 'exit' för att avsluta.")
    
    config = {"configurable": {"thread_id": "conversation-1"}}

    while True:

    # Get user input
        user_input = get_user_input("Ställ din fråga")

        if user_input.lower() in ["exit", "quit"]:
            break
    # Call the agent
        process_stream = agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode=STREAM_MODES,
    )

    # Stream the process
        handle_stream(process_stream)


if __name__ == "__main__":
    run()
