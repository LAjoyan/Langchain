from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input
from langgraph.checkpoint.memory import MemorySaver
# Notera att vi importerar andra tools här:
from util.tools import calculate, get_current_time 
from langchain.agents import create_agent

def run():
    # 0.0 i temperatur för att inte "hitta på" data
    model = get_model(temperature=0.0) 
    memory = MemorySaver()

    # Tools för att hantera siffror och tidstämplar
    tools = [calculate, get_current_time] 
    
    agent = create_agent(
        model=model,
        system_prompt=(
            "### ROLE\n"
    "Du är en Data Extraction Bot.\n\n"
    "### OBJECTIVES\n"
    "1. Analysera texten användaren skickar.\n"
    "2. Extrahera namn, datum och platser.\n\n"
    "### OUTPUT\n"
    "Svara strikt med en strukturerad punktlista eller en tabell. Ingen småprat."
        ),
        tools=tools,
        checkpointer=memory
    )
    print(f"Agenten {__file__} är redo! Skriv 'exit' för att avsluta.")

    config = {"configurable": {"thread_id": "data-session-1"}}

    while True:
        user_input = get_user_input("Skicka text för extraktion")

        if user_input.lower() in ["exit", "quit"]:
            break
            
        process_stream = agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode=STREAM_MODES,
        )

        handle_stream(process_stream)

if __name__ == "__main__":
    run()