from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input
from langgraph.checkpoint.memory import MemorySaver
from util.tools import read_local_file
from langchain.agents import create_agent

def run():
    model = get_model(temperature=0.0) 
    memory = MemorySaver()

    tools = [read_local_file] 
    
    agent = create_agent(
        model=model,
        system_prompt=(
            "### ROLE\n"
            "Du är en Data Extraction Bot specialiserad på strukturerad information.\n\n"
            "### OBJECTIVES\n"
            "1. Analysera texten användaren skickar och extrahera namn, datum och siffror.\n"
            "2. Använd 'calculate' om användaren ber om matematiska sammanställningar.\n"
            "3. Använd 'get_current_time' om användaren behöver tidstämplar för datan.\n\n"
            "### OUTPUT FORMAT\n"
            "Svara strikt med en strukturerad punktlista eller en Markdown-tabell. Ingen småprat eller introduktioner."
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