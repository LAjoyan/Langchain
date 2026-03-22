from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input
from langgraph.checkpoint.memory import MemorySaver
from util.tools import scrape_website, search_documents
from langchain.agents import create_agent

def run():
    model = get_model(temperature=0.8) 
    memory = MemorySaver()

    tools = [scrape_website, search_documents] 
    
    agent = create_agent(
        model=model,
        system_prompt=(
            "### ROLE\n"
    "Du är en kreativ Marketing Director.\n\n"
    "### OBJECTIVES\n"
    "1. Skapa engagerande rubriker och säljtexter.\n"
    "2. Förvandla tråkiga produktlistor till spännande kampanjer.\n\n"
    "### STYLE\n"
    "Använd emojis, var entusiastisk och skriv på ett sätt som fångar uppmärksamhet."

        ),
        tools=tools,
        checkpointer=memory
    )
    print(f"Agenten {__file__} är redo! Skriv 'exit' för att avsluta.")

    config = {"configurable": {"thread_id": "tool-session-1"}}

    while True:
        user_input = get_user_input("Ställ din fråga")

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
