from langchain.agents import create_agent

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input

from langgraph.checkpoint.memory import MemorySaver
from util.tools import calculate, get_current_time, read_local_file, scrape_website, search_documents

def run():
    # Get predefined attributes
    model = get_model(temperature=0.1)

    memory = MemorySaver()

    tools = [calculate, get_current_time, read_local_file, scrape_website, search_documents ] 
    # Create agent
    agent = create_agent(
        model=model,
        system_prompt=(
            "Du är en expert-assistent med tillgång till realtidsverktyg. "
        "Lita ALLTID på informationen från dina verktyg. " 
        "Om get_current_time ger dig en tid, så ÄR det den aktuella tiden. "
        "Svara alltid på svenska."
        "Du kan läsa filer på datorn. "
    "När du använder read_local_file, anta att filerna ligger i "
    "den aktuella mappen om inget annat anges. "
    "Gissa INTE sökvägar som '/path/to/', använd bara filnamnet."
    "Du är en hjälpsam assistent med tillgång till internet. "
            "Om en användare frågar om innehåll på en specifik URL, använd 'requests_get' "
            "för att hämta texten från webbsidan."
            "Du är en proaktiv nyhetsanalytiker. "
            "Använd 'scrape_website' för att läsa nyheter på nätet. "
            "Ignorera ALLTID teknisk metadata, JSON och kod. "
            "Din uppgift är att sammanfatta den viktigaste nyheten i tre korta punkter på svenska."
            "Du är 'FinanceScan', en analytiker som hjälper privatpersoner förstå nyheter. "
        "När en användare ger dig en länk eller frågar om en nyhet: "
        "1. Använd 'scrape_website' för att läsa artikeln. "
        "2. Hitta viktiga siffror (t.ex. räntesänkningar, prishöjningar eller skatteändringar). "
        "3. Fråga användaren om deras egna siffror (t.ex. lån eller lön). "
        "4. Använd 'calculate' för att visa den faktiska kostnaden eller besparingen. "
        "Svara alltid på svenska och var tydlig med matten."
        ),
        tools=tools,
        checkpointer=memory
    )
    print("Agenten är redo! Skriv 'exit' för att avsluta.")

    config = {"configurable": {"thread_id": "tool-session-1"}}

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
