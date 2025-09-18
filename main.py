import os
import csv
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# --------------------------------
# Core Function
# --------------------------------
def get_summary_and_sentiment(
        transcript: str, 
        *, 
        model: str = 'llama-3.1-8b-instant', 
        temperature: float = 0.3
    ) -> dict[str, str] | None:
    """Analyze a customer call transcript using a Groq LLM.

    Args:
        transcript (str): The customer call transcript text.
        model (str, optional): Groq model to use. Defaults to 'llama-3.1-8b-instant'.
        temperature (float, optional): Sampling temperature, between 0 (deterministic) to 1 (creative). Defaults to 0.3.

    Returns:
        dict[str, str] | None: Dictionary with keys `summary` and `sentiment`, or None if the API fails.
    """
    prompt = ChatPromptTemplate(
        messages= [
            (
                "system", 
                "ROLE: You are an AI assistant that analyzes customer support call transcripts.\n"
                "TASK:\n"
                "1. Summarize the conversation in 1-3 sentences, do not change the customers's intent or add new details.\n"
                "2. Extract the customer's sentiment (exactly one word: Positive, Neutral, or Negative.)\n"
                "OUTPUT:\n"
                "Format your response as JSON with two keys:\n"
                "{{\n"
                '   "summary": "string",\n'
                '   "sentiment": "string"\n'
                "}}\n"
                "- Do not include anything outside the JSON."
            ),
            ('human', 'TRANSCRIPT: {transcript}')
        ]
    )

    try:
        llm = ChatGroq(
            model_name= model,
            temperature= temperature
        )

        parser = JsonOutputParser(
            pydantic_object= {
                'summary': 'string',
                'sentiment': 'string'
            }
        )

        chain = prompt | llm | parser
        result = chain.invoke({'transcript': transcript})
        return result
    
    except Exception as e:
        print(f'[ERROR] Failed to analyze transcript: {e}')
        return None


# --------------------------------
# CSV Handling
# --------------------------------
def to_csv(
        transcript: str, 
        response: dict[str, str], 
        *, 
        path: str = 'call_analysis.csv'
    ) -> None:
    """Save transcript analysis results into a CSV file.

    Args:
        transcript (str): Original transcript text.
        response (dict[str, str]): Contains `summary` and `sentiment`.
        path (str, optional): File path for CSV. Defaults to 'call_analysis.csv'.
    """
    try:
        file_exists = os.path.isfile(path)

        with open(path, 'a', encoding= 'utf-8', newline= '') as csvfile:
            writer = csv.writer(
                csvfile,
                quoting= csv.QUOTE_ALL
            )

            # header
            if not file_exists:
                writer.writerow(['Transcript', 'Summary', 'Sentiment'])

            writer.writerow([
                transcript.strip(),
                response.get('summary', ''),
                response.get('sentiment', '')
            ])

    except Exception as e:
        print(f'[ERROR] Failed to write to CSV: {e}')


if __name__ == '__main__':
    load_dotenv()

    print('\n📞 Call Transcript Analysis App')
    print('Type/paste a transcript and press Enter.')
    print('Type "exit" or "quit" to quit.\n')

    while True:
        transcript = input('Enter transcript: ').strip()
        if transcript.lower() in ('exit', 'quit'):
            print('Exiting... Goodbye!')
            break

        result = get_summary_and_sentiment(transcript)
        if not result:
            print(f'[ERROR] Could not analyze transcript.\n')
            continue

        print("\n--- Analysis Result ---")
        print(f"Transcript: {transcript}")
        print(f"Summary   : {result['summary']}")
        print(f"Sentiment : {result['sentiment']}")
        print("------------------------\n")

        to_csv(transcript, result)
        print(f"✅ Saved to call_analysis.csv\n")

