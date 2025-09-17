import csv
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def get_summary_and_sentiment(
        transcript: str, 
        *, 
        model: str = 'llama-3.1-8b-instant', 
        temperature: float = 0.3
    ) -> dict[str, str]:
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


def to_csv(
        transcript: str, 
        response: dict[str, str], 
        *, 
        path: str = 'call_analysis.csv'
    ) -> None:
    with open(path, 'a', encoding= 'utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter= '|')
        writer.writerow([transcript] + [response.get('summary')] + [response.get('sentiment')])


if __name__ == '__main__':
    load_dotenv()

    with open('transcripts/negative.txt', 'r', encoding= 'utf-8') as f:
        transcript = f.read()

    result = get_summary_and_sentiment(transcript)
    print(result)

    to_csv(transcript, result)

