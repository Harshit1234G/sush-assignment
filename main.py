from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


def get_summary_and_sentiment(transcript: str) -> dict[str, str]:
    prompt = ChatPromptTemplate(
        messages= [
            (
                "system", 
                "ROLE: You are an AI assistant that analyzes customer support call transcripts.\n"
                "TASK:\n"
                "1. Summarize the conversation in 1-3 sentences, do not change the customers's intent or add new details.\n"
                "2. Extract the customer's sentiment (only one word: Positive, Neutral, or Negative.)\n"
                "OUTPUT:\n"
                "Format your response as JSON with two keys:\n"
                "{{\n"
                '   "summary": "string",\n'
                '   "sentiment": "string"\n'
                '}}'
            ),
            ('human', 'TRANSCRIPT: {transcript}')
        ]
    )


if __name__ == '__main__':
    load_dotenv()

