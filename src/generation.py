from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL
from src.prompts import (
    format_context,
    zero_shot_prompt,
    few_shot_prompt,
    chain_of_thought_prompt,
    inspect_prompt
)

# Initialize the OpenAI client once at module level
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_answer(prompt: str) -> str:
    """
    Sends an augmented prompt to the LLM and returns the answer.
    Args:
        prompt: Complete augmented prompt string from prompts.py
    Returns:
        LLM generated answer as a string
    """
    
    response = client.chat.completions.create(
        model=LLM_MODEL,          # gpt-4o-mini
        messages=[
            {
                # System message sets the LLM's overall behavior
                "role": "system",
                "content": (
                    "You are a precise document assistant. "
                    "Answer only from the provided context. "
                    "Never hallucinate or invent information."
                )
            },
            {
                # User message contains our full augmented prompt
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,    # Low temperature for factual accuracy
        max_tokens=1000     # Maximum length of the answer
    )
    
    # Extract just the text from the response object
    answer = response.choices[0].message.content
    
    return answer


def run_all_prompting_strategies(
    question: str,
    retrieved_chunks: list
) -> dict:
    """
    Runs the same question through all three prompting strategies
    and returns all answers for comparison.
    Args:
        question: User's question
        retrieved_chunks: List of retrieved Document objects  
    Returns:
        Dictionary with strategy names as keys and answers as values
    """
    
    context = format_context(retrieved_chunks)
    
    strategies = {
        "zero_shot": zero_shot_prompt(question, context),
        "few_shot": few_shot_prompt(question, context),
        "chain_of_thought": chain_of_thought_prompt(question, context)
    }
    
    results = {}
    
    for strategy_name, prompt in strategies.items():
        print(f"\nGenerating answer using {strategy_name}...")
        answer = generate_answer(prompt)
        results[strategy_name] = answer
        print(f"Done.")
    
    return results


def inspect_answers(results: dict, question: str) -> None:
    """
    Displays all answers side by side for easy comparison.
    Args:
        results: Dictionary from run_all_prompting_strategies()
        question: The original question asked
    """
    
    print("\n" + "="*50)
    print("GENERATION RESULTS")
    print("="*50)
    print(f"\nQuestion: '{question}'\n")
    
    for strategy_name, answer in results.items():
        print(f"\n--- {strategy_name.upper().replace('_', ' ')} ---")
        print(answer)
        print()