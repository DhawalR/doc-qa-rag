from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def evaluate_faithfulness(
    answer: str,
    retrieved_chunks: list
) -> dict:
    """
    Measures whether the answer is grounded in the retrieved context.
    Uses LLM-as-a-judge technique.
    
    Args:
        answer: Generated answer from generation.py
        retrieved_chunks: List of Document objects used to generate answer
        
    Returns:
        Dictionary with score and reasoning
    """
    
    # Format chunks into readable context
    context = "\n\n".join([
        f"[Chunk {i+1}]: {chunk.page_content}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    prompt = f"""You are an evaluation expert assessing AI-generated answers.

Your task: Rate how FAITHFUL the answer is to the provided context.

Faithfulness means every claim in the answer must be directly 
supported by the context. Penalize any information in the answer 
that cannot be found in the context (hallucinations).

Score on this scale:
5 — Completely faithful, every claim supported by context
4 — Mostly faithful, minor unsupported details
3 — Partially faithful, some hallucinations present
2 — Mostly hallucinated, few claims supported
1 — Completely hallucinated, no grounding in context

Context:
{context}

Answer to evaluate:
{answer}

Respond in exactly this format:
Score: [1-5]
Reasoning: [your explanation]"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0    # Zero temperature for consistent evaluation
    )
    
    result = response.choices[0].message.content
    
    # Parse score and reasoning from response
    lines = result.strip().split("\n")
    score_line = lines[0]
    reasoning_line = "\n".join(lines[1:])
    
    score = int(score_line.replace("Score:", "").strip())
    reasoning = reasoning_line.replace("Reasoning:", "").strip()
    
    return {
        "score": score,
        "reasoning": reasoning,
        "max_score": 5
    }


def evaluate_retrieval_relevance(
    question: str,
    retrieved_chunks: list
) -> dict:
    """
    Measures whether retrieved chunks are actually relevant 
    to the question asked.
    
    Args:
        question: Original user question
        retrieved_chunks: List of Document objects from retriever
        
    Returns:
        Dictionary with per-chunk scores and average
    """
    
    chunk_scores = []
    
    for i, chunk in enumerate(retrieved_chunks):
        prompt = f"""You are an evaluation expert.

Rate how RELEVANT this retrieved chunk is to the question.

Relevance means the chunk contains information that helps 
answer the question.

Score on this scale:
5 — Highly relevant, directly answers the question
4 — Relevant, contains useful related information  
3 — Somewhat relevant, tangentially related
2 — Mostly irrelevant, little useful information
1 — Completely irrelevant

Question: {question}

Retrieved Chunk:
{chunk.page_content}

Respond in exactly this format:
Score: [1-5]
Reasoning: [your explanation]"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        lines = result.strip().split("\n")
        score = int(lines[0].replace("Score:", "").strip())
        reasoning = "\n".join(lines[1:]).replace("Reasoning:", "").strip()
        
        chunk_scores.append({
            "chunk_number": i + 1,
            "page": chunk.metadata.get('page', 0) + 1,
            "score": score,
            "reasoning": reasoning
        })
    
    average_score = sum(c["score"] for c in chunk_scores) / len(chunk_scores)
    
    return {
        "chunk_scores": chunk_scores,
        "average_score": round(average_score, 2),
        "max_score": 5
    }


def evaluate_answer_relevance(
    question: str,
    answer: str
) -> dict:
    """
    Measures whether the answer actually addresses the question asked.
    
    Args:
        question: Original user question
        answer: Generated answer from generation.py
        
    Returns:
        Dictionary with score and reasoning
    """
    
    prompt = f"""You are an evaluation expert.

Rate how RELEVANT the answer is to the question.

Relevance means the answer directly addresses what was asked.
A faithful answer can still be irrelevant if it talks about 
something other than what was asked.

Score on this scale:
5 — Perfectly addresses the question
4 — Mostly addresses the question, minor gaps
3 — Partially addresses the question
2 — Barely addresses the question
1 — Does not address the question at all

Question: {question}

Answer:
{answer}

Respond in exactly this format:
Score: [1-5]
Reasoning: [your explanation]"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    result = response.choices[0].message.content
    lines = result.strip().split("\n")
    score = int(lines[0].replace("Score:", "").strip())
    reasoning = "\n".join(lines[1:]).replace("Reasoning:", "").strip()
    
    return {
        "score": score,
        "reasoning": reasoning,
        "max_score": 5
    }


def run_full_evaluation(
    question: str,
    retrieved_chunks: list,
    results: dict
) -> None:
    """
    Runs all three evaluations for all three prompting strategies
    and prints a complete evaluation report.
    
    Args:
        question: Original user question
        retrieved_chunks: List of retrieved Document objects
        results: Dictionary of answers from run_all_prompting_strategies()
    """
    
    print("\n" + "="*50)
    print("FULL EVALUATION REPORT")
    print("="*50)
    
    # Evaluate retrieval once — same chunks for all strategies
    print("\nEvaluating retrieval relevance...")
    retrieval_eval = evaluate_retrieval_relevance(question, retrieved_chunks)
    
    print(f"\n--- RETRIEVAL RELEVANCE ---")
    print(f"Average Score: {retrieval_eval['average_score']} / 5")

    for chunk_score in retrieval_eval['chunk_scores']:
        print(
            f"  Chunk {chunk_score['chunk_number']} "
            f"(Page {chunk_score['page']}): "
            f"{chunk_score['score']}/5"
        )
        print(f"  Reasoning: {chunk_score['reasoning']}")
        print()
    # for chunk_score in retrieval_eval['chunk_scores']:
    #     print(
    #         f"  Chunk {chunk_score['chunk_number']} "
    #         f"(Page {chunk_score['page']}): "
    #         f"{chunk_score['score']}/5 — {chunk_score['reasoning'][:80]}..."
    #     )
    
    # Evaluate each prompting strategy
    for strategy_name, answer in results.items():
        print(f"\n{'='*40}")
        print(f"STRATEGY: {strategy_name.upper().replace('_', ' ')}")
        print(f"{'='*40}")
        
        print("\nEvaluating faithfulness...")
        faith_eval = evaluate_faithfulness(answer, retrieved_chunks)
        
        print("\nEvaluating answer relevance...")
        relevance_eval = evaluate_answer_relevance(question, answer)
        
        print(f"\nFaithfulness:     {faith_eval['score']}/5")
        print(f"Answer Relevance: {relevance_eval['score']}/5")
        print(f"Overall:          {(faith_eval['score'] + relevance_eval['score']) / 2}/5")
        
        print(f"\nFaithfulness Reasoning:")
        print(f"  {faith_eval['reasoning']}")
        
        print(f"\nAnswer Relevance Reasoning:")
        print(f"  {relevance_eval['reasoning']}")

        # print(f"\nFaithfulness Reasoning:")
        # print(f"  {faith_eval['reasoning'][:200]}")
        
        # print(f"\nAnswer Relevance Reasoning:")
        # print(f"  {relevance_eval['reasoning'][:200]}")