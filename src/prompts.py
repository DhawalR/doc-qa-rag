def format_context(retrieved_chunks: list) -> str:
    """
    Takes retrieved chunks and formats them into a single
    readable context string to inject into the prompt.
    Args:
        retrieved_chunks: List of Document objects from retriever
    Returns:
        Formatted context string
    """
    
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks):
        page_num = chunk.metadata.get('page', 0) + 1
        context_parts.append(
            f"[Chunk {i+1} - Page {page_num}]\n{chunk.page_content}"
        )
    
    # Join all chunks with clear separation
    return "\n\n".join(context_parts)


def zero_shot_prompt(question: str, context: str) -> str:
    """
    Simple prompt with context but no examples or reasoning instructions.
    The baseline approach.
    Args:
        question: User's question
        context: Formatted context string from format_context() 
    Returns:
        Complete prompt string
    """
    
    return f"""You are a helpful assistant that answers questions about documents.
Use only the provided context to answer the question.
If the answer is not in the context, say "I cannot find this in the document."

Context:
{context}

Question: {question}

Answer:"""


def few_shot_prompt(question: str, context: str) -> str:
    """
    Prompt with example Q&A pairs to guide the LLM's response style.
    Examples teach the LLM the format and depth we expect.
    Args:
        question: User's question
        context: Formatted context string from format_context()
    Returns:
        Complete prompt string
    """
    
    return f"""You are a helpful assistant that answers questions about documents.
Use only the provided context to answer the question.
If the answer is not in the context, say "I cannot find this in the document."

Here are some examples of good answers:

Example 1:
Question: "What problem does the paper solve?"
Answer: "Based on the context, the paper addresses the limitation of 
sequential computation in RNNs by introducing the Transformer, which 
relies entirely on attention mechanisms and allows for parallelization."

Example 2:
Question: "What datasets were used for evaluation?"
Answer: "According to the provided context, the model was evaluated on 
the WMT 2014 English-German and English-French translation tasks."

Now answer the following:

Context:
{context}

Question: {question}

Answer:"""


def chain_of_thought_prompt(question: str, context: str) -> str:
    """
    Prompt that instructs the LLM to reason step by step before answering.
    Produces more accurate answers for complex questions.
    
    Args:
        question: User's question
        context: Formatted context string from format_context()
        
    Returns:
        Complete prompt string
    """
    
    return f"""You are a helpful assistant that answers questions about documents.
Use only the provided context to answer the question.
If the answer is not in the context, say "I cannot find this in the document."

Think through this step by step:
1. Identify which chunks are most relevant to the question
2. Extract the key information from those chunks
3. Synthesize a clear and accurate answer
4. Verify your answer is supported by the context

Context:
{context}

Question: {question}

Let's think step by step:
Answer:"""


def inspect_prompt(prompt: str, prompt_type: str) -> None:
    """
    Displays a prompt for inspection before sending to LLM.
    
    Args:
        prompt: The complete prompt string
        prompt_type: Name of the prompt strategy used
    """
    
    print("\n" + "="*50)
    print(f"PROMPT INSPECTION — {prompt_type.upper()}")
    print("="*50)
    print(f"\nTotal characters: {len(prompt)}")
    print(f"\nFull prompt:")
    print("-"*40)
    print(prompt)
    print("-"*40)