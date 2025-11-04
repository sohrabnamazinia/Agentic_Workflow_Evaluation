from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os


class StaticEvaluator:
    """LLM-based judge evaluator that evaluates code output against the task requirements."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the LLM judge evaluator.
        
        Args:
            model_name: OpenAI model name to use (default: gpt-4o-mini)
            temperature: Temperature for the LLM (default: 0.0 for deterministic output)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # System prompt for the judge
        self.system_prompt = """You are an expert code evaluator. Your task is to judge whether the generated code output correctly satisfies the given task requirements.

Instructions:
1. Carefully analyze the original task/query
2. Examine the output produced by the generated code
3. Determine if the output correctly fulfills the task requirements
4. Respond with ONLY one word: either "PASS" or "FAIL"
5. Do not provide explanations, justifications, or any other text
6. If the output correctly implements what was requested in the task, respond with "PASS"
7. If the output does not meet the requirements or is incorrect, respond with "FAIL"

Your response must be exactly one word: PASS or FAIL"""
    
    def evaluate(self, output: str, query: str) -> dict:
        """
        Evaluate the code output against the task query using LLM judge.
        
        Args:
            output: The output produced by running the generated code
            query: The original task/query that the code was supposed to fulfill
        
        Returns:
            Dictionary with 'pass' (bool), 'output' (str), 'query' (str), and 'judgment' (str)
        """
        user_prompt = f"""Task/Query: {query}

Code Output:
{output}

Does this output correctly fulfill the task requirements? Respond with PASS or FAIL."""
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", user_prompt)
        ])
        
        # Format and call the LLM
        formatted_prompt = prompt.format_messages()
        response = self.llm.invoke(formatted_prompt)
        
        # Extract the judgment
        judgment = response.content.strip().upper()
        
        # Determine pass/fail
        passed = "PASS" in judgment
        
        return {
            "pass": passed,
            "output": output,
            "query": query,
            "judgment": judgment
        }
    
    def evaluate_and_print(self, output: str, query: str) -> bool:
        """
        Evaluate and print the result.
        
        Args:
            output: The output produced by running the generated code
            query: The original task/query that the code was supposed to fulfill
        
        Returns:
            True if pass, False otherwise
        """
        result = self.evaluate(output, query)
        
        print(f"\nEvaluating code output against task:")
        print(f"Task/Query: {result['query']}")
        print(f"Code Output: {result['output']}")
        print(f"LLM Judgment: {result['judgment']}")
        
        if result['pass']:
            print("✅ PASS")
        else:
            print("❌ FAIL")
        
        return result['pass']
