from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os


class AgentState(TypedDict):
    """State for the coding agent."""
    task: str
    generated_code: str
    output_file: str


def create_code_generator_node(llm):
    """
    Create a code generator node that uses LLM to generate code.
    
    Args:
        llm: The LangChain LLM instance to use for code generation
    """
    # System prompt for code generation
    system_prompt = """You are an expert Python programmer. Your task is to generate clean, correct, and executable Python code based on the user's requirements.

Instructions:
1. Generate only Python code - no explanations, no markdown formatting, no code blocks
2. The code should be complete and executable
3. Include a main block that demonstrates the functionality with example inputs
4. Make sure to print the result so it can be evaluated
5. Write idiomatic Python code following best practices
6. Handle edge cases appropriately
7. The code should be self-contained and runnable

Generate the code directly without any additional text."""

    user_prompt_template = "Task: {task}\n\nGenerate Python code for this task:"
    
    def code_generator_node(state: AgentState) -> AgentState:
        """
        Single node that generates code based on the task using LLM.
        """
        task = state["task"]
        output_file = state.get("output_file", "generated_code.py")
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt_template)
        ])
        
        # Format the prompt with the task
        formatted_prompt = prompt.format_messages(task=task)
        
        # Call the LLM
        response = llm.invoke(formatted_prompt)
        
        # Extract the code from the response
        code = response.content.strip()
        
        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "").strip()
        elif code.startswith("```"):
            code = code.replace("```", "").strip()
        
        return {
            "generated_code": code,
            "output_file": output_file
        }
    
    return code_generator_node


class CodingAgent:
    """Simple coding agent using LangGraph with a single code generator node."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the coding agent graph.
        
        Args:
            model_name: OpenAI model name to use (default: gpt-4o-mini)
            temperature: Temperature for the LLM (default: 0.0 for deterministic output)
        """
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create the code generator node with the LLM
        code_generator_node = create_code_generator_node(self.llm)
        
        # Create the graph
        self.graph = StateGraph(AgentState)
        
        # Add the single code generator node
        self.graph.add_node("code_generator", code_generator_node)
        
        # Set entry point
        self.graph.set_entry_point("code_generator")
        
        # Connect to END
        self.graph.add_edge("code_generator", END)
        
        # Compile the graph
        self.app = self.graph.compile()
    
    def generate_code(self, task: str, output_file: str = "generated_code.py") -> dict:
        """
        Generate code for a given task and save it to a file.
        
        Args:
            task: Description of the coding task
            output_file: Path where the generated code should be saved
        
        Returns:
            Dictionary containing generated_code and output_file
        """
        # Run the agent
        initial_state = {
            "task": task,
            "generated_code": "",
            "output_file": output_file
        }
        
        result = self.app.invoke(initial_state)
        
        # Save the generated code to a file
        with open(output_file, "w") as f:
            f.write(result["generated_code"])
        
        return result
