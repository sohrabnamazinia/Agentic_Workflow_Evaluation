from coding_agent import CodingAgent
from static_evaluator import StaticEvaluator
import subprocess
import sys
import os


def run_code(code_file: str) -> tuple[str, str, bool]:
    """
    Run a Python file and return its output.
    
    Args:
        code_file: Path to the Python file to execute
    
    Returns:
        Tuple of (stdout, stderr, success)
    """
    if not os.path.exists(code_file):
        return "", f"File not found: {code_file}", False
    
    try:
        result = subprocess.run(
            [sys.executable, code_file],
            capture_output=True,
            text=True,
            timeout=10  # 10 second timeout
        )
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        success = result.returncode == 0
        
        return stdout, stderr, success
        
    except subprocess.TimeoutExpired:
        return "", "Execution timed out", False
    except Exception as e:
        return "", str(e), False


def main():
    """Test the coding agent and static evaluator."""
    
    # Initialize the coding agent
    print("Initializing coding agent...")
    agent = CodingAgent()
    
    # Define a test task
    task = "generate a python code to generate 10 random numbers between 1 and 100"
    output_file = "test_generated_code.py"
    
    print(f"\nTask: {task}")
    print("Generating code...")
    
    # Generate code
    result = agent.generate_code(task, output_file)
    
    print(f"\nGenerated code saved to: {output_file}")
    print("\nGenerated code:")
    print("-" * 50)
    print(result["generated_code"])
    print("-" * 50)
    
    # Run the generated code
    print("\nRunning generated code...")
    stdout, stderr, success = run_code(output_file)
    
    if not success:
        print(f"❌ Code execution failed!")
        print(f"Error: {stderr}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return
    
    print(f"Code Output:")
    print("-" * 50)
    print(stdout)
    print("-" * 50)
    
    # Initialize LLM judge evaluator
    evaluator = StaticEvaluator()
    
    print("\nRunning LLM judge evaluator...")
    print("-" * 50)
    
    # Evaluate the code output using LLM judge
    passed = evaluator.evaluate_and_print(stdout, task)
    
    # Cleanup
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"\nCleaned up: {output_file}")
    
    print("\n" + "=" * 50)
    print(f"Final Result: {'PASS' if passed else 'FAIL'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
