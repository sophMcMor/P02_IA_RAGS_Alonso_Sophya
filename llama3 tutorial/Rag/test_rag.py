import requests
from langchain_community.llms.ollama import Ollama

url = f"http://127.0.0.1:8080/ask_pdf"

EVAL_PROMPT = """
Exprected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def test_whatever():
    assert query_and_validate(
        question="¿Qué derechos establece la Constitución de Costa Rica?",
        expected_response="Derecho a la vida, derecho a la educación, derecho a la salud, derecho al trabajo."
    )

def query_and_validate(question: str, expected_response: str):
    response_text = requests.post(url, json = {"query": question})
    prompt = EVAL_PROMPT.format(
         expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
    
test_whatever()