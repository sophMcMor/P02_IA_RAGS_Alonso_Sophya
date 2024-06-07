import requests
from langchain_community.llms.ollama import Ollama

#url = f"http://127.0.0.1:8080/ask_pdf"
url= f"http://localhost:8080/ask_pdf"
EVAL_PROMPT = """
Exprected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def test_one():
    assert query_and_validate(
        question="¿Qué derechos establece la Constitución de Costa Rica?",
        expected_response="Derecho a la vida, derecho a la educación, derecho a la salud, derecho al trabajo."
    )

#Prompt para validar la respuesta dada por el modelo y la respuesta esperada
def query_and_validate(question: str, expected_response: str):
    #Está fallando porque jala el estado de la respuesta y no la respuesta

    response_text = requests.post(url, json = {"query": question})
    
    #response_text --> Imprime <Response [200]>
    print("Response Text: ", response_text)

    #response_text.text --> Imprime {"answer": "blablabla"}
    print("---Response Text.text: ---", response_text.text) 

    #response_text.text["answer"] --> Tira este error:TypeError: string indices must be integers
    print("---Response Text.text[answer]: ---", response_text.text["answer"]) 

    prompt = EVAL_PROMPT.format(
         expected_response=expected_response, actual_response=response_text.text
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
    
test_one()