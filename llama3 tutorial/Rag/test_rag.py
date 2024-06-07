import requests
from langchain_community.llms.ollama import Ollama

url= f"http://localhost:8080/ask_pdf"

EVAL_PROMPT = """
Exprected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? Even if it is more complex or detailed.
"""

def test_one():
    assert query_and_validate(
        question="¿Qué derechos establece la Constitución de Costa Rica?",
        expected_response="Derecho a la vida, derecho a la educación, derecho a la salud, derecho al trabajo."
    )

#Prompt para validar la respuesta dada por el modelo y la respuesta esperada
def query_and_validate(question: str, expected_response: str):
    #Envía el query de prueba a la API del modelo
    response_text = requests.post(url, json = {"query": question})

    #Creamos el prompt de la evaluación
    prompt = EVAL_PROMPT.format(
         expected_response=expected_response, actual_response=response_text.text
    )

    #Modelo auxiliar para comparar la respuesta obtenida y la respuesta esperada
    model = Ollama(model="llama3")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    #Se imprime el prompt para ver la respuesta obtenida y la esperada
    prompt_decoded = prompt.encode().decode('unicode_escape')
    print(prompt_decoded)

    #Resultado de la evaluación
    if "true" in evaluation_results_str_cleaned:
        #Imprime en verde si la respuesta fue correcta
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        #Imprime en rojo si la respuesta fue correcta
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

#Ejecución de la prueba  
test_one()