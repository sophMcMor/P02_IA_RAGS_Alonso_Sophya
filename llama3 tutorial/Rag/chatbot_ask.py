#ASK ChatBot

import requests

# API
URL = "http://localhost:8080/ask_pdf"

while True:
    #Entrada del usuario
    user_input = input(">> ")
    
    #Si el usuario ingresa "chao!", sale del ciclo y termina el programa
    if user_input.lower() == "chao!":
        print("ASK: Adiós!")
        break
    
    query = {"query": user_input}
    
    try:
        #Solicitud a la API del modelo
        response = requests.post(URL, json=query)
        response_data = response.json()
        
        #Nos quedamos sólo con la respuesta, no con el prompt completo
        answer = response_data.get("answer", "No se encontró la respuesta.")
        
        #Se imprime la respuesta (ASK es el chatbot)
        print("ASK:", answer)
    except requests.exceptions.RequestException as e:
        print("Query Error:", e)
