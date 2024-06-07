Contenidos de P02_IA_RAGS_Alonso_Garita_Sophya_McLean

- app.py: Programa tipo API Flask que carga y procesa documentos PDF y mediante llama3 genera respuestas con ese contexto. El modelo NO tiene Fine Tuning. Créditos a Thomas Jay.
- chatbot_ask.py: Chatbot que usa el servicio de app.py para generar respuestas con base en un query o pregunta. Se debe ejecutar app.py para poder utilizar el chatbot.
- test_rag.py: Ejemplo de prueba unitaria que se aplica al modelo llama3 de app.py.
- FineTuning_Proy02.ipynb: Cuaderno de Jupyter con el proceso paso a paso del Fine Tuning a un modelo llama3. Créditos a David Ondrej.
- dataset.json: Datos de entrenamiento para el Fine Tuning. El proceso está diseñado para la estructura de este dataset. Se recomienda ejecutarlo en un entorno de Google Colab.
- pdf: Carpeta con dos documentos PDF de ejemplo para introducirlos en los modelos de app.py y de FineTuning_Proy02.ipynb.
- Proyecto02_IA_RAGS.pdf: Reporte y documentación detallada del proceso de desarrollo del proyecto.