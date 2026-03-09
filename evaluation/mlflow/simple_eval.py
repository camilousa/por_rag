import mlflow
import openai
import pandas as pd
from typing import List
import os
from dotenv import load_dotenv
from mlflow.genai.scorers import Correctness, Guidelines

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("simple eval")


eval_data = pd.read_json("ground_truth_dataset.json")


@mlflow.trace
def poe_rag(question: str, context: str) -> str:
	predictions = []
#	print(inputs)
#	for _, row in inputs.iterrows():
#		predictions.append("No sé")
	return "No sé"

mlflow.langchain.autolog()
with mlflow.start_span(name="evaluation"):
	mlflow.genai.evaluate(
        	predict_fn=poe_rag,
	        data=eval_data,
	        scorers=[
	          Correctness()
	]
	)



