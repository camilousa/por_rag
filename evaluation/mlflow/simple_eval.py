import mlflow
import openai
import pandas as pd
from typing import List
import os
import sys
# Get the absolute path of the directory containing 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(project_root)
# Add the project root to the system path
sys.path.insert(0, project_root)

# Now you can import from src


import mlflow
from src.backend.generator import build_rag_chain, generate_answer
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
	answer, docs = generate_answer(question, k=5, k_candidates=8)
	return answer

mlflow.langchain.autolog()
with mlflow.start_span(name="evaluation"):
	mlflow.genai.evaluate(
        	predict_fn=poe_rag,
	        data=eval_data,
	        scorers=[
	          Correctness()
	]
	)



