import sys
import os

# Get the absolute path of the directory containing 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(project_root)
# Add the project root to the system path
sys.path.insert(0, project_root)

# Now you can import from src


import mlflow
from src.backend.generator import build_rag_chain
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.set_experiment("poe track")

rag_chain, retriever = build_rag_chain(k_candidates=8)



with mlflow.start_run() as run:
  model_info = mlflow.langchain.log_model(
      lc_model="chain.py",
      name="rag_chain",
      loader_fn=retriever,
  )
