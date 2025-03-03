import mlflow.exceptions
import mlflow.metrics
import mlflow
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from mlflow.metrics.genai.metric_definitions import relevance, faithfulness
from dataclasses import dataclass
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from loguru import logger
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
import os

os.environ["HF_TOKEN"] = ""


class GPT2Model:
    def __init__(self, model_name="openai-community/gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def log_to_mlflow(self):
        with mlflow.start_run(run_name='Model') as run:
            mlflow.set_experiment("SE_LLM_Experiment")
            mlflow.transformers.log_model(
                transformers_model={"model": self.model, "tokenizer": self.tokenizer},
                artifact_path="gpt2_model",
                task="text-generation"
            )
            logger.info("Model logged to MLflow.")

    def setup_llm(self):
        pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=1024,
        )
        gp2_llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("LLM setup completed.")
        return gp2_llm



class ElasticsearchRetrievalManager:
    def __init__(self):
        self.bgeEmbeddings = self.load_embedding_model()

    def load_embedding_model(self):
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}

        bgeEmbeddings = HuggingFaceBgeEmbeddings(
                                     model_name=model_name,
                                     model_kwargs=model_kwargs,
                                     encode_kwargs=encode_kwargs)
        logger.info("Embedding model loaded.")
        return bgeEmbeddings
    
    def connect_elasticSearch(self):
        try:
            logger.debug(f"Trying to connect DBEmbeddingsStore at: {'http://localhost:9200'}")

            elastic_vector_search = ElasticsearchStore(
                es_url="http://localhost:9200",
                index_name="skyelectric_docs_embeddings",
                embedding=self.bgeEmbeddings,
            )
            logger.info("Connected to Elasticsearch.")
            return elastic_vector_search

        except ConnectionError as ce:
            logger.error(
                f"ConnectionError: Could not establish connection to Elasticsearch. Reason: {ce}, closing server"
            )
            exit()

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}, closing server")
            exit()


class RetrieverPartEvaluation:
    def __init__(self, elastic_vector_search):
        self.retriever = elastic_vector_search.as_retriever(search_type = "similarity")

    def retrieve_doc_ids(self, question):
        docs = self.retriever.get_relevant_documents(question)
        return [os.path.basename(doc.metadata["source"]) for doc in docs]

    def retriever_model_function(self, question_df):
        return question_df["question"].apply(self.retrieve_doc_ids)
    
    def evaluate_retriever(self, eval_data):
        with mlflow.start_run(run_name="retriever part evaluation with different values") as run:
            mlflow.search_experiments("SE_LLM_Experiment")
            evaluate_results = mlflow.evaluate(
                model=self.retriever_model_function,
                data=eval_data,
                targets="source",
                evaluators="default",
                extra_metrics=[
                    mlflow.metrics.precision_at_k(1),
                    mlflow.metrics.precision_at_k(2),
                    mlflow.metrics.precision_at_k(3),
                    mlflow.metrics.recall_at_k(1),
                    mlflow.metrics.recall_at_k(2),
                    mlflow.metrics.recall_at_k(3),
                    mlflow.metrics.ndcg_at_k(1),
                    mlflow.metrics.ndcg_at_k(2),
                    mlflow.metrics.ndcg_at_k(3),
                ],
            )
            logger.info("Retriever evaluation completed.")
            return evaluate_results.tables["eval_results_table"]


class PhiModel:
    def __init__(self, model_name="microsoft/phi-1_5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model_end_point = None

    def log_to_mlflow(self):
        with mlflow.start_run(run_name='Phi-1.5 model') as run:
            mlflow.transformers.log_model(
                transformers_model={"tokenizer": self.tokenizer, "model": self.model},
                artifact_path="Phi-1.5 Model",
            )
            self.model_end_point = f"runs:/{run.info.run_id}/Phi-1.5 Model"
            logger.info("Phi-1.5 model logged to MLflow.")

    @staticmethod
    def Faithfulness_Examples():  
        faithfulness_examples = [
            EvaluationExample(
                input="How does the NGM app facilitate system initialization?",
                output="The NGM app facilitates system initialization by providing configurations to components and ensuring communication channels are established.",
                score=5,
                justification="The output accurately describes how the NGM app facilitates system initialization, matching the information provided in the context.",
                grading_context={
                    "context": "The NGM app is responsible for communicating with various components such as the BMS, Cloud, HV Control Board, and Inverter. It provides configurations for initial setup and ensures communication channels are established before the system starts functioning."
                },
            ),
            EvaluationExample(
                input="What is the role of the BMS Manager?",
                output="The BMS Manager is responsible for handling all NGM based actions related to the battery and cloud. It communicates with all BMSes attached to the system, reads data such as pack current and voltage, and processes this data.",
                score=2,
                justification="While the output provides some information about the BMS Manager, it lacks detail and does not fully capture its role as described in the context.",
                grading_context={
                    "context": "The BMS Manager serves as a crucial link between higher-level actions related to the battery and cloud interactions. It communicates with all BMSes attached to the system, reads data such as pack current and voltage, and processes this data before passing relevant information to other processes."
                },
            ),
        ]
        
        return faithfulness_examples

    def calculate_relevance_metric(self):
        relevance_metric = relevance(model=self.model_end_point)
        return relevance_metric
    
    def calculate_faithfulness_metric(self):
        examples = self.Faithfulness_Examples()
        faithfulness_metric = faithfulness(
                model=self.model_end_point, examples=examples
            )
        return faithfulness_metric



@dataclass
class EvaluationExample:
    input: str
    output: str
    score: int
    justification: str
    grading_context: dict
    


class ModelEvaluator:
    def __init__(self, es_obj, faithfulness_metric, relevance_metric, eval_df, gp2_llm):
        self.relevance_metric = relevance_metric
        self.faithfulness_metric = faithfulness_metric
        self.eval_df = eval_df
        self.es_obj = es_obj
        self.gp2_llm = gp2_llm
        self.chain=self.get_chain()

    def get_chain(self):
        chain = RetrievalQA.from_chain_type(
                llm=self.gp2_llm,
                chain_type="stuff",
                retriever=self.es_obj.as_retriever(fetch_k=4),
                return_source_documents=True,
            )
        return chain
    
    def model(self,input_df):
        return input_df["questions"].map(self.chain).tolist()
    
    def evaluate_model(self):
        results = mlflow.evaluate(
            model=self.model,  
            data=self.eval_df,
            model_type="question-answering",
            evaluators="default",
            predictions="result",
            extra_metrics=[self.faithfulness_metric, self.relevance_metric, mlflow.metrics.latency()],
            evaluator_config={"col_mapping": {"inputs": "questions", "context": "source_documents"}},
        )
        logger.info("Model evaluation completed.")
        return results.metrics, results.tables["eval_results_table"]

        
def main():
    gpt2_model = GPT2Model()
    gpt2_model.log_to_mlflow()
    gp2_llm = gpt2_model.setup_llm()

    retrieval_manager = ElasticsearchRetrievalManager()
    es_obj=retrieval_manager.connect_elasticSearch()
    evaluation_manager = RetrieverPartEvaluation(es_obj)
    
    eval_data = pd.read_csv("retriever_part_sample_data.csv", converters={"source": eval})

    evaluate_results = evaluation_manager.evaluate_retriever(eval_data)


    phi_model = PhiModel()
    phi_model.log_to_mlflow()
    calculate_faithfulness_metric = phi_model.calculate_faithfulness_metric()
    calculate_relevance_metric = phi_model.calculate_relevance_metric()

    eval_df = pd.DataFrame(
        {
        "questions": [
        "When will the lab heating was started early morning at approximately?",
        "Explore the interactions between the NGM app, BMS Manager, and Inverter Manager in ensuring efficient communication and data exchange across various system components, highlighting their roles in initialization and operational control.",
        "How do the Cloud Manager, WiFi Wizard, and Database Manager collaborate to maintain seamless connectivity and data synchronization between the local system and the cloud, considering their respective responsibilities and interactions?",
            ],
        }
    )
    
    eval_df.to_csv("GPart_questions_sample.csv")
    eval_df = pd.read_csv("GPart_questions_sample.csv")

    model_evaluator = ModelEvaluator(es_obj, calculate_faithfulness_metric, calculate_relevance_metric, eval_df, gp2_llm)
    metrics, eval_results_table = model_evaluator.evaluate_model()
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Evaluation Results Table: {eval_results_table}")


if __name__ == "__main__":
    main()
