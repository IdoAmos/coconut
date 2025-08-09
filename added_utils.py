"""
Utility functions used for datasets not in the source COCONUT repository.
"""
import sys
import os
main_project_path = os.path.join(os.environ["PROJECTS"], "cot")
sys.path.append(main_project_path)  # Adjust path to import from the parent directory

from src.metric.hopping import MetricPipeline

# remove cot from path
sys.path = [p for p in sys.path if "cot" not in p]

class HoppingUtils:

    def __init__(self, inverted=True):
        self.metric_pipeline = MetricPipeline(inverted_task=inverted, tokenizer=None)
        self.task_arg_keys = ["all_answers", "is_1hop"]
        self.num_eval_samples = None
        self.num_train_batches = None # 100

    def prediction_extraction_fn(self, preds: str) -> str:
        preds = preds.split("#")[-1].replace(",", "").strip()
        if len(preds) == 0:
            return None
        return preds

    def check_predictions(self, preds: str, sample_special_vals: dict) -> bool:
        """
        Check if predictions match the labels.
        :param preds: List of predictions.
        :param labels: List of labels.
        :return: List of boolean values indicating if each prediction matches the corresponding label.
        """
        if preds is None or len(preds) == 0:
            return False
        labels = sample_special_vals["all_answers"]
        preds = self.prediction_extraction_fn(preds)
        labels = self.metric_pipeline.label_extraction_fn(labels)
        score = self.metric_pipeline.check_prediciton(preds, labels)
        return score == 1
        

registry = {
    "hopping": HoppingUtils,
}