"""
W&B / Weave Tracing & Evaluation Wrapper for Unified Hybrid ALinkNER.
Provides graceful fallbacks when weave is not installed or when offline.
"""

try:
    import weave
    BaseModel = weave.Model
    weave_op = weave.op
except ImportError:
    weave = None
    BaseModel = object
    def weave_op(*args, **kwargs):
        def decorator(f):
            return f
        return decorator if kwargs or not args else decorator(args[0])

from src.common.evaluation import evaluate_one_file


class LSFNEREval(BaseModel):
    """
    Wrap pipeline for Weave evaluation.
    Even if predictions are already computed, this wrapper returns a dict
    compatible with your scorer.
    """
    @weave_op()
    def predict(
        self,
        file_name: str,
        ground_truth_sub: str = "train_phase",
        predicted_BIO_sub: str = None
    ):
        result = evaluate_one_file(
            file_name,
            ground_truth_sub=ground_truth_sub,
            mode="overlap",
            predicted_BIO_sub=predicted_BIO_sub
        )

        output = {
            "processed_tokens": result.get("processed_tokens"),
            "found_correct_phrases": result.get("found_correct_phrases"),
            "found_guessed_phrases": result.get("found_guessed_phrases"),
            "correct_phrases": result.get("correct_phrases"),
            "accuracy": result.get("accuracy"),
        }

        # Add overall metrics
        if "overall" in result:
            output.update({
                "overall_precision": result["overall"]["precision"],
                "overall_recall": result["overall"]["recall"],
                "overall_f1": result["overall"]["f1"],
            })

        # Flatten per-type metrics
        for t, vals in result.get("by_type", {}).items():
            if isinstance(vals, dict):
                for metric_name, metric_val in vals.items():
                    output[f"{t}_{metric_name}"] = metric_val

        return output
