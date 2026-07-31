import pandas as pd

from src.app.predictor import predict_sentiment


def predict_csv(
    df: pd.DataFrame,
    text_column: str = "text",
) -> pd.DataFrame:

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in CSV."
        )

    results = []

    for text in df[text_column]:

        if pd.isna(text) or not str(text).strip():

            results.append(
                {
                    "prediction": "Invalid",
                    "confidence": 0.0,
                    "confidence_level": "N/A",
                    "inference_time_ms": 0.0,
                }
            )

            continue

        result = predict_sentiment(
            str(text)
        )

        results.append(
            {
                "prediction": result["label"],
                "confidence": result["confidence"],
                "confidence_level": result["confidence_level"],
                "inference_time_ms": result[
                    "inference_time_ms"
                ],
            }
        )

    result_df = pd.DataFrame(results)

    output_df = df.copy()

    output_df["prediction"] = result_df[
        "prediction"
    ]

    output_df["confidence"] = (
        result_df["confidence"] * 100
    ).round(2)

    output_df["confidence_level"] = result_df[
        "confidence_level"
    ]

    output_df["inference_time_ms"] = result_df[
        "inference_time_ms"
    ].round(2)

    return output_df