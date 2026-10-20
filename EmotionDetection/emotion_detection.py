"""Call Watson NLP's emotion classifier and normalize its response."""
import os
import requests
WATSON_EMOTION_URL = os.getenv(
    "WATSON_EMOTION_URL",
    "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/"
    "NlpService/EmotionPredict",
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
REQUEST_TIMEOUT = 15
def emotion_detector(text_to_analyze):
    """Return the five emotion scores and the dominant emotion.
    For an empty input or an HTTP 400 response, return an empty result with
    ``dominant_emotion`` set to ``None``. Other request failures are allowed
    to propagate so the web layer can report a service error.
    """
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
    headers = {
        "grpc-metadata-mm-model-id": MODEL_ID,
        "Content-Type": "application/json",
    }
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(
        WATSON_EMOTION_URL,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
    response.raise_for_status()
    emotions = response.json()["emotionPredictions"][0]["emotion"]
    names = ("anger", "disgust", "fear", "joy", "sadness")
    scores = {name: emotions[name] for name in names}
    scores["dominant_emotion"] = max(names, key=scores.__getitem__)
    return scores
