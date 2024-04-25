from fastapi import FastAPI
from triton_client import TritonClient

app  = FastAPI()

client = TritonClient()

@app.get("/list_label")
async def list_label():
    return ["business", "science and technology", "entertainment", "health"]

@app.post("/classify")
async def classify(input_string: str):
    response = client.send_request_to_bert_classifier(input_string)
    return {
        'text': input_string,
        'predicted_label': response[0],
        'prob': response[1]
    }
