import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(".") / "envs/.env"
load_dotenv(dotenv_path=env_path)
TRITON_SERVER_URL = os.environ.get('TRITON_SERVER_URL')
MODEL_NAME = os.environ.get('MODEL_NAME')


class TritonClient:
    def __init__(self) -> None:
        self.client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)

    def send_request_to_bert_classifier(self, input_string: str):
        numpy_input = np.array([bytes(input_string, encoding='utf-8')])

        inputs = [
            grpcclient.InferInput('CONTENT', numpy_input.shape, "BYTES")
        ]

        inputs[0].set_data_from_numpy(numpy_input)

        response = self.client.infer(model_name=MODEL_NAME, inputs=inputs)

        label = response.as_numpy('OUTPUT')
        prob = response.as_numpy('PROB')

        decoded_label = [output.decode('utf-8') for output in label.flatten()]

        return decoded_label[0], prob[0]
    
if __name__ == "__main__":
    client = TritonClient()
    print(client.send_request_to_bert_classifier("Edge of Tomorrow clever, funny action"))