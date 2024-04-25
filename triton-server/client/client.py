import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

TRITON_SERVER_URL = 'localhost:8000'
MODEL_NAME = 'bert_classifier'


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