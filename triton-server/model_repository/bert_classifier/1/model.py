import json
import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import AutoTokenizer
import torch
        

class TritonPythonModel:

    def initialize(self, args):
        print("start...")
        self._index = 0

        self.model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT")

        self.news_classifier = torch.jit.load("models/bert_classifier/1/checkpoint/jitted_cls_bert_base_uncase_upsample.pt")
        self.news_classifier.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("models/bert_classifier/1/tokenizer", local_files_only=True)
        self.label_mapping = {
            0: "business",
            1: "entertainment",
            2: "health",
            3: "science and technology"
        }
    def convert_tensor_to_string(self, input_tensor):
        return str(input_tensor.as_numpy().tolist()[0].decode("UTF-8"))

    def convert_string_to_output(self, list_output_str):
        null_chars_array = np.array(
                [output_str.encode('utf-8') for output_str in list_output_str], dtype=np.object_)
        null_char_data = null_chars_array.reshape([1, -1])
        out_tensor_0 = pb_utils.Tensor(
            "OUTPUT",
            null_char_data.astype(np.bytes_))
        return out_tensor_0

    def execute(self, requests):

        responses = []

        for request in requests:

            content_tensor = pb_utils.get_input_tensor_by_name(request, "CONTENT")
            content = self.convert_tensor_to_string(content_tensor)
            input_ids = self.tokenizer([content], return_tensors='pt', padding='max_length', truncation=True, max_length=30)#.to('cuda')

            pred = self.news_classifier(input_ids["input_ids"],input_ids["attention_mask"])
            probs = torch.softmax(pred[0], dim=0)
            label = torch.argmax(probs).cpu().item()

            print(self.label_mapping[label])
            out_tensor_0 = self.convert_string_to_output([self.label_mapping[label]])
            out_tensor_1 = pb_utils.Tensor("PROB", np.array([probs[label].item()]))
            inference_response = pb_utils.InferenceResponse([out_tensor_0, out_tensor_1])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')
