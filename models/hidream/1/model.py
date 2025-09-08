import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from diffusers import HiDreamImagePipeline
from io import BytesIO

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_id = "HiDream-ai/HiDream-I1-Full"  # Example repo

        try:
            self.pipeline = HiDreamImagePipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16
            ).to("cuda")
            self.logger.log_info("HiDream model loaded")
        except Exception as e:
            self.logger.log_error(f"Init error: {str(e)}")
            raise

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                prompt = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                prompt_str = prompt.as_numpy()[0].decode()

                image = self.pipeline(
                    prompt=prompt_str,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]

                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_np = np.frombuffer(img_byte_arr.getvalue(), dtype=np.uint8)

                output_tensor = pb_utils.Tensor("GENERATED_IMAGE", img_np)
                responses.append(pb_utils.InferenceResponse([output_tensor]))

            except Exception as e:
                self.logger.log_error(f"Error: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=str(e)))
        return responses

    def finalize(self):
        self.pipeline = None
        torch.cuda.empty_cache()
