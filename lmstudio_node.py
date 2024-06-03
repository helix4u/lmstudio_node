import os
import base64
from typing import Optional
import random as r
import time
from io import BytesIO
from pydantic import Field
import openai
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
    invocation_output,
    InputField,
    OutputField,
    UIComponent
)
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin

# Initialize OpenAI client by setting the API key and endpoint
openai.api_key = "lm-studio"
openai.api_base = "http://localhost:1234/v1"

@invocation_output("openai_assistant_output")
class OpenAIAssistantInvocationOutput(BaseInvocationOutput):
    generatedPrompt: str = OutputField(description="The generated prompt")

@invocation("openai_assistant", title="LM Studio API Assistant", tags=["text", "prompt", "openai", "lmstudio", "api", "assistant"], version="1.0.1")
class OpenAIAssistantInvocation(BaseInvocation):
    """LM Studio API Assistant Prompt Generator node"""

    # Inputs
    lmstudioContext: str = InputField(
        default="fill the details in the stable diffusion prompt the user enters. Please be as detailed as possible. Don't ask the user for information, but fill in the blanks for him. Be creative! User prompt :",
        description="context"
    )
    prompt: str = InputField(default="")
    image: Optional[ImageField] = InputField(default=None, description="The image file input")
    max_tokens: int = InputField(default=2048, description="Maximum number of tokens to generate")
    temperature: float = InputField(default=0.8)
    seed: int = InputField(default=-1)
    trigger: int = InputField(default=0, description="Used to trigger the generator without changing values")
    # For local server
    HOST: str = InputField(default="localhost:1234", description="Host:port")

    def run(self, context: InvocationContext):
        r.seed()
        random_delay = r.randint(0, 5)
        time.sleep(3 + random_delay)

        history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": self.lmstudioContext + "\n" + self.prompt + "\n"},
        ]

        if self.image:
            try:
                # Get the PIL image using context._services.images.get_pil_image
                pil_image = context._services.images.get_pil_image(self.image.image_name)
                
                # Convert the PIL image to bytes and then to a base64 string
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                history.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What’s in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                )
            except Exception as e:
                print(f"Couldn't process the image. Error: {e}")
                return "Error processing the image"

        request = {
            'model': "xtuner/llava-phi-3-mini-gguf",
            'messages': history,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'seed': self.seed,
        }

        try:
            response = openai.ChatCompletion.create(**request)
            generatedPrompt = response.choices[0].message['content']
            print(f"\nGenerated prompt: {generatedPrompt}\nSeed:{self.seed}\nTemp:{self.temperature}\n")
            return generatedPrompt
        except Exception as e:
            print(f"Error in API call: {e}")
            return "Error in API call"

    def invoke(self, context: InvocationContext) -> OpenAIAssistantInvocationOutput:
        generated_prompt = self.run(context)
        if generated_prompt is None:
            # Handle the error or set a default value
            generated_prompt = "Default value or error message"
        return OpenAIAssistantInvocationOutput(generatedPrompt=generated_prompt)
