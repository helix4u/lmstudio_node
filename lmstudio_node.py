import base64
import random as r
import time
from io import BytesIO
from typing import Optional

import requests

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.primitives import StringOutput

@invocation(
    "openai_assistant",
    title="LM Studio API Assistant",
    tags=["text", "prompt", "lmstudio", "api", "assistant"],
    version="1.0.4",
)
class OpenAIAssistantInvocation(BaseInvocation):
    """LM Studio API Assistant Prompt Generator node"""

    lmstudioContext: str = InputField(
        default=(
            "fill the details in the stable diffusion prompt the user enters. "
            "Please be as detailed as possible. Don't ask the user for information, "
            "but fill in the blanks for him. Be creative! User prompt :"
        ),
        description="Context template",
    )
    prompt: str = InputField(default="", description="User prompt")
    model: str = InputField(
        default="MaziyarPanahi/WizardLM-2-7B-GGUF", description="Model name"
    )
    image: Optional[ImageField] = InputField(
        default=None, description="Optional image input"
    )
    image_prompt: Optional[str] = InputField(
        default=(
            "Describe this image in a very detailed and intricate way, "
            "as if you were describing it to a blind person for accessibility."
        ),
        description="Instruction for image description",
    )
    max_tokens: int = InputField(default=2048, description="Max tokens")
    temperature: float = InputField(default=0.8, description="Temperature")
    seed: int = InputField(default=-1, description="Random seed")
    trigger: int = InputField(default=0, description="Reinvoke trigger")
    HOST: str = InputField(default="localhost:1234", description="Host:port")

    def invoke(self, context: InvocationContext) -> StringOutput:
        # Seed RNG
        if self.seed >= 0:
            r.seed(self.seed)
        else:
            r.seed()

        # Optional delay
        time.sleep(3 + r.randint(0, 5))

        # Build messages
        history = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent assistant. Provide well-reasoned, "
                    "correct, and helpful answers."
                ),
            },
            {"role": "user", "content": f"{self.lmstudioContext}\n{self.prompt}\n"},
        ]

        # Embed image if provided
        if self.image:
            try:
                pil_image = context.images.get_pil(self.image.image_name)
                buf = BytesIO()
                pil_image.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                history.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.image_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                )
            except Exception as e:
                print(f"Image error: {e}")
                return StringOutput(value="Error processing the image")

        # Prepare request
        url = f"http://{self.HOST}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": history,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }

        # Send HTTP request directly
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            prompt = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"HTTP error: {e}")
            prompt = "Error in API call"

        return StringOutput(value=prompt)
