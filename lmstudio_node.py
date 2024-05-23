import os
from typing import Literal, Optional, Union
import random as r
import re
import requests
import time
from pydantic import Field
import openai
from invokeai.backend.util.devices import choose_torch_device, choose_precision
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
from invokeai.app.invocations.primitives import StringOutput

# Initialize OpenAI client by setting the API key and endpoint
openai.api_key = "lm-studio"
openai.api_base = "http://localhost:1234/v1"

@invocation_output("openai_assistant_output")
class OpenAIAssistantInvocationOutput(BaseInvocationOutput):
    generatedPrompt: str = OutputField(description="The generated prompt")

@invocation("openai_assistant", title="LM Studio API Assistant", tags=["text", "prompt", "openai", "lmstudio", "api", "assistant"], version="1.0.0")
class OpenAIAssistantInvocation(BaseInvocation):
    """LM Studio API Assistant Prompt Generator node"""

    # Inputs
    lmstudioContext: str = InputField(
        default="fill the details in the stable diffusion prompt the user enters. Please be as detailed as possible. Don't ask the user for information, but fill in the blanks for him. Be creative! User prompt :",
        description="context"
    )
    prompt: str = InputField(default="")
    max_tokens: int = InputField(default=2048, description="maximum number of tokens to generate")
    temperature: float = InputField(default=0.8)
    seed: int = InputField(default=-1)
    trigger: int = InputField(default=0, description="Used to trigger the generator without changing values")
    # For local server
    HOST: str = InputField(default="localhost:1234", description="host:port")

    def run(self):
        r.seed()
        random_delay = r.randint(0, 5)
        time.sleep(3 + random_delay)
        
        history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": self.lmstudioContext + "\n" + self.prompt + "\n"},
        ]

        request = {
            'model': "MaziyarPanahi/WizardLM-2-7B-GGUF",
            'messages': history,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'seed': self.seed,
        }
        
        response = openai.ChatCompletion.create(**request)
        generatedPrompt = response.choices[0].message['content']
        print(f"\nGenerated prompt: {generatedPrompt}\nSeed:{self.seed}\nTemp:{self.temperature}\n")
        return generatedPrompt

    def invoke(self, context: InvocationContext) -> OpenAIAssistantInvocationOutput:
        generated_prompt = self.run()
        if generated_prompt is None:
            # Handle the error or set a default value
            generated_prompt = "Default value or error message"
        return OpenAIAssistantInvocationOutput(generatedPrompt=generated_prompt)
