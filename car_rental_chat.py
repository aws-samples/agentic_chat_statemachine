# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import json
import boto3
from typing import Dict, List, Any, Union
from pathlib import Path

from langchain.schema.output import LLMResult
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import BedrockChat

from agentic_chat_statemachine import ChatStateMachine

# Callback handlder for logging.
class LlmCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>>LLM Prompt\n{prompts[0]}\n<<<<<<<<<<<<<<<<<<<<<<<<<<LLM Prompt\n"
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>>LLM Result\n{response.generations[0][0].text}\n<<<<<<<<<<<<<<<<<<<<<<<<<<LLM Result\n"
        )

    def on_llm_error( self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>>LLM Error\n{error, Any}\n<<<<<<<<<<<<<<<<<<<<<<<<<<LLM Error\n"
        )

# Prepare Bedrock LLM.
region = os.environ["BEDROCK_REGION"]
model_id = os.environ["BEDROCK_MODEL_ID"]
model_kwargs = {
    "max_tokens": 2048,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "anthropic_version": "bedrock-2023-05-31",
}
max_agent_iterations = 500

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
)
llm = BedrockChat(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
    callbacks=[LlmCallbackHandler()],
).bind(stop=['Human:', '\nObservation:'])

# Prepare chains for the car rental usecase.
initial_chain = PromptTemplate.from_template(Path(os.path.join(os.path.dirname(__file__), "car_rental_prompts/initial.txt")).read_text()) | llm | StrOutputParser()
car_details_chain = PromptTemplate.from_template(Path(os.path.join(os.path.dirname(__file__), "car_rental_prompts/car_details.txt")).read_text()) | llm | StrOutputParser()
booking_chain = PromptTemplate.from_template(Path(os.path.join(os.path.dirname(__file__), "car_rental_prompts/booking.txt")).read_text()) | llm | StrOutputParser()
post_booking_chain = PromptTemplate.from_template(Path(os.path.join(os.path.dirname(__file__), "car_rental_prompts/post_booking.txt")).read_text()) | llm | StrOutputParser()
cancellation_chain = PromptTemplate.from_template(Path(os.path.join(os.path.dirname(__file__), "car_rental_prompts/cancellation.txt")).read_text()) | llm | StrOutputParser()

# Create the chat statemachine.
statemachine = ChatStateMachine(llm, json.load(open(os.path.join(os.path.dirname(__file__), "car_rental_statemachine.json"))), {
    "initial": initial_chain,
    "car_details": car_details_chain,
    "booking": booking_chain,
    "post_booking": post_booking_chain,
    "cancellation": cancellation_chain
})

print("Enter your message (or 'exit' to quit): ")
while True:
    # Get user input.
    user_input = input("Customer:")
    if user_input.lower() == "exit":
        break

    # Process the user input.
    response = statemachine.send_message(user_input)
    print("\nResponse:", response)

print("Exiting...")
