# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import json
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.ai import AIMessage

# Implements a statemachine for chatting with AI. It uses a statemachine definition json file and uses natual language descriptions for branching decisions.
class ChatStateMachine():

    # Constructor.
    def __init__(self, llm, statemachine: dict, chains: dict):
        self.validate(statemachine, chains)

        self.branch_chain = PromptTemplate.from_template(Path(os.path.join(os.path.dirname(__file__), "./prompts/branch.txt")).read_text()) | llm | StrOutputParser()
        self.confirmation_chain = PromptTemplate.from_template(Path(os.path.join(os.path.dirname(__file__), "./prompts/confirmation.txt")).read_text()) | llm | StrOutputParser()
        self.decision_chain = PromptTemplate.from_template(Path(os.path.join(os.path.dirname(__file__), "./prompts/decision.txt")).read_text()) | llm | JsonOutputParser()

        self.statemachine = statemachine
        self.chains = chains

        self.set_state(None)

    # Hydrate this statemachine from a serialized data. This is useful when deploying on to stateless platforms like AWS Lambda.
    def set_state(self, state):
        if state != None:
            state = json.loads(state)
            self.chat_history = [SystemMessage(chat['content']) if chat['type'] == 'system' else HumanMessage(chat['content']) if chat['type'] == 'human' else AIMessage(chat['content']) for chat in state['chat_history']]
            self.chat_history_position = state['chat_history_position']
            self.state_stack = state['state_stack']
            self.confirmation_active = state['confirmation_active']
            self.confirmation_chat_history = [SystemMessage(chat['content']) if chat['type'] == 'system' else HumanMessage(chat['content']) if chat['type'] == 'human' else AIMessage(chat['content']) for chat in state['confirmation_chat_history']]
            self.pending_branch = state['pending_branch']
        else:
            self.chat_history = [SystemMessage(self.statemachine["system_message"])]
            self.chat_history_position = 0
            self.state_stack = [self.statemachine["initial_state"]]

            self.confirmation_active = False
            self.confirmation_chat_history = [SystemMessage(self.statemachine["system_message"])]
            self.pending_branch = None

    # Get a serialized representation of this statemachine's current state.
    def get_state(self):
        return json.dumps({
            'chat_history': [{'type': chat.type, 'content': chat.content} for chat in self.chat_history],
            'chat_history_position': self.chat_history_position,
            'state_stack': self.state_stack,
            'confirmation_active': self.confirmation_active,
            'confirmation_chat_history': [{'type': chat.type, 'content': chat.content} for chat in self.confirmation_chat_history],
            'pending_branch': self.pending_branch
        })
    
    # Get all messages.
    def get_messages(self):
        return self.chat_history[1:] + self.confirmation_chat_history[1:]

    # Process a new message. This message as well as AI's reply will be added to the set of all messages.
    def send_message(self, message: str):
        if self.confirmation_active:
            self.confirmation_chat_history.append(HumanMessage(message))
        else:
            self.chat_history.append(HumanMessage(message))

        print(f'Current state: {self.state_stack[-1]}')

        # A state change confirmation phase is active.
        if self.confirmation_active:
            # Get human confirmation.
            arguments = {
                "current_task": self.statemachine["states"][self.state_stack[-1]]["description"],
                "next_task": self.statemachine["states"][self.pending_branch["next"]]["description"] if self.pending_branch != None else None,
                "switching_reason": self.pending_branch["condition"] if self.pending_branch != None else None,
                "chat_history": self.format_chat_history(self.confirmation_chat_history)
            }
            reply = self.decision_chain.invoke(arguments)
            self.confirmation_chat_history.append(AIMessage(reply["message"]))
            response = reply["message"]

            # Human wants to proceed with the state change.
            if reply["action"] == "PROCEED":
                print(f"Proceeding to state {self.pending_branch['next']} after confirmation.")
                if self.pending_branch["next"] == "__previous_state__":
                    self.state_stack.pop()
                else:
                    self.state_stack.append(self.pending_branch["next"])

                self.chat_history_position = len(self.chat_history)
                self.confirmation_chat_history = []
                self.confirmation_active = False
            
            if reply["action"] == "CANCEL":
                self.chat_history_position = len(self.chat_history)
                self.confirmation_chat_history = []
                self.confirmation_active = False
        
            # Human confirmation phase has completed. This block must be executed after the state change above to ensure availability of pending_branch.
            elif reply["action"] != "QUESTION":
                self.clear_chat_history(self.confirmation_chat_history, "ALL")
                self.pending_branch = None
                self.confirmation_active = False

        # We are not in a confirmation phase.
        else:
            while True:
                # Execute all 'before' branches required.
                while self.change_state(False) != None:
                    pass

                if self.confirmation_active:
                    self.confirmation_chat_history = [HumanMessage(message)]

                    # Get human confirmation.
                    arguments = {
                        "current_task": self.statemachine["states"][self.state_stack[-1]]["description"],
                        "next_task": self.statemachine["states"][self.pending_branch["next"]]["description"] if self.pending_branch != None else None,
                        "switching_reason": self.pending_branch["condition"] if self.pending_branch != None else None,
                        "chat_history": self.format_chat_history(self.confirmation_chat_history)
                    }
                    reply = self.confirmation_chain.invoke(arguments)
                    self.confirmation_chat_history = [AIMessage(reply)]
                    response = reply
                    break

                # Wait for next human message.
                if self.statemachine["states"][self.state_stack[-1]].get("needs_human_input", True) and message == None:
                    print("Wait for human input.")
                    break

                # Invoke current state.
                arguments = {
                    "chat_history": self.format_chat_history(self.chat_history),
                }
                response = self.invoke_chain(self.chat_history, arguments, self.chains[self.state_stack[-1]])
                message = None

                # Execute a single 'after' branch.
                if self.change_state(True) == None:
                    break

        return response
    
    # Executes a state change for both before and after branch types. It also handles the flags such as that for confirmation.
    def change_state(self, afterwards):
        print("Executing ", "(after)" if afterwards else "(before)", "for state: ", self.state_stack[-1])

        # Prepare state change branches.
        available_branches = []
        unconditional_branch = None
        for branch in self.statemachine["states"][self.state_stack[-1]]["branches"]:
            if not (afterwards ^ branch.get("afterwards", False)):
                if "condition" not in branch:
                    available_branches = None
                    unconditional_branch = branch
                    break

                available_branches.append(branch)
        print('available_branches, unconditional_branch', available_branches, unconditional_branch)

        # Identify the final selected branch.
        selected_branch = None
        if unconditional_branch != None:
            selected_branch = unconditional_branch

        elif self.chat_history_position >= len(self.chat_history):
            print('Not branching as there has not been any conversations yet in this state.')
            selected_branch = None

        elif len(available_branches) > 0:
            # Prepare state change options for the prompt.
            options = ""
            for index, branch in enumerate(available_branches):
                options += f"{index+1}. {branch['condition']}\n"

            options += f'{len(available_branches)+1}. None of the above.'

            # Invoke branch selection chain.
            arguments = {
                "current_task": self.statemachine["states"][self.state_stack[-1]]["description"],
                "chat_history": self.format_chat_history(self.chat_history[self.chat_history_position:]),
                "branches": options,
            }
            branch_number = int(self.branch_chain.invoke(arguments)) - 1
            
            if branch_number < len(available_branches):
                selected_branch = available_branches[branch_number]

        # Change the state.
        if selected_branch != None:
            if selected_branch.get("confirmation") == True:
                self.confirmation_active = True
                self.pending_branch = selected_branch
                selected_branch = None

            else:
                if selected_branch["next"] == "__previous_state__":
                    self.state_stack.pop()
                else:
                    self.state_stack.append(selected_branch["next"])

                self.chat_history_position = len(self.chat_history)

        print('selected_branch', selected_branch)
        return selected_branch
    
    # Helper function to format the chat to be placed into prompts.
    def format_chat_history(self, chat_history):
        return "\n".join([f'{"Human" if chat.type == "human" else "System" if chat.type == "system" else "Assistant"}: {chat.content}' for chat in chat_history])
   
    # Helper function to invoke LLM and parse the response.
    def invoke_chain(self, chat_history, arguments, chain):
        reply = chain.invoke(arguments)
        
        # Handle different kinds of LLM responses, as it can vary depending on the actual chain that gets invoked.
        response = None
        if isinstance(reply, str):
            chat_history.append(AIMessage(reply))
        elif "output" in reply:
            chat_history.append(AIMessage(reply["output"]))
        else:
            chat_history.append(AIMessage(json.dumps(reply)))

        if isinstance(reply, str):
            response = reply
        elif "output" in reply:
            response = reply["output"]
        elif "message" in reply:
            response = reply["message"]

        return response
        
    # Helper function to clear chat history in various ways.
    def clear_chat_history(self, chat_history, type):
        if type == "ALL":
            chat_history.clear()
            chat_history.append(SystemMessage(self.statemachine["system_message"]))

        elif type == "RETAIN_LAST_HUMAN_MESSAGE":
            for message in reversed(chat_history):
                if isinstance(message, HumanMessage):
                    chat_history.clear()
                    chat_history.append(SystemMessage(self.statemachine["system_message"]))
                    chat_history.append(message)
                    break

        elif type == "RETAIN_LAST_AI_MESSAGE":
            for message in reversed(chat_history):
                if isinstance(message, AIMessage):
                    chat_history.clear()
                    chat_history.append(SystemMessage(self.statemachine["system_message"]))
                    chat_history.append(message)
                    break

        elif type == "RETAIN_LAST_AI_AND_HUMAN_MESSAGES":
            human_message = None
            for message in reversed(chat_history):
                if isinstance(message, HumanMessage):
                    human_message = message
                    break

            ai_message = None
            for message in reversed(chat_history):
                if isinstance(message, AIMessage):
                    ai_message = message
                    break

            chat_history.clear()
            chat_history.append(SystemMessage(self.statemachine["system_message"]))

            if human_message != None:
                chat_history.append(human_message)

            if ai_message != None:
                chat_history.append(ai_message)

        elif type == "DROP_LAST_HUMAN_MESSAGE":
            for message in reversed(chat_history):
                if isinstance(message, HumanMessage):
                    chat_history.remove(message)
                    break

        elif type == "DROP_LAST_AI_MESSAGE":
            for message in reversed(chat_history):
                if isinstance(message, AIMessage):
                    chat_history.remove(message)
                    break

        elif type == "DROP_LAST_AI_AND_HUMAN_MESSAGES":
            for message in reversed(chat_history):
                if isinstance(message, HumanMessage):
                    chat_history.remove(message)
                    break

            for message in reversed(chat_history):
                if isinstance(message, AIMessage):
                    chat_history.remove(message)
                    break

    # Validates the statemachine definition.
    def validate(self, statemachine: dict, chains: dict):
        if "name" not in statemachine:
            raise KeyError("name not present in the statemachine definition.")
        if "system_message" not in statemachine:
            raise KeyError("system_message not present in the statemachine definition.")
        if "states" not in statemachine:
            raise KeyError("states not present in the statemachine definition.")
        if "initial_state" not in statemachine:
            raise KeyError("initial_state not present in the statemachine definition.")
        if len(statemachine["states"]) == 0:
            raise ValueError("Atleast one state must be present in the statemachine definition.")
        