import json
import gymnasium as gym
from openai import OpenAI
import subprocess

# CONFIGURAZIONE

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lmstudio",
    max_retries=0
)

MODEL = "google/gemma-4-12b-qat"


SYSTEM_PROMPT = (
    "Solve the problem using BFS, maintaining a queue, visited states "
    "and the first parent of each discovered state. "
    "Use only transitions returned by the tools. "
    "Skip terminal successors unless they are the goal. "
    "Consider actions in numerical order. Keep reasoning concise. "
    "When the goal is discovered, reconstruct the path and immediately "
    "call submit_solution. Do not repeatedly recheck the solution."
)

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode="human"
)


# FUNZIONI ACCESSIBILI AL MODELLO

def get_environment():
    return {
        "initial_state": 0,
        "goal_state": 15,
        "num_states": 16,
        "actions": {"0": "LEFT", "1": "DOWN", "2": "RIGHT", "3": "UP"},
        "step_cost": 1,
        "description": (
            "Deterministic 4x4 grid. Terminal states other than the goal "
            "are holes. Use get_successors to discover transitions."
        )
    }


def get_successors(state):
    if type(state) is not int or not 0 <= state < 16:
        return {"error": "state must be an integer between 0 and 15"}

    successors = []
    for action, transitions in env.unwrapped.P[state].items():
        for _, next_state, reward, terminated in transitions:
            successors.append({
                "action": int(action),
                "next_state": int(next_state),
                "cost": 1,
                "terminated": bool(terminated),
                "reward": float(reward)
            })

    return {"state": state, "successors": successors}


# Tools disponibili per il modello
tools = [
    {
        "type": "function",
        "name": "get_environment",
        "description": "Get the initial state, goal and problem definition.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "get_successors",
        "description": "Get all transitions from a state.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "state": {"type": "integer"},
            },
        },
    },
    {
        "type": "function",
        "name": "submit_solution",
        "description": "Submit the ordered list of states from initial state to goal.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        },
    },
]



# RICERCA: IL MODELLO DECIDE, PYTHON ESEGUE LE FUNZIONI

def run_search():
    inputs = [{"role": "user", "content": "Solve the FrozenLake problem."}]
    previous_id = None

    functions = {
        "get_environment": get_environment,
        "get_successors": get_successors,
    }

    while True:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=inputs,
            previous_response_id=previous_id,
            tools=tools,
            store=True,
        )

        print(response.output_text)

        previous_id = response.id
        inputs = []

        for call in response.output:
            if call.type != "function_call":
                continue

            print(f"{call.name}({call.arguments})")
            args = json.loads(call.arguments)

            if call.name == "submit_solution":
                return args["path"]

            result = functions[call.name](**args)
            print(f"Risultato: {result}")

            inputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            })

        if not inputs:
            return response.output_text


# MAIN

if __name__ == "__main__":
    try:
        path = run_search()
        print("Percorso del modello:", path)

        with open("llm_result.json", "w", encoding="utf-8") as file:
            json.dump({"path": path}, file, indent=2)


    finally:
        env.close()
        client.close()
        subprocess.run(
            ["lms", "unload", MODEL],
            check=True,
            timeout=30,
        )