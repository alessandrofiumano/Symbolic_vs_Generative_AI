import os
import json

import gymnasium as gym
from openai import OpenAI


# ============================================================
# CONFIGURAZIONE
# ============================================================

client = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"]
)
MODEL = os.environ["LLM_MODEL"]

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

MAX_ITERATIONS = 100

# Qui descrivi la strategia (BFS, DFS, ...) e le regole di tie-breaking
SYSTEM_PROMPT = (
    "You are a search agent. Solve the given search problem using BFS. "
    "Use the tools to get the problem definition and the successors of states. "
    "Do the search yourself, keeping frontier and visited states. "
    "When finished, call submit_solution with the ordered list of states "
    "from the initial state to the goal."
)

USER_PROMPT = "Solve the FrozenLake search problem."


# ============================================================
# FUNZIONI DELL'AMBIENTE
# ============================================================

def get_environment():
    return {
        "initial_state": 0,
        "goal_state": 15,
        "num_states": int(env.observation_space.n),
        "actions": {"0": "LEFT", "1": "DOWN", "2": "RIGHT", "3": "UP"},
        "step_cost": 1,
        "description": (
            "4x4 grid, deterministic transitions. Some cells are holes "
            "(terminal, to be avoided). Use get_successors to discover "
            "transitions."
        )
    }


def get_successors(state):
    state = int(state)
    if not 0 <= state < env.observation_space.n:
        return {"error": "Invalid state"}

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


# ============================================================
# TOOL
# ============================================================

tools = [
    {
        "type": "function",
        "name": "get_environment",
        "description": "Get the problem definition: initial state, goal, actions.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_successors",
        "description": "Get all transitions from a state.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"state": {"type": "integer"}},
            "required": ["state"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "submit_solution",
        "description": (
            "Submit the final path as an ordered list of states, "
            "initial state to goal."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "array", "items": {"type": "integer"}}
            },
            "required": ["path"],
            "additionalProperties": False
        }
    }
]


# ============================================================
# LOOP
# ============================================================

def run_search():
    next_input = [{"role": "user", "content": USER_PROMPT}]
    previous_response_id = None

    for _ in range(MAX_ITERATIONS):
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=next_input,
            previous_response_id=previous_response_id,
            tools=tools,
            tool_choice="required",
            store=True
        )

        previous_response_id = response.id
        next_input = []
        path = None

        for call in response.output:
            if call.type != "function_call":
                continue

            args = json.loads(call.arguments or "{}")

            if call.name == "get_environment":
                result = get_environment()
            elif call.name == "get_successors":
                result = get_successors(args["state"])
            elif call.name == "submit_solution":
                path = args["path"]
                result = {"status": "received"}
            else:
                result = {"error": "Unknown tool"}

            next_input.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            })

        if path is not None:
            return path

    raise RuntimeError("Maximum iterations exceeded")



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    path = run_search()
    print("Path:", path)

    with open("llm_result.json", "w") as f:
        json.dump({"path": path}, f, indent=2)

    env.close()
