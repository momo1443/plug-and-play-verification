"""HotpotQA prompts for Qwen search and minimal answers."""

HOTPOTQA_SYSTEM_PROMPT = (
    "You are a research agent. Your goal is to answer the User Query using Wikipedia search evidence."
)

HOTPOTQA_FINAL_TURN_PROMPT = """FINAL TURN: Do not call search again. Use the best available evidence.
Reply only with `<answer>MINIMAL_ANSWER</answer>`.
- Return only the smallest answer span that directly answers the question.
- For a yes/no question, MINIMAL_ANSWER must be exactly `yes` or `no`.
- Do not include explanations or information that the question did not request.
- For a date, match the requested granularity exactly: year, month, or full date."""

HOTPOTQA_USER_PROMPT = """### User Query
{user_query}

### History Actions
{history_actions}

### Retrieved Passages
{passage_list}

### Recent tool / format issues
{tool_feedback}

### Instructions
Analyze the **Retrieved Passages** and **History Actions** and determine the next action.
Output only one search tool call or one final `<answer>`; never expose reasoning as prose.
Make at most one search call per turn.
When passages name a plausible entity, search that entity together with the missing attribute
instead of repeating all clues.
**Attend to the history actions and avoid repeating the same search queries.**
When you can answer the question from the current passages, or when told that it is the final turn:
- Return only the smallest answer span inside `<answer></answer>`.
- For yes/no questions, return exactly `<answer>yes</answer>` or `<answer>no</answer>`.
- Do not include explanations or information the question did not request.
- For dates, match the requested granularity exactly: year, month, or full date.

### Visible Output Format
For a search turn, output only:
<tool_call>
[One tool call]
</tool_call>

For a final answer, output only:
<answer>[Minimal answer span]</answer>
"""

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search Wikipedia for passages relevant to the user question. "
            "Use natural-language or keyword queries; must differ from prior history queries when possible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A single search query (natural language or keywords). "
                        "Must differ from all history queries when seeking new evidence."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

HOTPOTQA_TOOL_SCHEMAS = [SEARCH_TOOL_SCHEMA]
