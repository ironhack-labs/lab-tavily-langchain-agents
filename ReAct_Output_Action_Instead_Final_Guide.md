# Guide: Fixing ReAct Returning `Action:` Instead Of A Final Report

## Error

You ran an agent query like:

```python
response = agent_executor.invoke({"input": prompt})
```

The returned object has this shape:

```python
{
    "input": "...",
    "output": "Action: TavilySafeSearch\nAction Input: ...",
    "intermediate_steps": [...]
}
```

The problem is that `"output"` is not the final Markdown report. It is another tool action:

```text
Action: TavilySafeSearch
Action Input: "OpenAI ChatGPT Team vs Anthropic Claude Pro comparison 2025"
```

Inside `intermediate_steps`, you also see repeated parser errors:

```text
Invalid Format: Missing 'Action:' after 'Thought:'
```

## Diagnosis

The Tavily tool is being called. The issue is not that search failed.

The issue is that the ReAct agent did not finish cleanly.

The model generated a useful-looking Markdown report inside the ReAct loop, but it did not prefix that report with the exact parser marker:

```text
Final Answer:
```

LangChain's ReAct parser expects one of two valid formats:

```text
Thought: ...
Action: ...
Action Input: ...
```

or:

```text
Thought: ...
Final Answer: ...
```

When the model wrote the report directly after `Thought:` without a valid `Final Answer:`, the parser rejected it. Because your executor has:

```python
handle_parsing_errors=True
```

LangChain retried instead of crashing. That is why the same report appears multiple times inside `intermediate_steps`.

Eventually, the executor stopped before receiving a clean final answer, so the returned `"output"` is just an unfinished `Action:` block.

## Why This Happens

### 1. ReAct Is A Text Protocol

The ReAct agent is not using a structured function-calling protocol. It relies on the model writing exact text labels:

```text
Thought:
Action:
Action Input:
Observation:
Final Answer:
```

If the model writes a polished Markdown report but forgets `Final Answer:`, the parser treats it as invalid.

### 2. Long Report Prompts Make The Model Leave ReAct Format

Your prompt asks for a full Markdown report with pricing, features, context windows, user reviews, source links, and a verdict.

After retrieving evidence, the model naturally starts writing the report. But the ReAct parser still expects the report to appear only after:

```text
Final Answer:
```

### 3. `handle_parsing_errors=True` Causes Retries

This setting is useful for small parser mistakes, but for this case it causes repeated loops:

```python
handle_parsing_errors=True
```

The model writes a report, the parser rejects it, LangChain asks it to try again, and the model writes a similar report again.

### 4. `early_stopping_method="generate"` Can Return Awkward Output

If the agent reaches `max_iterations` or `max_execution_time`, `early_stopping_method="generate"` asks the model to generate a final response from the scratchpad.

With ReAct formatting problems, this can produce a partial action or another malformed response instead of a clean report.

## Fix Option 1: Tell The Agent Exactly How To Finish

Add a final-format instruction to the user prompt:

```python
prompt = """
Compare OpenAI's ChatGPT Team and Anthropic's Claude Pro using recent 2025 sources.
Create a concise Markdown report including pricing, features, context window,
collaboration tools, model capabilities, limitations, user reviews, and best use cases.
Include a comparison table, source links, and a short final verdict.
Avoid marketing language and distinguish confirmed facts from opinions.

Important ReAct formatting rule:
After you have enough search results, your final response must begin exactly with:

Final Answer:

Put the complete Markdown report after Final Answer.
Do not write the report before Final Answer.
"""
```

### Why This Works

The parser can only return a final result when it sees:

```text
Final Answer:
```

This instruction makes the final-answer marker explicit in the task prompt, not just hidden inside the generic ReAct prompt.

## Fix Option 2: Use A More Helpful Parsing Error Message

Replace:

```python
handle_parsing_errors=True
```

with:

```python
handle_parsing_errors=(
    "Invalid ReAct format. If you have enough information, respond with "
    "'Final Answer:' followed by the complete Markdown report. "
    "Otherwise use exactly 'Action:' and 'Action Input:'."
)
```

### Why This Works

When the parser fails, LangChain sends this message back to the model.

Instead of a generic retry, the model receives a precise repair instruction:

- If done, use `Final Answer:`.
- If not done, use `Action:` and `Action Input:`.

This directly targets the failure shown in your traceback.

## Fix Option 3: Avoid `early_stopping_method="generate"` For This Lab

Use:

```python
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=(
        "Invalid ReAct format. If you have enough information, respond with "
        "'Final Answer:' followed by the complete Markdown report. "
        "Otherwise use exactly 'Action:' and 'Action Input:'."
    ),
    return_intermediate_steps=True,
    max_iterations=4,
    max_execution_time=60,
    early_stopping_method="force",
)
```

### Why This Works

`early_stopping_method="force"` avoids asking the model to improvise a final answer after a failed loop.

If the agent fails to finish, it fails more clearly instead of returning a misleading partial action as `"output"`.

This makes debugging easier.

## Fix Option 4: Use Search-First Synthesis For Report Tasks

For comparison reports, a ReAct loop is not the most reliable pattern.

Use this two-step approach:

1. Search with Tavily.
2. Ask the LLM to write the report using only the search results.

Example:

```python
search_query = (
    "ChatGPT Team Claude Pro pricing features context window collaboration "
    "2025 OpenAI Anthropic comparison"
)

search_context = safe_search(search_query)

synthesis_prompt = f"""
You are writing a concise Markdown comparison report.

Use only the search context below. Distinguish confirmed facts from opinions.
Include pricing, features, context window, collaboration tools, model capabilities,
limitations, user reviews, best use cases, source links, and a short final verdict.

Search context:
{search_context}
"""

answer = llm.invoke(synthesis_prompt).content
display(Markdown(answer))
```

### Why This Works

The task does not require an agent to decide between many tools. It needs retrieval followed by synthesis.

Search-first synthesis avoids the fragile ReAct text parser entirely:

- No `Thought:`.
- No `Action:`.
- No `Action Input:`.
- No `Final Answer:` parser requirement.

For report-writing tasks, this is usually more reliable than ReAct.

## Fix Option 5: Detect Unfinished Agent Output

Add a guard before displaying:

```python
response = agent_executor.invoke({"input": prompt})

output = response["output"]

if output.lstrip().startswith("Action:"):
    raise RuntimeError(
        "The agent stopped before producing a final answer. "
        "Check intermediate_steps for parser errors."
    )

display(Markdown(output))
```

### Why This Works

It prevents the notebook from silently treating an unfinished tool action as the final answer.

This is useful while debugging agents.

## Which Fix Should You Use?

For this lab, use **Fix Option 1 + Fix Option 2 + Fix Option 3** if you want to keep the ReAct agent.

Use **Fix Option 4** if your goal is the best report quality and fewer agent-formatting problems.

Recommended path:

```text
Learning agents? Use ReAct, but add strict final-answer formatting.
Writing reliable reports? Use search-first synthesis.
```

## Corrected ReAct Setup

```python
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=(
        "Invalid ReAct format. If you have enough information, respond with "
        "'Final Answer:' followed by the complete Markdown report. "
        "Otherwise use exactly 'Action:' and 'Action Input:'."
    ),
    return_intermediate_steps=True,
    max_iterations=4,
    max_execution_time=60,
    early_stopping_method="force",
)
```

Corrected prompt ending:

```python
prompt = """
Compare OpenAI's ChatGPT Team and Anthropic's Claude Pro using recent 2025 sources.
Create a concise Markdown report including pricing, features, context window,
collaboration tools, model capabilities, limitations, user reviews, and best use cases.
Include a comparison table, source links, and a short final verdict.
Avoid marketing language and distinguish confirmed facts from opinions.

Important ReAct formatting rule:
After using Tavily and gathering enough evidence, write:

Final Answer:

Then put the complete Markdown report below it.
"""
```

Display:

```python
response = agent_executor.invoke({"input": prompt})
display(Markdown(response["output"]))
```

## Corrected Search-First Pattern

```python
search_context = safe_search(
    "ChatGPT Team Claude Pro pricing features context window collaboration "
    "2025 OpenAI Anthropic comparison"
)

answer = llm.invoke(f"""
Use only the search context below to write a concise Markdown comparison report.
Include source links and distinguish confirmed facts from opinions.

Search context:
{search_context}
""").content

display(Markdown(answer))
```

## Summary

The output is wrong because the ReAct agent did not complete the required parser format.

It found sources and drafted a report, but the report was not accepted because it was missing:

```text
Final Answer:
```

Use one of these fixes:

- Add explicit `Final Answer:` instructions to the prompt.
- Use a custom `handle_parsing_errors` message.
- Avoid `early_stopping_method="generate"` while debugging.
- Prefer search-first synthesis for comparison reports.

The most robust solution for this specific task is search-first synthesis.
