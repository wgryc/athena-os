# Change Log for v0.1.1

- When running `athena frontend` without a local `config.json` file, we originally showed an empty dashboard. Now, we show a dashboard with high-level performance metrics and line chart of historical value.
- Running `athenaos version` prints the current version of the package.
- Significant change to how the frontend chatbot operates:
    - "frontend_llm_chat_style" setting in `config.json` allows you to set a style for how the LLM writes. For exmaple, ask it to be concise, avoid markdown, or explain things in simple terms.
    - Added support for compacting chats to avoid running out of context window. The Python code has a `COMPACT_ON_NUM_TOKENS` variable for when this should be run. Upon reaching the amount, the chat is compacted before continuing.
    - The LLM recieves `SYSTEM INFO` when it receives a message. Right now this only includes a timestamp, but can include additional information about the chat.
- You can now download the "debug log" via the frontend.
- `config.json` can now be edited via the frontend.

# Change Log for v0.1.0

This is the first version. Yay!