# Agent tools

Call `get_tool_specs(config)` to expose the enabled read-only tools. Pass the provider's raw argument string to `dispatch(name, raw_args, index, config)` and return its text to the model. Citations identify the checked-out commit's source ranges.
