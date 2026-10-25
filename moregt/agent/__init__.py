"""Agent-derived columns.

Before fine-tuning, an LLM agent (Claude Code) reads a database through a small read-only tool layer
and proposes new row-local columns for a task. Proposals are screened without labels or a model, and
the accepted ones are added to their tables as ordinary numerical cells.

* :mod:`.blind`       the database as the agent sees it: data and names, no declared keys
* :mod:`.keyinfer`    primary/foreign-key candidates inferred from the data alone
* :mod:`.tools`       the tool registry and the four tools the agent may call
* :mod:`.grammar`     the row-local expression grammar, the falsification checks and :func:`screen`
* :mod:`.runner`      the Claude Code subprocess and the audit of its tool calls
* :mod:`.derive_loop` the propose / screen / revise loop
* :mod:`.columns`     reading accepted columns and adding them to a :class:`GraphBundle`
"""
