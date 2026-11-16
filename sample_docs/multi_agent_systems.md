# Multi-agent systems

A multi-agent system splits one large task among several specialised language model agents. Each agent has a narrow role, its own instructions, and sometimes its own tools. A planner agent decides what needs to be done, worker agents carry out the pieces, and a critic agent checks the result before it is returned.

Separating the writer from the critic is valuable because a model that grades its own draft tends to approve it. An independent reviewer with a different prompt is more likely to catch missing citations or unsupported claims.

Independent sub-tasks can run in parallel, which cuts latency. Feedback loops let the critic send a draft back for revision, but every loop needs a hard limit on iterations so a stubborn disagreement cannot run forever or burn through the budget.
