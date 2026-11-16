# Evaluating language model applications

Evaluation turns opinions about quality into numbers that can be tracked over time. For a research assistant, three questions matter most. Did the system choose the right sources? Is each claim in the answer supported by the evidence it cites? Does the answer contain the facts a domain expert expects?

Faithfulness measures whether claims are supported by the retrieved context. Context recall measures whether the retrieved passages contain the information needed to answer. Routing accuracy measures how often the planner picks the correct source for a question.

Deterministic checks, such as verifying that every citation number exists, are cheap and perfectly repeatable. Judgments made by another model cover subtler errors but vary from run to run, so they work best alongside deterministic checks rather than instead of them.
