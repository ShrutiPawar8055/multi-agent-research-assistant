# Prompting and context windows

A prompt is the text a model receives before it generates. Clear instructions, a stated output format, and a few worked examples usually improve reliability more than clever wording. Telling the model to answer only from supplied evidence, and to say so when the evidence is insufficient, is a simple way to discourage invented facts.

Chain-of-thought prompting asks a model to reason step by step before answering, which helps on multi-step problems. The context window is the maximum amount of text a model can read at once. Models can overlook information placed in the middle of a very long context, so ordering the most relevant passages first helps.

Lower sampling temperatures make outputs more deterministic, which is preferable for fact-based question answering.
