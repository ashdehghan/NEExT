# Shared experiments

Experiments meant to be run by anyone on the team. Each subdirectory holds one experiment: a notebook laid out as a
full study (question, setup, method, results, how to read them) and a README with the question, how to run it, the
runtime, the protocol, and a "Findings" section to fill in after the run. Downloaded data and outputs stay out of git
(each experiment's `.gitignore`).

Pull the `dev` branch and the experiments come with the library code they need.

| Directory | Question | Status |
|---|---|---|
| [`egonext-ring-vs-pooled/`](egonext-ring-vs-pooled/README.md) | Do separate hop rings carry more label signal than the pooled ego-net bag? | ready to run |
