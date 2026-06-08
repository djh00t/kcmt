# New Feature: GPT Token Efficiency

## GPT Token Efficiency

### Key Practices to Save Tokens

- **Context discipline:** Use grep/search first and fetch only the files you actually need.
  Auto-loading large repo slices for small fixes creates major input waste.

- **Avoid resending full repo context on retries:** Cap retries and resend only diffs/context deltas.
  Reattaching the same large payload on every retry is one of the fastest ways to spike cost.

- **Prompt caching and stable prefixes:** Cache repeated, unchanged prompt prefixes so identical context is not billed repeatedly.

- **Batch small questions:** Ask related questions in one call instead of many separate turns.
  This avoids repeated prefix charges and can save roughly 70 to 90 percent on routine workflows.

- **Use multi-model routing:** Route routine execution to cheaper/smaller models and reserve premium reasoning models for planning and review.

- **Use lower-effort modes for implementation:** Use high/xhigh for planning, and medium/low for build steps when quality allows.

- **Trim reply verbosity (caveman style):** Require concise answers to reduce filler tokens.

- **Keep AGENTS.md and guardrails minimal:** Remove irrelevant instruction text so the model processes only what matters.

- **Avoid full-file rewrites:** Prefer targeted diffs/snippets over broad rewrites to reduce read/write context volume.

- **Profile tool calls before prompt tuning:** Identify whether planning, retries, or tool loops drive spend, then optimize the biggest cost source first.

## Model and Mode Tradeoffs (Community Consensus)

- **GPT-5.4 vs GPT-5.5:** GPT-5.5 often uses more tokens at high/xhigh. GPT-5.4 is frequently more cost-efficient for similar output quality.

- **Use GPT-5.5 strategically:** Best for planning/troubleshooting when deeper reasoning can reduce back-and-forth. Use cheaper models for routine edits.

- **Daily-driver pattern:** Many teams prefer GPT-5.4 xhigh or GPT-5.4-mini for day-to-day cost/quality balance, using GPT-5.5 selectively.

- **Per-case variance (N=1 caveat):** Results vary by workload. Small-sample benchmarks are directional, not absolute.

## Orchestration and Governance to Reduce Waste

- **Harness engineering and skill maps:** Route tasks through focused sub-skills/subagents to keep context narrow and improve accuracy.

- **Explicit steps and boundaries:** Specify exact files, steps, and constraints to prevent over-generalization and unnecessary repo reads.

- **Lower reasoning for mechanical tasks:** Reserve xhigh for exploration and ambiguity; use mini/medium for repetitive execution work.

## Quick Tactical Checklist

- Grep/search before fetching context.
- Cache stable prompt prefixes.
- Batch related small asks.
- Cap retries and resend diff-only context.
- Keep AGENTS.md lean and relevant.
