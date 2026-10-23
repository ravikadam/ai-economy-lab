# Submitting to arXiv

Everything needed is in this folder. `arxiv_submission.tar.gz` is the upload; `main.pdf` is the compiled paper to proofread first.

## 1. Before you start
1. Read `main.pdf` end to end. You are the sole author and responsible for every claim.
2. Create or log in to your account at https://arxiv.org/user/register (use your real name: Ravi Kadam).
3. **Endorsement.** First-time submitters to a category usually need an endorsement from an existing arXiv author in that category. If arXiv asks for one during submission, it gives you an endorsement code to send to a researcher who publishes in cs.MA or cs.LG. See https://info.arxiv.org/help/endorsement.html

## 2. Start a new submission
- https://arxiv.org/submit
- **License:** CC BY 4.0 is recommended (lets others reuse with attribution). Choose before uploading; it cannot be made more restrictive later.
- **Primary category:** `cs.MA` (Multiagent Systems)
- **Cross-lists:** `cs.LG` (Machine Learning), `econ.GN` (General Economics)

## 3. Upload files
Upload `arxiv_submission.tar.gz` as a single file. arXiv unpacks it and compiles `main.tex` with pdfLaTeX (it uses the included `main.bbl`; it does not run BibTeX). Check the compiled preview arXiv shows: 31 pages, 12 figures, references before the appendices (glossary, reproducibility, reward equations, audit trail).

## 4. Metadata (copy and paste)

**Title**
How an AI Economy Behaves: A Multi-Agent Simulation in Which AI Companies Learn to Compete by Reinforcement Learning

**Authors**
Ravi Kadam

**Abstract**
We present an open, reproducible simulation of a stylised AI industry over ten years in which AI product companies learn to compete by reinforcement learning (RL). It links a chip fabricator, chip designers, data centres, frontier and open-weight model providers, 5-12 AI product companies and 300 enterprise customers in three sectors; companies choose use cases, models, budgets and when to pilot, launch, scale or stop, facing uncertain technology, hidden customer value, compute shortages and rivals. With the test plan fixed in advance, five PPO agents trained on a laptop created $19.3M of value per company on 100 unseen markets against $14.3M for the strongest hand-written rule; four of five were clearly better, narrowly missing our pre-set bar of five. They also led under faster or slower technology progress and a wave of entrants, but showed no detectable difference from the rule in a severe compute crunch, where they went bankrupt more often. RL added $4.7M over imitating the rule; learning from scratch succeeded in three of five runs, and its successes outperformed imitation-started agents. We also introduce Paired-Fork Advantage against a Rule (PFAR), which copies the market mid-game and compares the agent's action with the rule's under identical future randomness. PFAR cut the noise of the learning signal 1.7-3.9 fold but, in a pre-registered test, produced no better agents. In the simulated market, adoption follows S-curves, the regulated sector is held back by reliability, first launchers gain no lasting share, and supply shocks nearly quadruple compute prices. Written for readers new to RL, with a glossary. Code, agents and a live viewer are open source.

**Comments**
31 pages, 12 figures. Code, trained agents, test plan and live viewer: https://github.com/ravikadam/ai-economy-lab

**ACM class (optional):** I.2.6; I.2.11; J.4

## 5. Preview and submit
Check the generated PDF and the metadata page, then submit. New submissions are announced after moderation, usually within one to two working days (arXiv announces Sunday to Thursday evenings, US Eastern time).

## Rebuilding the package
From the repo root: `python paper/make_figures.py`, then in `paper/`: `tectonic --keep-intermediates main.tex` and re-create the tarball (main.tex with `\pdfoutput=1` as its first line and `\aixivtrue` changed to `\aixivfalse`, so the arXiv version lists Ravi Kadam as author, main.bbl, references.bib, figures/*.pdf).
