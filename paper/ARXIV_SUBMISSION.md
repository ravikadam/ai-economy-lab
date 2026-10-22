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
Upload `arxiv_submission.tar.gz` as a single file. arXiv unpacks it and compiles `main.tex` with pdfLaTeX (it uses the included `main.bbl`; it does not run BibTeX). Check the compiled preview arXiv shows: 26 pages, 11 figures, references before the appendices (glossary, reproducibility, reward equations, audit trail).

## 4. Metadata (copy and paste)

**Title**
How an AI Economy Behaves: A Multi-Agent Simulation in Which AI Companies Learn to Compete by Reinforcement Learning

**Authors**
Ravi Kadam

**Abstract**
We present an open, reproducible simulation of a stylised AI industry over ten years and let AI product companies learn to compete in it by reinforcement learning (RL). The simulated economy links a chip fabricator, chip designers, data centres, frontier and open-weight model providers, 5-12 competing AI product companies and 300 enterprise customers in three sectors. Companies decide which use cases to pursue, which model to build on, how much to spend, and when to pilot, launch, scale or stop, facing uncertain technology, hidden customer value, compute shortages and rivals. We fixed the training changes, test markets and success bar in a protocol written before the v2 experiments, trained five agents on a laptop in 13-30 minutes each, and tested each on never-used markets. In 100 new markets the agents created $19.4M of value per company against $14.3M for the strongest hand-written rule; four of five were clearly better and none worse, narrowly missing our pre-set bar of all five. They also led under faster or slower technology progress and a wave of entrants; with harder use cases the lead was not statistically clear, and in a severe compute crunch there was no statistically detectable difference from the rule, while the agents went bankrupt more often. Five-run comparisons found no detectable benefit from varied training markets; RL added $4.7M over imitating the rule alone; learning from scratch succeeded in three of five runs. Scores on the practice markets used to pick each agent overstated test performance by about $2.8M. In the simulated market, adoption follows S-curves, the regulated sector is held back by reliability, and supply shocks nearly quadruple compute prices. Written for readers new to RL, it explains each idea where it first matters, with a glossary, and states real-world lessons as hypotheses. Code, trained agents and a live viewer are open source.

**Comments**
26 pages, 11 figures. Code, trained agents, test plan and live viewer: https://github.com/ravikadam/ai-economy-lab

**ACM class (optional):** I.2.6; I.2.11; J.4

## 5. Preview and submit
Check the generated PDF and the metadata page, then submit. New submissions are announced after moderation, usually within one to two working days (arXiv announces Sunday to Thursday evenings, US Eastern time).

## Rebuilding the package
From the repo root: `python paper/make_figures.py`, then in `paper/`: `tectonic --keep-intermediates main.tex` and re-create the tarball (main.tex with `\pdfoutput=1` as its first line, main.bbl, references.bib, figures/*.pdf).
