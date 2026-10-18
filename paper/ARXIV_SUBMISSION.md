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
Upload `arxiv_submission.tar.gz` as a single file. arXiv unpacks it and compiles `main.tex` with pdfLaTeX (it uses the included `main.bbl`; it does not run BibTeX). Check the compiled preview arXiv shows: 17 pages, 10 figures, references at the end.

## 4. Metadata (copy and paste)

**Title**
How an AI Economy Behaves: A Multi-Agent Simulation in Which AI Companies Learn to Compete by Reinforcement Learning

**Authors**
Ravi Kadam

**Abstract**
We present an open, reproducible simulation of a stylised AI industry over ten years and let AI product companies learn to compete in it with reinforcement learning (RL). The simulated economy links a chip fabricator, chip designers, data centres, frontier and open-weight model providers, 5-12 competing AI product companies and 300 enterprise customers in three sectors. Companies decide which use cases to pursue, which model to build on, how much to spend, and when to pilot, launch, scale or quit, facing uncertain technology, hidden customer value, compute shortages and rivals. Learning from scratch failed: the agent stopped exploring and lost most of its capital. Imitating a competent rule and then improving with Proximal Policy Optimisation succeeded in under eight minutes on a laptop. Across five independently trained agents evaluated on 100 unseen markets, the agents created $15.5M of value per company on average against $14.0M for the best rule-based strategy; three of five were significantly better and none worse, mainly by choosing harder, less crowded use cases and timing pilots to customer readiness. Under a severe compute crunch, all five trailed the rule and survived less often. An apparently decisive single-run advantage (+$5.9M) shrank to +$1.5M once training and action-sampling randomness were accounted for. In the simulated market, adoption follows S-curves, a regulated sector is held back by reliability rather than demand, first launchers gain no lasting share, and supply shocks quadruple compute prices. Written for readers new to RL, the paper explains each RL concept alongside the result it produces and states nine hypotheses about real AI markets, each tied to the modelled assumptions it depends on and the evidence needed to test it. Code, trained agents and a live viewer are open source.

**Comments**
17 pages, 10 figures. Code, trained agent and live viewer: https://github.com/ravikadam/ai-economy-lab

**ACM class (optional):** I.2.6; I.2.11; J.4

## 5. Preview and submit
Check the generated PDF and the metadata page, then submit. New submissions are announced after moderation, usually within one to two working days (arXiv announces Sunday to Thursday evenings, US Eastern time).

## Rebuilding the package
From the repo root: `python paper/make_figures.py`, then in `paper/`: `tectonic --keep-intermediates main.tex` and re-create the tarball (main.tex with `\pdfoutput=1` as its first line, main.bbl, references.bib, figures/*.pdf).
