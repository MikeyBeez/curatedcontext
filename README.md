# Context Curation Improves LLM Answer Quality

Empirical evaluation showing that curating context before presenting it to a language model improves answer quality while dramatically reducing token usage.

## Key Results (200 SQuAD 2.0 questions)

| Metric | Baseline (raw context) | Curated | Delta |
|--------|----------------------|---------|-------|
| Exact Match | 55.0% | 60.5% | **+5.5 pp** |
| F1 | 76.1% | 76.9% | +0.8 pp |
| Prompt Tokens | 1,191 | 95 | **-92.0%** |

## Design

Two conditions tested on every question:
- **Baseline**: Noisy context (1 relevant + 6 distractor paragraphs) → answer model
- **Curated**: Same noisy context → curator model extracts relevant sentences → answer model

Models: Llama 3.1 8B (answer), Qwen 2.5 7B (curator), both via Ollama.

## Reproduce

```bash
# Install Ollama, then:
ollama pull llama3.1:8b
ollama pull qwen2.5:latest

# Clone and run
git clone https://github.com/mikeybeez/curatedcontext.git
cd curatedcontext
python3 experiment.py
```

Requires: Python 3.8+, Ollama running on localhost:11434. No pip dependencies.

Runtime: ~2 hours for 200 questions on Apple Silicon Mac Mini.

## Files

- `experiment.py` — Complete experiment script
- `paper.txt` — Plain text version of the paper
- `paper.tex` — LaTeX source
- `paper.pdf` — Compiled PDF
- `results.jsonl` — Per-question results (200 entries)
- `summary.json` — Aggregate statistics

## Citation

```
Bee, M. (2026). Context Curation Improves LLM Answer Quality: An Empirical
Evaluation on SQuAD. https://github.com/mikeybeez/curatedcontext
```

## Related

- [Understanding Is Getting the Context Right](https://doi.org/10.5281/zenodo.18571717) — The theoretical paper this experiment validates.

## License

MIT
