# Contributing

Thank you for helping improve AI Trading Bot. Contributions should remain
reproducible, testable, and explicit about financial assumptions.

## Before opening a pull request

1. Create or reference an issue describing the problem.
2. Branch from `main` using `fix/<topic>`, `feat/<topic>`, or
   `docs/<topic>`.
3. Install the development dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the quality checks:

   ```bash
   python -m black --check main.py src test_technical_analysis.py
   python -m flake8 main.py src test_technical_analysis.py
   python -m pytest test_technical_analysis.py
   ```

## Contribution standards

- Add tests for behavior changes and bug fixes.
- Do not commit API keys, exchange credentials, datasets, model weights, or
  generated results.
- Keep training and evaluation data separated chronologically.
- Document assumptions involving fees, slippage, latency, and execution price.
- Do not present backtest results as guaranteed future performance.

## Pairing and co-authorship

Pair programming is welcome when both contributors make substantive changes.
Record co-authorship only with the collaborator's verified GitHub email:

```text
Co-authored-by: Full Name <verified-email@example.com>
```

The pull request should briefly describe each contributor's work.
