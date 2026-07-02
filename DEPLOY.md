# Deploying to Hugging Face Spaces (free, Docker)

Same playbook as your RAG Eval Harness. The Space builds the Docker image, runs
the pipeline once at build time (on the synthetic sample, so it boots without the
150 MB CSV), and serves the Streamlit app on port 7860.

## Steps

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: **CPU basic (free)**

2. The Space's `README.md` needs YAML frontmatter at the very top so HF knows how
   to run it. Prepend this block to the top of `README.md` **before pushing**
   (keep the rest of the README below it):

   ```
   ---
   title: Fraud Cost Explorer
   emoji: 💳
   colorFrom: red
   colorTo: gray
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. Push the repo:

   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<your-username>/fraud-cost-explorer
   git add .
   git commit -m "Fraud detection cost-optimization study"
   git push origin main
   ```

4. The Space builds automatically. First build takes a few minutes (xgboost +
   shap compile). Once it's green, the cost explorer is live.

## Notes

- `data/creditcard.csv` is gitignored — the Space runs on the synthetic sample,
  which is enough to demonstrate the interaction. For real numbers in the app,
  you'd commit the CSV (it fits within HF's limits via git-LFS) or run locally.
- If the build times out compiling shap, drop `shap` from `requirements.txt` for
  the deployed Space (the SHAP plots are pre-rendered PNGs in `outputs/` anyway)
  and keep it for local runs only.
