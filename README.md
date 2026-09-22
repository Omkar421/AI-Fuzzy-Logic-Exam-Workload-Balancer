# AI + Fuzzy Logic Exam Workload Balancer

This mini project satisfies the Internal Assessment requirements by combining **LangChain language understanding** with a genuine **fuzzy inference system**. The application helps a student convert an informal exam-planning message into an explainable workload-risk score and a recommended weekly focus time.

## Why this topic

Exam preparation is a real-world education problem. Students often know that a subject is difficult or urgent, but their information is incomplete and expressed in natural language. The project uses an LLM to understand that language and fuzzy logic to reason with gradual concepts such as low confidence, medium urgency, and high difficulty.

## Where each compulsory component sits

### AI/LLM component using LangChain

`ai_chain.py` creates a LangChain prompt-and-model chain. It uses `ChatPromptTemplate`, a Hugging Face hosted model through `HuggingFaceEndpoint` and `ChatHuggingFace`, and `StrOutputParser`.

The first chain performs real language work: it extracts a subject, difficulty, days left, confidence, study hours, and weekly capacity from a free-text message. The second chain explains the fuzzy result in three short sentences. The LLM does not calculate the final risk score, which keeps the decision logic auditable.

### Fuzzy Logic component

`fuzzy_engine.py` uses `scikit-fuzzy` control-system primitives. It defines membership functions for difficulty, urgency, confidence, study load, and output risk. The system fuzzifies crisp inputs, evaluates eight fuzzy rules, and defuzzifies the aggregated output into a numeric risk score from 0 to 100.

This is not a collection of ordinary `if/else` conditions. A numeric input can partially belong to more than one fuzzy set, and the rules combine those membership degrees before producing a continuous output.

## Example input

```text
I have Data Structures in 5 days. It is difficulty 8, my confidence is 4,
I need 12 hours, and I can study 24 hours weekly.
```

The LangChain component extracts the numeric signals. The fuzzy system then activates rules such as:

- High difficulty AND high urgency → critical risk.
- Low confidence AND high study load → critical risk.
- High difficulty AND low confidence → high risk.
- Difficulty low AND confidence high AND urgency low → low risk.

The resulting defuzzified risk is displayed with the active memberships, active rules, and recommended study time. The LangChain explanation then states why the subject received that result and suggests a practical next step.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
streamlit run app.py
```

Without a Hugging Face token, the app runs in a clearly labelled demo mode with deterministic fallback extraction and explanation. For the assessment submission, configure the token so the live LangChain component is active.

## Test the fuzzy engine

```bash
pytest -q
```

## Deploy to Streamlit Community Cloud

1. Create a public GitHub repository and push this folder.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **New app**, choose the repository, branch `main`, and file `app.py`.
4. In the app settings, open **Advanced settings → Secrets** and add:

```toml
HUGGINGFACEHUB_API_TOKEN = "hf_your_real_token"
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
```

5. Deploy the app. Streamlit will provide a live URL for the submission.

Never commit a real API token to GitHub. Streamlit's Secrets interface is the correct place for it.

## Viva flow

Explain the application in this order:

1. The student enters an informal sentence.
2. LangChain converts language into structured numeric inputs.
3. Fuzzification calculates degrees of membership for low, medium, and high sets.
4. The fuzzy rules combine those degrees using AND operations.
5. Defuzzification converts the combined output into one risk score.
6. The second LangChain call explains the result conversationally.

The important design decision is that the LLM handles language and explanation, while fuzzy logic handles the numeric decision. This prevents the LLM from becoming an untestable black box.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface and workflow orchestration |
| `ai_chain.py` | LangChain extraction and explanation chains |
| `fuzzy_engine.py` | Membership functions, rules, and defuzzification |
| `test_fuzzy_engine.py` | Basic fuzzy-engine tests |
| `requirements.txt` | Python dependencies |
| `REPORT.md` | Five-page write-up draft |
