
# langchain-udemy (local Ollama)

This workspace contains small LangChain examples, including a multi-agent credit scoring demo that can run **locally** using **Ollama** (no OpenAI/Anthropic API keys).

## Prereqs

- Windows (PowerShell) + Python 3.12+
- Ollama installed and running

## 1) Install Ollama

1. Download and install Ollama: https://ollama.com/download
2. Verify it works:

```powershell
ollama --version
```

If `ollama` is not found, restart your terminal (or reboot) after installing.

## 2) Pull a model (recommended)

Recommended defaults (pick one):

- **Best balance (most PCs):** `llama3.1:8b`
- **Faster on CPU / lower RAM:** `llama3.2:3b`
- **Higher quality (needs more VRAM/RAM):** `qwen2.5:14b` or `llama3.1:70b` (only if your machine can handle it)

Pull a model:

```powershell
ollama pull llama3.1:8b
```

If you see an Ollama error like:

`llama runner process has terminated` (HTTP 500) or `model requires more system memory ... than is available`

…switch to a smaller model:

```powershell
ollama pull llama3.2:3b
$env:OLLAMA_MODEL = "llama3.2:3b"
```

Check what you have installed:

```powershell
ollama list
```

## 3) Install Python dependencies

This repo uses `pyproject.toml`. From the repo root:

```powershell
py -m pip install -U pip
py -m pip install -e .
```

If you don’t want editable mode, use:

```powershell
py -m pip install .
```

### Common issue: `python` not found

On Windows, `python` can be shadowed by the Microsoft Store alias. Use the `py` launcher as shown above.

## 4) Run the local credit scoring demo

Set the provider to Ollama and pick a model:

```powershell
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "llama3.1:8b"
py credit_scoring.py
```

If you see:

`Missing dependency: 'langchain-ollama'`

…re-run step (3). That error means the Python environment you’re running doesn’t have the dependency installed.

## 5) Run the search agent demo (optional)

`main.py` uses Tavily search tools.

```powershell
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "llama3.1:8b"
py main.py
```

If Tavily isn’t installed in your environment you’ll get a clear runtime error. Install deps with step (3).

## Serve as a URL (HTTP API)

You can expose the credit-scoring pipeline as an HTTP service using FastAPI.

### Run the API locally

```powershell
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "llama3.1:8b"

py -m pip install fastapi "uvicorn[standard]"
py -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open:

- http://localhost:8000/health
- http://localhost:8000/docs

### Call the endpoint

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/credit-score -ContentType "application/json" -Body '
{
	"name": "Aarav Mehta",
	"age": 29,
	"employment_type": "Salaried",
	"years_employed": 6,
	"annual_income": 1800000,
	"monthly_debt": 22000,
	"loan_amount_requested": 900000,
	"loan_purpose": "Vehicle",
	"credit_score": 768,
	"missed_payments": 0,
	"savings": 350000
}
'
```

### Make it public (optional)

For a quick public URL while developing, you can use a tunnel (e.g., ngrok or Cloudflare Tunnel) to expose `http://localhost:8000`.

## UI: Streamlit "agent flow" (step-by-step)

This repo also includes a small Streamlit app that renders the multi-agent run as a UI flow:

```powershell
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "llama3.1:8b"

py -m pip install -e .
py -m streamlit run ui_streamlit.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Optional: Add a traditional ML credit model (fast baseline)

You can add a classic ML model (e.g., logistic regression) that predicts **probability of default (PD)** and maps it to an approximate **300–900** score.
The LangChain “agents” can then use that baseline as an extra signal for the final decision/explanation.

### Recommended (matches the Streamlit form inputs)

The Streamlit app (and `ApplicantProfile`) uses fields like age, income, DTI, credit score, etc.
To make the ML baseline meaningful for those inputs, use the ApplicantProfile-style generator/trainer:

```powershell
py generate_applicant_training_data.py --rows 5000 --out .\data\applicant_training.csv
py train_applicant_credit_model.py --data .\data\applicant_training.csv --out .\models\applicant_credit_model.joblib

$env:USE_ML_MODEL = "1"
$env:ML_MODEL_PATH = "C:\langchain_udemy\models\applicant_credit_model.joblib"
```

Then run either:

```powershell
py credit_scoring.py
py -m streamlit run ui_streamlit.py
```

### Train a model

1. Create a CSV (example columns):

`utilisation,dpd_days,cash_credit_ratio,cash_debit_ratio,inbound_cheque_bounce_count,inbound_cheque_bounce_amt,outbound_cheque_bounce_count,outbound_cheque_bounce_amt,total_amt_credit,total_amt_debit,no_of_banks,defaulted`

Where `defaulted` is `0` or `1`.

2. Train and save a model:

```powershell
py -m pip install scikit-learn joblib
py train_credit_model.py --data .\data\credit_training.csv --out .\models\credit_model.joblib

# Optional: override which columns to use
# py train_credit_model.py --data .\data\credit_training.csv --out .\models\credit_model.joblib --features utilisation,dpd_days,...
```

### Enable ML scoring in the app

```powershell
$env:USE_ML_MODEL = "1"
$env:ML_MODEL_PATH = ".\models\credit_model.joblib"
py credit_scoring.py
```


## Notes on “best” Ollama model

“Best” depends on your hardware:

- If you have **no GPU** or a modest GPU: start with `llama3.1:8b`.
- If you want **faster**: `llama3.2:3b`.
- If you have **lots of VRAM/RAM** and want higher quality: `qwen2.5:14b` (or larger) / `llama3.1:70b`.

