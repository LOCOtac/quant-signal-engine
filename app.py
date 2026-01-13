from fastapi import FastAPI, Query
from simple_signal_model_fmp import run_signal_analysis

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "service": "quant-signal-engine"}

@app.get("/analyze")
def analyze(symbol: str = Query(..., description="Stock ticker symbol")):
    try:
        result = run_signal_analysis(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}

