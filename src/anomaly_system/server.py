"""Web frontend for the anomaly detection system — FastAPI + SSE streaming."""

import asyncio
import json
import os
import sys
from pathlib import Path

import markdown
import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from anomaly_system.config import DATA_DIR, PLOTS_DIR, REPORTS_DIR, PROJECT_ROOT

app = FastAPI(title="Anomaly Detection System")

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Map command names to CLI args (data_path placeholder replaced at runtime)
COMMANDS: dict[str, list[str]] = {
    "health":         [sys.executable, "-m", "anomaly_system", "--health"],
    "pipeline":       [sys.executable, "-m", "anomaly_system", "--pipeline"],
    "agent":          [sys.executable, "-m", "anomaly_system", "--agent"],
    "agent-no-think": [sys.executable, "-m", "anomaly_system", "--agent", "--no-think"],
    "vision-test":    [sys.executable, "-m", "anomaly_system", "--vision-test"],
}

CSV_TEMPLATE = """feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,label
0.12,-0.34,0.56,0.78,-0.91,0.23,-0.45,0.67,0.89,-0.12,0
-0.23,0.45,-0.67,0.89,0.12,-0.34,0.56,-0.78,0.91,0.23,0
0.34,-0.56,0.78,-0.91,0.23,0.45,-0.67,0.89,-0.12,0.34,0
4.50,-3.80,5.20,-4.10,3.90,-4.60,5.30,-3.70,4.80,-5.10,1
-4.20,3.60,-5.40,4.30,-3.50,4.70,-5.60,3.80,-4.90,5.20,1
""".lstrip()

# Simple lock to prevent concurrent runs
_running = False


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the main UI page."""
    html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/run/{command}")
async def run_command(command: str, data_path: str | None = None) -> StreamingResponse:
    """Run a CLI command and stream output via SSE."""
    global _running

    if command not in COMMANDS:
        async def error_stream():
            yield f"data: {json.dumps(f'Unknown command: {command}')}\n\n"
            yield "event: done\ndata: {\"exit_code\": 1}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    if _running:
        async def busy_stream():
            yield f"data: {json.dumps('A command is already running. Please wait.')}\n\n"
            yield "event: done\ndata: {\"exit_code\": 1}\n\n"
        return StreamingResponse(busy_stream(), media_type="text/event-stream")

    # Build command with optional data path
    cmd = list(COMMANDS[command])
    if data_path:
        cmd.extend(["--data-path", data_path])

    async def stream():
        global _running
        _running = True
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )

            async for line in process.stdout:
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                yield f"data: {json.dumps(text)}\n\n"

            await process.wait()
            yield f"event: done\ndata: {json.dumps({'exit_code': process.returncode})}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps('Command timed out.')}\n\n"
            yield "event: done\ndata: {\"exit_code\": -1}\n\n"
        finally:
            _running = False

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/plots")
async def list_plots() -> list[str]:
    """Return list of available plot filenames."""
    if not PLOTS_DIR.exists():
        return []
    return sorted(f.name for f in PLOTS_DIR.iterdir() if f.suffix == ".png")


@app.get("/api/plots/{filename}")
async def get_plot(filename: str) -> FileResponse:
    """Serve a plot image."""
    path = PLOTS_DIR / filename
    if not path.exists() or not path.is_file():
        return FileResponse(path)  # will 404
    return FileResponse(path, media_type="image/png")


@app.get("/api/reports")
async def list_reports() -> list[str]:
    """Return list of report filenames, newest first."""
    if not REPORTS_DIR.exists():
        return []
    files = sorted(REPORTS_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    return [f.name for f in files if f.suffix == ".md"]


@app.get("/api/reports/{filename}")
async def get_report(filename: str) -> HTMLResponse:
    """Return a report rendered as HTML from markdown."""
    path = REPORTS_DIR / filename
    if not path.exists():
        return HTMLResponse("<p>Report not found.</p>", status_code=404)
    md_text = path.read_text(encoding="utf-8")
    html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    return HTMLResponse(html)


@app.get("/api/data")
async def list_data_files() -> list[dict]:
    """Return list of uploaded data files with size info."""
    if not DATA_DIR.exists():
        return []
    files = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.suffix in (".csv", ".parquet"):
            size_kb = f.stat().st_size / 1024
            files.append({"name": f.name, "size_kb": round(size_kb, 1), "path": str(f)})
    return files


@app.post("/api/upload")
async def upload_csv(file: UploadFile) -> dict:
    """Upload a CSV file to the data directory."""
    if not file.filename or not file.filename.endswith(".csv"):
        return {"error": "Only .csv files are accepted."}

    content = await file.read()
    dest = DATA_DIR / file.filename
    dest.write_bytes(content)

    size_kb = len(content) / 1024
    return {"success": True, "filename": file.filename, "size_kb": round(size_kb, 1), "path": str(dest)}


@app.get("/api/data/preview/{filename}")
async def preview_data(filename: str, max_rows: int = 2000) -> dict:
    """Return CSV data as JSON for timeseries plotting.

    Returns up to max_rows rows with feature columns and optional label column.
    """
    path = DATA_DIR / filename
    if not path.exists() or not path.is_file():
        return {"error": "File not found."}
    try:
        df = pd.read_csv(path, nrows=max_rows)
    except Exception as e:
        return {"error": str(e)}

    # Separate label column if present
    label = None
    if "label" in df.columns:
        label = df["label"].tolist()
        df = df.drop(columns=["label"])

    # Keep only numeric columns
    df = df.select_dtypes(include="number")
    if df.empty:
        return {"error": "No numeric columns found."}

    return {
        "columns": df.columns.tolist(),
        "rows": df.values.tolist(),
        "label": label,
        "n_rows": len(df),
    }


@app.get("/api/template.csv")
async def download_template() -> Response:
    """Download a CSV template file."""
    return Response(
        content=CSV_TEMPLATE,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=anomaly_template.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
