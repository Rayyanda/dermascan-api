"""
Convenience launcher: `python run.py`
(equivalent to `uvicorn app.main:app --reload`)

Reads PORT from the environment when present (Render, Railway, Heroku-style
platforms all inject this), falling back to 8000 for local development.
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    is_production = os.environ.get("RENDER") is not None  # Render sets this automatically
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=not is_production)
