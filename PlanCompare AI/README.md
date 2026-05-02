# PlanCompare AI

## Frontend

- `frontend/` contains the Next.js app
- Run:
  - `cd frontend`
  - `npm install`
  - `npm run dev`

## Backend

- `backend/` contains the FastAPI server
- Run:
  - `cd backend`
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

## Usage

- Open the frontend at `http://localhost:3000`
- Upload `Previous drawing` and `Revised drawing`
- Click `Compare drawings`
- The app posts to `http://localhost:8000/compare-drawings`
