# Run Backend and Frontend (Windows PowerShell)

## 1) Start Backend (FastAPI)

```powershell
Push-Location "D:\Code\Projects\rag-chatbot\rag-chatbot\backend"
C:/Users/farha/anaconda3/Scripts/conda.exe run -p "D:\Code\Ai start\1\venvai" --no-capture-output python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Backend URL:

- http://localhost:8000
- Docs: http://localhost:8000/docs

---

## 2) Start Frontend (Vite)

Open a second terminal, then run:

```powershell
Push-Location "D:\Code\Projects\rag-chatbot\rag-chatbot\frontend"
npm install
npm run dev
```

Frontend URL (usually):

- http://localhost:5173

If 5173 is busy, Vite will pick the next port (for example 5174).

---

## 3) Quick API Test (Optional)

```powershell
$body = @{ query = 'How do I use for loops in Python?' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8000/chat' -Method Post -ContentType 'application/json' -Body $body
```

---

## 4) If a Port Is Already In Use

```powershell
# Check listeners
Get-NetTCPConnection -LocalPort 8000,5173,5174 -State Listen

# Stop process on 8000 (if needed)
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force }
```

---

## 5) Stop Servers

Press Ctrl+C in each terminal running the servers.
