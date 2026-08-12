from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import uvicorn
import sys

app = FastAPI()

model = joblib.load('/app/model.joblib')

@app.get('/ping')
def ping():
    return JSONResponse(content={'status': 'Healthy'}, status_code=200)

@app.post('/invocations')
async def predict(request: Request):
    data = await request.json()
    features = np.array(data['inputs'])
    predictions = model.predict(features)
    return JSONResponse(content={'predictions': predictions.tolist()})

@app.get('/')
def root():
    return {'message': 'EV Loan Prediction Model Container'}

if __name__ == '__main__':
    # Starts Uvicorn listening on port 8080 regardless of CLI args like 'serve'
    uvicorn.run('app:app', host='0.0.0.0', port=8080)

