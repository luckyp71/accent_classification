import uvicorn
from fastapi import FastAPI

from routers.accent_classification_router import accent_classification_router

app = FastAPI()
app.include_router(accent_classification_router)

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=8000, reload=True)