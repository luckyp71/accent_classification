from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, HttpUrl
from models.response_model import ResponseModel
from controllers.accent_classification_controller import predict_audio_from_url

accent_classification_router = APIRouter(
    prefix="/accent",
    tags=["accent_classification"]
)


class VideoURLRequest(BaseModel):
    url: HttpUrl


@accent_classification_router.post("/classify")
async def classify_audio(request: VideoURLRequest):
    try:
        prediction_result = predict_audio_from_url(str(request.url))

        if prediction_result is None:
            raise HTTPException(status_code=500, detail="Failed to extract features or make prediction")

        response_model = ResponseModel(
            response_code=status.HTTP_200_OK,
            data={
                "url": str(request.url),
                "prediction": prediction_result
            }
        )
        return response_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
