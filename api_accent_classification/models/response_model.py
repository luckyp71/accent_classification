from typing import Union
from pydantic import BaseModel
from typing_extensions import List

class ResponseModel(BaseModel):
    response_code: int
    data: Union[dict, List[dict]]