"""Request and response contracts for the scoring API."""

from typing import Dict, List, Literal

from pydantic import BaseModel, Field

HomeOwnership = Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
LoanIntent = Literal["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
LoanGrade = Literal["A", "B", "C", "D", "E", "F", "G"]
DefaultFlag = Literal["N", "Y"]
Decision = Literal["approve", "review", "reject"]


class ApplicantRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=100, examples=[25])
    person_income: float = Field(..., gt=0, examples=[50000])
    person_emp_length: float = Field(..., ge=0, examples=[3.0])
    loan_amnt: float = Field(..., gt=0, examples=[5000])
    loan_int_rate: float = Field(..., gt=0, examples=[11.5])
    cb_person_cred_hist_length: int = Field(..., ge=0, examples=[4])
    person_home_ownership: HomeOwnership = "RENT"
    loan_intent: LoanIntent = "MEDICAL"
    loan_grade: LoanGrade = "A"
    cb_person_default_on_file: DefaultFlag = "N"
    threshold_low: float = Field(0.35, ge=0.0, le=1.0)
    threshold_high: float = Field(0.70, ge=0.0, le=1.0)


class Contribution(BaseModel):
    feature: str
    value: float


class PredictionResponse(BaseModel):
    probability: float
    decision: Decision
    loan_percent_income: float
    contributions: List[Contribution]
    model: str
    threshold_low: float
    threshold_high: float


class ModelInfo(BaseModel):
    model: str
    algorithm: str
    feature_count: int
    features: List[str]
    metrics: Dict[str, str]
    params: Dict[str, object]
    default_thresholds: Dict[str, float]
    health: str


class HealthResponse(BaseModel):
    status: str
    model: str
