from dataclasses import dataclass
from typing import List

@dataclass
class Job:
    projectId: str
    job_title: str
    job_description: str
    tags: List[str]
    client_state: str
    client_country: str
    client_average_rating: float
    client_review_count: int
    min_price: float
    max_price: float
    average_price: float
    currency: str
    rate_type: str

@dataclass
class Jobber:
    jobberId: str
    name: str
    about: str
    rate_price: float
    rate_type: str
    currency: str
    skills: List[str]
    locality: str
    total_jobs: int
    average_job_rating: float
