from ast import literal_eval
import pandas as pd
import pandas._typing as typing

jobs_df = pd.read_csv('random/freelancer_job_postings.csv')
jobbers_df = pd.read_csv('upwork/upwork_data_scientists.csv')

# >>> print(jobs_df.keys())
#   [
#       'projectId', 'job_title', 'job_description', 'tags', 'client_state',
#       'client_country', 'client_average_rating', 'client_review_count',
#       'min_price', 'max_price', 'avg_price', 'currency', 'rate_type'
#   ]

# >>> print(jobbers_df.keys())
#   [
#       'country', 'description', 'hourlyRate', 'jobSuccess', 'locality',
#       'name', 'skills', 'title', 'totalHours', 'totalJobs'
#   ]

def get_values_distribution(df: pd.DataFrame, label: str, generator: typing.AggFuncType) -> list:
    _set = []
    for items in df[label].apply(generator).values:
        for item in items:
            _set.append(item.lower().strip())

    return _set

job_tags_set = set(get_values_distribution(jobs_df, 'tags', literal_eval))
jobber_skills_set = set(get_values_distribution(jobbers_df, 'skills', lambda x: x.split("|")))

common_artifacts_set = job_tags_set.intersection(jobber_skills_set)
total_artifacts_set = job_tags_set.union(jobber_skills_set)
