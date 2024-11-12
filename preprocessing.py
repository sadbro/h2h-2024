from db import *
from interfaces import *
from collections import Counter
from models import answer

# Data cleaning and preprocessing

skills_d = Counter(get_values_distribution(jobbers_df, 'skills', lambda x: x.split("|")))
tags_d = Counter(get_values_distribution(jobs_df, 'tags', literal_eval))

# print(tags_d.most_common(5), skills_d.most_common(5), sep='\n')

ex1 = jobs_df['job_description'].values[0]
question = "Instruct: Get all the skill tags from the following description in a array format: {}\nOutput:\n".format(ex1)
ans = answer(question).split("\n")[2]
print(ans)
