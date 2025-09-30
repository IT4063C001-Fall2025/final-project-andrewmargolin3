#!/usr/bin/env python
# coding: utf-8

# # Mental Health Analysis📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# The increasing prevelance of mental illnesses without understanding all the causing of the illnesses which reduces people's quality of life.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# I am trying to answer the the question of what factors can cause increases in mental illness and how can we use that knowledge to help others?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# An answer to me would be prevelant enviromental factors that can cause increased rates of mental illness and how we could reduce those enviromental factors.

# ![graphmap](./assets/graphmap.jpeg)

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# https://mhanational.org/research-and-reports/
# 
# https://www.cdc.gov/mental-health/about-data/suicidal-thoughts-and-behavior.html
# 
# https://www.cdc.gov/mental-health/about-data/conditions-care.html
# 
# I am going to relate this data by looking at the corralation of anxiety and depression rates among different groups and how that affects suicide rate among those same groups and use that data in tandem with analysis on how things like region and living area (urban, rural, etc) have different rates of mental illness and determine what differences could cause that.
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# My approach will be to use data visualizations such as bar graphs and map visualizations to display variations of different factors such as suicide rates, depression/anxiety rates. This would make understanding various factors at once much easier because you can see how they would connect.
# 

# In[ ]:


# Start your code here
import pandas as pd
from tabula import read_pdf
mha_state_of_mental_health_report_2024 = read_pdf("https://mhanational.org/wp-content/uploads/2024/12/2024-State-of-Mental-Health-in-America-Report.pdf",pages="all")
mha_state_of_mental_health_report_2023 = read_pdf("https://mhanational.org/wp-content/uploads/2024/12/2023-State-of-Mental-Health-in-America-Report.pdf",pages="all")




# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

