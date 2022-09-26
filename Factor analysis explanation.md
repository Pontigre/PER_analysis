# Thorough explanation of factor analysis/principal component analysis

## Data
The easiest format to use is the CSV created by Qualtrics.
- Use the “Export & Import” dropdown menu to Export Data. Make sure that “Use choice text” is selected and export the data as a CSV.
This document will give information about the submission (IP address, start/end times, location) and the text responses for each submission of the survey. If there are, say, 60 responses, then the CSV should have 63 rows (60 responses + 3 header rows).

## Preparing data for analysis
1. First step is to read in all the relevant data. This excludes all of Qualtric's extra fluff by excluding any columns with spceific headers
- 'Start Date', 'End Date', 'Response Type', 'IP Address', 'Progress', 'Duration (in seconds)', 'Finished', 'Recorded Date', 'Response ID', 'Recipient Last Name', 'Recipient First Name', 'Recipient Email', 'External Data Reference', 'Location Latitude', 'Location Longitude', 'Distribution Channel', 'User Language'
2. Convert text responses to numerical resuults
- Strongly disagree = 1, ..., Strongly agree = 5 (5-point questions)
- Strongly disagree = 1, ..., Neither agree nor disagree = 3, ..., Strongly agree = 6 (6-point questions)
- Outputs results as SAGE_Raw.csv
3. For the sake of this initial analysis, we remove the four demographic questions. 
4. Calculate statistics on each questions
- Mean, standard deviation
- Percentages of SD+D, N, A+SA
- Outputs results as SAGE_Stats.csv
5. Invert negatively worded questions
- Strongly disagree = 5, ..., Strongly agree = 1
- Check whether the distributions are the same using Mann-Whitney U-test (non-parametric, different sized arrays). Similar distributions should give p>0.5.
- If the same, then combine the two columns together. If different, drop both columns (if not then will show up as partial responses).
6. Remove any partial responses
7. Renormalize data so that 5- and 6-point responses are equal
- $\frac{x-\mu}{\sigma}$

## Confirmatory Factor Analysis

## Correlation Matrix

## Exploaratory Factor Analysis

## Principal Component Analysis
