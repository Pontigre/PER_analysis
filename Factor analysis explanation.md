# Explanation of factor analysis/principal component analysis

## Data
The easiest format to use is the CSV created by Qualtrics.
- Use the “Export & Import” dropdown menu in Data & Analysis to Export Data. Make sure that “Use choice text” is selected and export the data as a CSV.
This document will give information about the submission (IP address, start/end times, location) and the text responses for each submission of the survey. If there are, say, 60 responses, then the CSV should have 63 rows (60 responses + 3 header rows).

## Preparing data for analysis
1. First step is to read in all the relevant data. This excludes all of Qualtric's extra fluff by excluding any columns with spceific headers
- 'Start Date', 'End Date', 'Response Type', 'IP Address', 'Progress', 'Duration (in seconds)', 'Finished', 'Recorded Date', 'Response ID', 'Recipient Last Name', 'Recipient First Name', 'Recipient Email', 'External Data Reference', 'Location Latitude', 'Location Longitude', 'Distribution Channel', 'User Language'
2. Convert text responses to numerical resuults
- Strongly disagree = 1, ..., Strongly agree = 5 (5-point questions)
- Strongly disagree = 1, ..., Neither agree nor disagree = 3, ..., Strongly agree = 6 (6-point questions)
- Outputs results as SAGE_Raw.csv
4. Calculate statistics on each question
- Mean, standard deviation
- Percentages of SD+D, N, A+SA
- Outputs results as SAGE_Stats.csv
5. Invert negatively worded questions
- Strongly disagree = 5, ..., Strongly agree = 1
- Check whether the distributions are the same using Mann-Whitney U-test (non-parametric, different sized arrays). Similar distributions should give p>0.5.
- If the same, then combine the two columns together. If different, drop both columns (if not then will show up as partial responses).
6. Remove any partial responses
7. Renormalize 6-point Likert questions to 5-point scale ($\times 5/6$)

## Confirmatory Factor Analysis
Uses only the questions that correspond to factored questions in Kouros and Abrami 2006.
- Each question is loaded specifically to its corresponding factor based on a dictionary.
- Outputs the loadings to SAGE_CFA.csv

## Correlation Matrix
Uses the Spearman method to compute pairwise correlations between items.
- Outputs to SAGE_CorrM.csv, .png

## Exploratory Factor Analysis
Follows Section II.A from Eaton et al. 2019
1. Calculate the Kaiser-Meyer-Olkin (KMO) values for every item. If any items have a KMO below the cutoff value, then the item with the lowest value is removed and the step is repeated. KMO values above 0.6 are kept, though above 0.8 are preferred.
2. Check whether the items can be factored using Bartlett's test of sphericity. A low p-score indicates that factor analysis can be performed.
3. Calculate the EFA model using factoring and a specified number of factors.
4. Calculate the commonalities, which are the proportion of the item's variance explained by the factors. If any item is below the cutoff (<0.4), then the item with the lowest value is dropped and then restart at Step 1.
5. Calculate the item loadings. If there are items that fail to load to any mfactor, then remove the item with the smallest max loading and then restart at Step 1.
6. Create a model by placing each item onto the factor that contains the item's largest loading. If any items load equally onto more than one factor, then add to all factors where this is the case.
7. Fit this model to the original data using CFA and extract a fit statistic (confirmatory fit index, Akaike information criterion, or similar).
8. Change the number of factors and repeat the above steps.
9. Plot the fit statistic vs the number of factors. The model with the local minimum index is the preferred model.