# PER_analysis

All files written using Python 3.8.10

Repository of files used for analysis of data in the Group Dynamics project at UT Austin
Included are:
- BasicAnalysis.py: this is an old code that produced the initial figures in Sp22.
- AnonyData.py: The file used to scrub traceable student results from raw data files.
- PCA.py: The file used to do Principle Component Analysis on the SAGE Survey.

To run SAGE analyis:
1.	In the Qualtrics survey, go to the “Data and Analysis” tab. 
2.	Then on the right side, use the “Export & Import” dropdown menu to Export Data. Make sure that “Use choice text” is selected and export the data as a CSV.
3.	Move the downloaded file to the same location as the .py file you intend to use.
4.	Make sure you have a folder called "ExportedFiles" in the same location. This is where any figures, csv files will be outputted.
5.	Run the desired .py file.
