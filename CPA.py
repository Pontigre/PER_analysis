# USED FOR ERROR TRACKING
import os
import traceback
import sys

# NORMAL PACKAGES
import readline, glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# FOR PRINCIPLE COMPONENT ANALYSIS
from sklearn.decomposition import PCA

image_dir = 'ExportedFiles'

def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

def RepositoryofFunctions():
    # GIVE ALL ROWS WITH NO STUDENT (DOUBLE MAJOR)
    print(df.loc[df['Student'].isnull()])

    # PRINT ROW OF A GIVEN STUDENT
    print(df.loc[df['Student'] == 'TEST']) 

    # PRINT ALL ROWS IN A DATAFRAME
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print()

    # CAN SEPARATE DATA BY SPECIFIC CONDITIONS
    # data_NSc = data.loc[data['Sch'].str.contains('E')]

def save_fig(fig, figure_name):
    fname = os.path.expanduser(f'{image_dir}/{figure_name}')
    plt.savefig(fname + '.png')
    # plt.savefig(fname + '.svg')

def main():
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete)
    my_file = input('SAGE Filename: ')

    # READ IN DATA FROM SAGE BASED ON THE CSV COLUMN NAMES
    headers = [*pd.read_csv(my_file, nrows=1)]
    ExcludedHeaders = ['Start Date', 'End Date', 'Response Type', 'IP Address', 'Progress', 'Duration (in seconds)',
    'Finished', 'Recorded Date', 'Response ID', 'Recipient Last Name', 'Recipient First Name',
    'Recipient Email', 'External Data Reference', 'Location Latitude', 'Location Longitude', 'Distribution Channel', 'User Language']
    df = pd.read_csv(my_file, encoding = "utf-8", usecols=lambda x: x not in ExcludedHeaders, skiprows=1)
    df = df.drop([0])

    # CONVERT RESPONSES TO NUMERICAL VALUE (I DON'T TRUST QUALTRICS TO DO THIS WELL)
    # 6-point Likert first, then the rest
    Phys_Int_Cols = [col for col in df.columns if 'physics intelligence' in col]
    for i in Phys_Int_Cols:
        df.loc[df[i] == 'Strongly disagree', i] = 1
        df.loc[df[i] == 'Disagree', i] = 2
        df.loc[df[i] == 'Somewhat disagree', i] = 3
        df.loc[df[i] == 'Somewhat agree', i] = 4
        df.loc[df[i] == 'Agree', i] = 5
        df.loc[df[i] == 'Strongly agree', i] = 6

    df.mask(df == 'Strongly disagree', 1, inplace = True)
    df.mask(df == 'Somewhat disagree', 2, inplace = True)
    df.mask(df == 'Neither agree nor disagree', 3, inplace = True)
    df.mask(df == 'Somewhat agree', 4, inplace = True)
    df.mask(df == 'Strongly agree', 5, inplace = True)

    # INVERT NEGATIVELY WORDED QUESTIONS, THEN COMBINE WITH POSITIVELY WORDED QUESTIONS
    ### NEEDS TO CHANGE WITH NEW LANGUAGE ###
    Neg_List = [
    'The work takes less time to complete when I work with other students.', ####
    'My group members do not respect my opinions.',
    'I prefer when no one takes on a leadership role.',
    'I do not let the other students do most of the work.',
    'I prefer to take on tasks that will help me better learn the material.']
    for i in Neg_List:
        df[i].replace([1,2,4,5],[5,4,2,1],inplace=True)
    df['The work takes more time to complete when I work with other students.'] = df[['The work takes more time to complete when I work with other students.', ###
    'The work takes less time to complete when I work with other students.']].sum(axis=1) ####
    df.drop(['The work takes less time to complete when I work with other students.'], axis=1, inplace=True) ###
    df['My group members respect my opinions.'] = df[['My group members respect my opinions.','My group members do not respect my opinions.']].sum(axis=1)
    df.drop(['My group members do not respect my opinions.'], axis=1, inplace=True)
    df['I prefer when one student regularly takes on a leadership role.'] = df[['I prefer when one student regularly takes on a leadership role.',
    'I prefer when no one takes on a leadership role.']].sum(axis=1)
    df.drop(['I prefer when no one takes on a leadership role.'], axis=1, inplace=True)
    df['I let the other students do most of the work.'] = df[['I let the other students do most of the work.',
    'I do not let the other students do most of the work.']].sum(axis=1)
    df.drop(['I do not let the other students do most of the work.'], axis=1, inplace=True)
    Take_on_tasks_cols = [col for col in df.columns if 'take on tasks' in col]
    df[Take_on_tasks_cols[0]] = df[Take_on_tasks_cols].sum(axis=1)
    df.drop(['I prefer to take on tasks that will help me better learn the material.'], axis=1, inplace = True)

    # REMOVE THE DEMOGRAPHICS QUESTIONS
    df.drop(columns=df.columns[-4:], axis=1, inplace = True)

    # REMOVE PARTIAL RESPONSES
    df.dropna(axis=0, how='any', inplace = True)

    # TAKE AVERAGES AND STD.DEV., PERCENTAGE SA+A, U, D+SD
    # df_norm = (df - df.mean())/df.std()
    # pca = PCA(n_components=4)
    # pca.fit(df_norm)

    # # Reformat and view results
    # loadings = pandas.DataFrame(pca.components_.T, columns=['PC%s' % _ for _ in range(len(df_norm.columns))],index=df.columns)
    # print(loadings)

    # fig, ax = plt.subplots()
    # plt.plot(pca.explained_variance_ratio_)
    # plt.ylabel('Explained Variance')
    # plt.xlabel('Components')
    # save_fig(fig, 'PCA')

    df.to_csv('ExportedFiles/SAGE_anony.csv', encoding = "utf-8", index=False)

try:
    if __name__ == '__main__':
        main()
except Exception as err:
    traceback.print_exc(file=sys.stdout)
    input("Press Enter to exit...")