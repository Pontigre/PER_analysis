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
import factor_analyzer.factor_analyzer as fa
from sklearn.decomposition import PCA
import pingouin as pg

# WHEN I RUN THIS I HAVE A FOLDER WHERE ALL THE CREATED FILES GO CALLED 'ExportedFiles'
image_dir = 'ExportedFiles'

# ALLOWS THE USER TO TAB-AUTOCOMPLETE IN COMMANDLINE
def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

# STORAGE OF USEFUL FUNCTIONS
def RepositoryofFunctions():
    # GIVE ALL ROWS WITH NO STUDENT (DOUBLE MAJOR)
    print(df.loc[df['Student'].isnull()])

    # PRINT ROW OF A GIVEN STUDENT
    print(df.loc[df['Student'] == 'TEST']) 

    # PRINT ALL ROWS IN A DATAFRAME
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print()

    # CAN SEPARATE DATA BY SPECIFIC CONDITIONS
    data_NSc = data.loc[data['Sch'].str.contains('E')]

# CREATE A FUNCCTION TO SAVE ANY FIGURE IN THE DESIRED DIRECTORY
# CAN ALSO CREATE AN SVG FOR FIGURE TWEAKING LATER
def save_fig(fig, figure_name):
    fname = os.path.expanduser(f'{image_dir}/{figure_name}')
    plt.savefig(fname + '.png')
    # plt.savefig(fname + '.svg')

def main():
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete)
    my_file = input('SAGE Filename: ')

    # READ IN DATA FROM SAGE QULATRICS SURVEY BASED ON THE CSV COLUMN NAMES
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

    # REMOVE THE DEMOGRAPHICS QUESTIONS
    df.drop(columns=df.columns[-4:], axis=1, inplace = True)

    # CALCULATE MEAN, STD DEV OF EACH COLUMN
    df_mean = df.mean()
    df_stddev = df.std()
    df_summary = pd.merge(df_mean.to_frame(), df_stddev.to_frame(), left_index = True , right_index =True)
    df_summary.rename(columns={'0_x': 'Mean', '0_y': 'Std.Dev.'}, inplace = True)

    # COUNT THE AMOUNT OF SD+D (1,2), N (3), AND SA+A (4,5) IN EACH COLUMN
    my_list = list(df)
    col_list = [x for x in my_list if x not in Phys_Int_Cols]
    df_summary['SD+D'] = np.nan
    df_summary['N'] = np.nan
    df_summary['SA+A'] = np.nan
    for i in col_list:
        s = df[i].value_counts(normalize=True).sort_index().rename_axis('unique_values').reset_index(name='counts')
        df_summary.at[i,'SD+D'] = s[(s.unique_values == 1) | (s.unique_values == 2)].sum()['counts']
        df_summary.at[i,'N'] = s[s.unique_values == 3].sum()['counts']
        df_summary.at[i,'SA+A'] = s[(s.unique_values == 4) | (s.unique_values == 5)].sum()['counts']

    for i in Phys_Int_Cols:
        s = df[i].value_counts(normalize=True).sort_index().rename_axis('unique_values').reset_index(name='counts')
        df_summary.at[i,'SD+D'] = s[(s.unique_values == 1) | (s.unique_values == 2)].sum()['counts']
        df_summary.at[i,'N'] = s[(s.unique_values == 3) | (s.unique_values == 4)].sum()['counts']
        df_summary.at[i,'SA+A'] = s[(s.unique_values == 5) | (s.unique_values == 6)].sum()['counts']

    df_summary.to_csv('ExportedFiles/SAGE_Stats.csv', encoding = "utf-8", index=True)

    # INVERT NEGATIVELY WORDED QUESTIONS, THEN COMBINE COLUMNS WITH POSITIVELY WORDED QUESTIONS
    Neg_List = [
    'The work takes less time to complete when I work with other students.',
    'My group members do not respect my opinions.',
    'I prefer when no one takes on a leadership role.',
    'I do not let the other students do most of the work.',
    'I prefer to take on tasks that will help me better learn the material.']
    for i in Neg_List:
        df[i].replace([1,2,4,5],[5,4,2,1],inplace=True)
    df['The work takes more time to complete when I work with other students.'] = df[['The work takes more time to complete when I work with other students.', ###
    'The work takes less time to complete when I work with other students.']].sum(axis=1)
    df.drop(['The work takes less time to complete when I work with other students.'], axis=1, inplace=True)
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

    # REMOVE PARTIAL RESPONSES
    df.dropna(axis=0, how='any', inplace = True)

    # RENORMALIZE DATA SUCH THAT 5-POINT AND 6-POINT ARE EQUAL
    df_norm = ((df - df.mean())/df.std()).astype(float)

    # EXPORTS THE DATAFRAME TO AN ANONYMIZED VERSION AS A CSV
    df.to_csv('ExportedFiles/SAGE_anony.csv', encoding = "utf-8", index=False)

    # CORRELATION MATRIX
    corrM = df_norm.corr()
    fig, ax = plt.subplots()
    plt.matshow(corrM)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix')
    save_fig(fig,'CorrM')

    ## FIRST ATTEMPT AT PCA
    # BARTLETT'S TEST
    chi_square_value, p_value = fa.calculate_bartlett_sphericity(df_norm)
    print('Bartletts Chi Square =', chi_square_value, '; p-value: ', p_value)

    # KAISER-MEYER-OLKIN MEASURE OF SAMPLING ADEQUACY
    kmo_all, kmo_model = fa.calculate_kmo(df_norm)
    print('KMO Measure of Sampling Adequacy: ', kmo_model)

    # CRONBACH'S ALPHA TEST OF CONSISTENCY
    print('Cronbachs alpha test of consistency: ', pg.cronbach_alpha(data=df_norm))

    # PRINCIPLE COMPONENT ANALYSIS, N=4 AS PER KOUROS, ABRAMI
    # NEEDS TO CHANGE SINCE WE'VE ADDED QUESTIONS
    pca = PCA(n_components=4)
    pca.fit(df_norm)
    print('Number of components: 4', 'Explained Variance Ratio: ', pca.explained_variance_ratio_)
    print('PCA Singular Values: ', pca.singular_values_)

    # SCREE PLOT
    PC_values = np.arange(pca.n_components_) + 1
    fig, ax = plt.subplots()
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    save_fig(fig, 'Scree')    

# WHERE THE CODE IS ACTUALLY RUN
try:
    if __name__ == '__main__':
        main()
except Exception as err:
    traceback.print_exc(file=sys.stdout)
    input("Press Enter to exit...")
