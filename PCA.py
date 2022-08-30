# USED FOR ERROR TRACKING
import os
import traceback
import sys

# NORMAL PACKAGES
import readline, glob
import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

# FOR PRINCIPLE COMPONENT ANALYSIS
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import FactorAnalysis, PCA
import pingouin as pg

# WHEN I RUN THIS I HAVE A FOLDER WHERE ALL THE CREATED FILES GO CALLED 'ExportedFiles'
image_dir = 'ExportedFiles'

# ALLOWS THE USER TO TAB-AUTOCOMPLETE IN COMMANDLINE
def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

def save_fig(fig, figure_name):
    fname = os.path.expanduser(f'{image_dir}/{figure_name}')
    plt.savefig(fname + '.png')
    # plt.savefig(fname + '.svg')

def Analysis(df_norm):
    # CORRELATION MATRIX
    corrM = df_norm.corr()
    fig, ax = plt.subplots()
    plt.matshow(corrM)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix')
    save_fig(fig,'SAGE_CorrM')

    ## FIRST VALIDATE THE SAGE QUESTIONS
    not_SAGE = ['My group did higher quality work when my group members worked on tasks together.', 
    'My group did higher quality work when group members worked on different tasks at the same time.', 
    'You have a certain amount of physics intelligence, and you can’t really do much to change it.', 
    'Your physics intelligence is something about you that you can’t change very much.',
    'You can learn new things, but you can’t really change your basic physics intelligence.',
    'I prefer to take on tasks that I’m already good at.',
    'I prefer when one student regularly takes on a leadership role.',
    'I prefer when the leadership role rotates between students.']
    df_SAGE = df_norm.drop(not_SAGE, axis=1)

    # BARTLETT'S TEST
    chi_square_value, p_value = calculate_bartlett_sphericity(df_SAGE)
    print('Bartletts Chi Square =', chi_square_value, '; p-value: ', p_value)

    # KAISER-MEYER-OLKIN MEASURE OF SAMPLING ADEQUACY
    kmo_all, kmo_model = calculate_kmo(df_SAGE)
    print('KMO Measure of Sampling Adequacy: ', kmo_model)

    # # CRONBACH'S ALPHA TEST OF CONSISTENCY
    # print('Cronbachs alpha test of consistency: ', pg.cronbach_alpha(data=df_SAGE))

    # FACTOR ANALYSIS, N=4 AS PER KOUROS, ABRAMI
    fa = FactorAnalyzer(4, rotation='varimax', method='minres', use_smc=True)
    fa.fit(df_SAGE)
    FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
               method='minres', n_factors=4, rotation='varimax',
               rotation_kwargs={}, use_smc=True)
    m = pd.DataFrame(fa.loadings_)
    m.to_csv('ExportedFiles/SAGE_Comps.csv', encoding = "utf-8", index=True)

    fig, ax = plt.subplots()
    plt.imshow(fa.loadings_, cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    save_fig(fig, 'SAGE_FA')

    # PCA ANALYSIS FOR ALL QUESTIONS
    pca = PCA(n_components = 10)
    principalComponents = pca.fit_transform(df_norm)
    principalDf = pd.DataFrame(data = principalComponents)
    print('Number of components: 10,', 'Explained Variance Ratio:', pca.explained_variance_ratio_)
    print('PCA Singular Values: ', pca.singular_values_)

    # SCREE PLOT
    PC_values = np.arange(pca.n_components_) + 1
    fig, ax = plt.subplots()
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    save_fig(fig, 'Scree')    

    # FACTOR ANALYSIS, N=6
    fa = FactorAnalyzer(6, rotation='varimax', method='minres', use_smc=True)
    fa.fit(df_norm)
    FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
               method='minres', n_factors=6, rotation='varimax',
               rotation_kwargs={}, use_smc=True)
    m = pd.DataFrame(fa.loadings_)
    m.to_csv('ExportedFiles/SAGE_Comps.csv', encoding = "utf-8", index=True)

    fig, ax = plt.subplots()
    plt.imshow(fa.loadings_, cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    save_fig(fig, 'SAGE_PCA')

def Prepare_data(df):
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

    df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_Stats.csv', encoding = "utf-8", index=True)

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

    return df_norm

def Gender_differences(df):
    # Split based on gender
    dfM = df[df['Which gender(s) do you most identify (select all that apply)? - Selected Choice'] == 'Male'].copy()
    dfF = df[df['Which gender(s) do you most identify (select all that apply)? - Selected Choice'] == 'Female'].copy()

    # REMOVE THE DEMOGRAPHICS QUESTIONS
    dfM.drop(columns=dfM.columns[-4:], axis=1, inplace = True)
    dfF.drop(columns=dfF.columns[-4:], axis=1, inplace = True)

    # CALCULATE MEAN, STD DEV OF EACH COLUMN
    dfM_mean = dfM.mean()
    dfM_stderr = dfM.std()/np.sqrt(dfM.count())
    dfF_mean = dfF.mean()
    dfF_stderr = dfF.std()/np.sqrt(dfF.count())
    dfM_summary = pd.merge(dfM_mean.to_frame(), dfM_stderr.to_frame(), left_index = True, right_index=True)
    dfF_summary = pd.merge(dfF_mean.to_frame(), dfF_stderr.to_frame(), left_index = True, right_index=True)
    dfM_summary.rename(columns={'0_x': 'Mean (male)', '0_y': 'Std.Err. (male)',}, inplace = True)
    dfF_summary.rename(columns={'0_x': 'Mean (female)', '0_y': 'Std.Err. (female)'}, inplace = True)
    df_summary = pd.merge(dfM_summary, dfF_summary, left_index = True, right_index=True)
    significant = np.where( scipy.stats.t.sf( abs((df_summary['Mean (male)']-df_summary['Mean (female)'])/(np.sqrt(df_summary['Std.Err. (male)']**2 + df_summary['Std.Err. (female)']**2) )),df=8) <=0.1)
    print(df_summary.iloc[significant[0]])
    df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_GenSig.csv', encoding = "utf-8", index=True)

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

    # df_norm = Prepare_data(df)
    # Analysis(df_norm)
    Gender_differences(df)



# WHERE THE CODE IS ACTUALLY RUN
try:
    if __name__ == '__main__':
        main()
except Exception as err:
    traceback.print_exc(file=sys.stdout)
    input("Press Enter to exit...")


def repository():
    # PRINT ALL ROWS IN A DATAFRAME
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print()

    methods = [
        ('PCA', PCA()),
        ('Unrotated FA', FactorAnalysis()),
        ('Varimax FA', FactorAnalysis(rotation='varimax')),
    ]
    fig, axes = plt.subplots(ncols=len(methods))
    for ax, (method, f) in zip(axes, methods):
        f.set_params(n_components=n_comps)
        f.fit(df_SAGE)

        components = f.components_.T
        print('\n\n %s :\n' % method)
        print(components)

        ax.imshow(components, cmap="viridis", vmax=1.0, vmin=-1.0)
        ax.set_title(str(method))
    fig.suptitle('Factors')
    ax1=plt.gca() #get the current axes
    for PCM in ax1.get_children():
        if type(PCM) == matplotlib.image.AxesImage:
            break  
    plt.colorbar(PCM, ax=ax1)
    save_fig(fig, 'SAGE_PCA')

    # EXPORTS THE DATAFRAME TO AN ANONYMIZED VERSION AS A CSV
    df.to_csv('ExportedFiles/SAGE_Raw.csv', encoding = "utf-8", index=False)