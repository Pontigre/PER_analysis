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
import seaborn as sns

# FOR PRINCIPLE COMPONENT ANALYSIS
import factor_analyzer
from factor_analyzer import (ConfirmatoryFactorAnalyzer,ModelSpecificationParser)
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

def SAGE_validation(df_norm):
    # FIRST VALIDATE THE SAGE QUESTIONS
    not_SAGE = ['My group did higher quality work when my group members worked on tasks together.', 
    'My group did higher quality work when group members worked on different tasks at the same time.', 
    'You have a certain amount of physics intelligence, and you can’t really do much to change it.', 
    'Your physics intelligence is something about you that you can’t change very much.',
    'You can learn new things, but you can’t really change your basic physics intelligence.',
    'I prefer to take on tasks that I’m already good at.',
    'I prefer when one student regularly takes on a leadership role.',
    'I prefer when the leadership role rotates between students.',

    # ADDED BECAUSE REVERSE CODING IS QUESTIONABLE
    # 'The work takes more time to complete when I work with other students.',
    # 'My group members respect my opinions.',
    # 'I let the other students do most of the work.'
    ]
    df_SAGE = df_norm.drop(not_SAGE, axis=1)

    not_cfa = ['When I work in a group, I end up doing most of the work.', 'I do not think a group grade is fair.', 'I try to make sure my group members learn the material.', 
    'When I work with other students the work is divided equally.', 'When I work with other students, we spend too much time talking about other things.']
    df_SAGE_cfa = df_SAGE.drop(not_cfa, axis=1)

    # CONFIRMATORY FACTOR ANALYSIS

    # FIRST DEFINE WHICH QUESTIONS SHOULD READ INTO EACH FACTOR, TAKEN FROM KOUROS AND ABRAMI 2006
    model_dict = {'F1': ['When I work in a group I do higher quality work.', 'The material is easier to understand when I work with other students.', 'My group members help explain things that I do not understand.', 
                        'I feel working in groups is a waste of time.', 'The work takes more time to complete when I work with other students.', 'The workload is usually less when I work with other students.'], 
    'F2': ['My group members respect my opinions.', 'My group members make me feel that I am not as smart as they are.', 'My group members do not care about my feelings.',
            'I feel I am part of what is going on in the group.', 'When I work in a group, I am able to share my ideas.'], 
    'F3': ['Everyone’s ideas are needed if we are going to be successful.', 'We cannot complete the assignment unless everyone contributes.', 'I let the other students do most of the work.',
            'I also learn when I teach the material to my group members.', 'I learn to work with students who are different from me.'], 
    'F4': ['I become frustrated when my group members do not understand the material.', 'I have to work with students who are not as smart as I am.']
    }

    model_spec = ModelSpecificationParser.parse_model_specification_from_dict(df_SAGE_cfa,model_dict)

    cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    cfa.fit(df_SAGE_cfa)

    df_cfa = pd.DataFrame(cfa.loadings_,index=model_spec.variable_names)
    df_cfa.round(decimals = 4).to_csv('ExportedFiles/SAGE_CFA.csv', encoding = "utf-8", index=True)
    trunc_cfa = np.ma.masked_where(abs(df_cfa) < 0.01, df_cfa)
    fig, ax = plt.subplots()
    plt.imshow(trunc_cfa, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.tight_layout()
    save_fig(fig, 'SAGE_CFA')
    plt.clf()

    # CORRELATION MATRIX
    corrM = df_SAGE.corr(method='spearman')
    corrM.round(decimals = 4).to_csv('ExportedFiles/SAGE_CorrM.csv', encoding = "utf-8", index=True)

    truncM = corrM[abs(corrM)>=0.4]
    labels = list(df_SAGE)
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(ax=ax, data=corrM,  vmin=-1, vmax=1, annot=False, cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title('Correlation Matrix')
    save_fig(fig,'SAGE_CorrM')
    plt.clf()

    fig, ax = plt.subplots()
    plt.title('Correlation Matrix')
    heatmap = sns.heatmap(ax=ax, data=truncM,  vmin=-1, vmax=1, annot=False, cmap='viridis', xticklabels=labels, yticklabels=labels)
    save_fig(fig,'SAGE_CorrM_sig')
    plt.clf()

    # BARTLETT'S TEST
    chi_square_value, p_value = calculate_bartlett_sphericity(df_SAGE)
    print('Bartletts Chi Square =', chi_square_value, '; p-value: ', p_value)

    # KAISER-MEYER-OLKIN MEASURE OF SAMPLING ADEQUACY
    kmo_all, kmo_model = calculate_kmo(df_SAGE)
    print('KMO Measure of Sampling Adequacy: ', kmo_model)

    # # CRONBACH'S ALPHA TEST OF CONSISTENCY (test)
    # print('Cronbachs alpha test of consistency: ', pg.cronbach_alpha(data=df_SAGE))

    # EFA
    efa = FactorAnalyzer(rotation=None)
    efa.fit(df_SAGE)
    ev, v = efa.get_eigenvalues()
    print('FA Eigenvalues:', ev)

    fig, ax = plt.subplots()
    plt.plot(ev, '.-', linewidth=2, color='blue')
    plt.hlines(1, 0, 22, linestyle='dashed')
    plt.title('Factor Analysis Scree Plot')
    plt.xlabel('Factor')
    plt.ylabel('Eigenvalue')
    plt.xlim(-0.5,22)
    plt.ylim(0,5.5)
    save_fig(fig, 'SAGE_Scree')
    plt.clf()

    # Based on the scree plot and Kaiser criterion, n=6 (or 7)
    fa = FactorAnalyzer(n_factors=6, rotation='varimax')
    fa.fit(df_SAGE)
    m = pd.DataFrame(fa.loadings_)
    m.to_csv('ExportedFiles/SAGE_EFA.csv', encoding = "utf-8", index=True)

    fig, ax = plt.subplots()
    plt.imshow(m, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.tight_layout()
    save_fig(fig, 'SAGE_EFA')
    plt.clf()

    truncm = m[abs(m)>=0.5]
    fig, ax = plt.subplots()
    plt.imshow(truncm, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.tight_layout()
    save_fig(fig, 'SAGE_EFA_0.5')
    plt.clf()

    # PCA ANALYSIS
    pca1 = PCA(n_components=len(df_SAGE.columns)-1)
    pca1.fit(df_SAGE)
    fig, ax = plt.subplots()
    fig_x = np.arange(pca1.n_components) + 1
    fig_y = 100*pca1.explained_variance_ratio_
    plt.bar(fig_x, fig_y, label='Individual explained variance')
    plt.step(fig_x, fig_y.cumsum(), where = 'post', c='red', label='Cumulative explained variance')
    plt.title('PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Eplained variance percentage')
    plt.legend()
    save_fig(fig, 'PCA_full_var')
    plt.clf()

    # Specifying the variance to be >=0.75
    pca = PCA(n_components=0.75)
    pca.fit(df_SAGE)
    print('Number of components for variance >= 0.75:', pca.n_components_)

    fig, ax = plt.subplots()
    fig_x = np.arange(pca.n_components_) + 1
    fig_y = 100*pca.explained_variance_ratio_
    plt.bar(fig_x, fig_y, label='Individual explained variance')
    plt.step(fig_x, fig_y.cumsum(), where = 'post', c='red', label='Cumulative explained variance')
    plt.title('PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Eplained variance percentage')
    save_fig(fig, 'PCA_var')
    plt.clf()

    # SCREE PLOT
    PC_values = np.arange(pca.n_components_) + 1
    fig, ax = plt.subplots()
    plt.plot(pca.explained_variance_, '.-', linewidth=2, color='blue')
    plt.title('PCA Scree Plot')
    plt.xlabel('Principal Component')
    plt.hlines(1, 0, 22, linestyle='dashed')
    plt.xlim(-0.5,22)
    plt.ylim(0,5.5)
    plt.ylabel('Eigenvalue')
    save_fig(fig, 'PCA_Scree')
    plt.clf()

    mm = pca.components_.T
    fig, ax = plt.subplots()
    plt.imshow(mm, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.tight_layout()
    save_fig(fig,'SAGE_PCA')

    trunc = np.ma.masked_where(abs(mm) < 0.5, mm)
    fig, ax = plt.subplots()
    plt.imshow(trunc, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.tight_layout()
    save_fig(fig, 'SAGE_PCA_0.5')
    plt.clf()

def FullAnalysis(df_norm):
    # PCA ANALYSIS FOR ALL QUESTIONS
    pca = PCA(n_components = 10)
    principalComponents = pca.fit_transform(df_norm)
    principalDf = pd.DataFrame(data = principalComponents)
    print('Number of components: 6,', 'Explained Variance Ratio:', pca.explained_variance_ratio_)
    print('PCA Singular Values: ', pca.singular_values_)

    # SCREE PLOT
    PC_values = np.arange(pca.n_components_) + 1
    fig, ax = plt.subplots()
    plt.plot(PC_values, pca.explained_variance_ratio_, '.-', linewidth=2, color='blue')
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

    # SAVE RAW DATA
    df.to_csv('ExportedFiles/SAGE_Raw.csv', encoding = "utf-8", index=False)

    # CALCULATE MEAN, STD DEV OF EACH COLUMN
    df_mean = df.mean()
    df_stddev = df.std()
    df_summary = pd.merge(df_mean.to_frame(), df_stddev.to_frame(), left_index = True , right_index =True)
    df_summary.rename(columns={'0_x': 'Mean', '0_y': 'Std.Dev.'}, inplace = True)

    # COUNT THE AMOUNT OF SD+D (1,2), N (3), AND SA+A (4,5) IN EACH COLUMN
    Phys_Int_Cols = [col for col in df.columns if 'physics intelligence' in col]
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
    'I do not let the other students do most of the work.']#,
    #'I prefer to take on tasks that will help me better learn the material.']
    for i in Neg_List:
        df[i].replace([1,2,4,5],[5,4,2,1],inplace=True)

    # CHECK IF CONSITENCY BETWEEN INVERSELY WORDED QUESTIONS
    x = np.array(df['The work takes more time to complete when I work with other students.'].dropna(), dtype=np.uint8)
    y = np.array(df['The work takes less time to complete when I work with other students.'].dropna(), dtype=np.uint8)
    res = scipy.stats.mannwhitneyu(x,y,nan_policy='omit')
    # print('The work takes more time to complete when I work with other students.:', x.mean().round(decimals = 2), x.std().round(decimals = 2))
    # print('The work takes less time to complete when I work with other students.:', y.mean().round(decimals = 2), y.std().round(decimals = 2))
    # print(res) #Distinguishable

    df['The work takes more time to complete when I work with other students.'] = df[['The work takes more time to complete when I work with other students.', ###
    'The work takes less time to complete when I work with other students.']].sum(axis=1)
    df.drop(['The work takes less time to complete when I work with other students.'], axis=1, inplace=True)

    x = np.array(df['My group members respect my opinions.'].dropna(), dtype=np.uint8)
    y = np.array(df['My group members do not respect my opinions.'].dropna(), dtype=np.uint8)
    res = scipy.stats.mannwhitneyu(x,y,nan_policy='omit')
    # print('My group members respect my opinions.:', x.mean().round(decimals = 2), x.std().round(decimals = 2))
    # print('My group members do not respect my opinions.:', y.mean().round(decimals = 2), y.std().round(decimals = 2))
    # print(res) # p=0.225

    df['My group members respect my opinions.'] = df[['My group members respect my opinions.','My group members do not respect my opinions.']].sum(axis=1)
    df.drop(['My group members do not respect my opinions.'], axis=1, inplace=True)

    x = np.array(df['I prefer when one student regularly takes on a leadership role.'].dropna(), dtype=np.uint8)
    y = np.array(df['I prefer when no one takes on a leadership role.'].dropna(), dtype=np.uint8)
    res = scipy.stats.mannwhitneyu(x,y,nan_policy='omit')
    # print('I prefer when one student regularly takes on a leadership role.:', x.mean().round(decimals = 2), x.std().round(decimals = 2))
    # print('I prefer when no one takes on a leadership role.:', y.mean().round(decimals = 2), y.std().round(decimals = 2))
    # print(res) # p=0.523

    df['I prefer when one student regularly takes on a leadership role.'] = df[['I prefer when one student regularly takes on a leadership role.',
    'I prefer when no one takes on a leadership role.']].sum(axis=1)
    df.drop(['I prefer when no one takes on a leadership role.'], axis=1, inplace=True)

    x = np.array(df['I let the other students do most of the work.'].dropna(), dtype=np.uint8)
    y = np.array(df['I do not let the other students do most of the work.'].dropna(), dtype=np.uint8)
    res = scipy.stats.mannwhitneyu(x,y,nan_policy='omit')
    # print('I let the other students do most of the work.:', x.mean().round(decimals = 2), x.std().round(decimals = 2))
    # print('I do not let the other students do most of the work.:', y.mean().round(decimals = 2), y.std().round(decimals = 2))
    # print(res) # p = 0.135
    df['I let the other students do most of the work.'] = df[['I let the other students do most of the work.',
    'I do not let the other students do most of the work.']].sum(axis=1)
    df.drop(['I do not let the other students do most of the work.'], axis=1, inplace=True)

    Take_on_tasks_cols = [col for col in df.columns if 'take on tasks' in col]
    x = np.array(df[Take_on_tasks_cols[0]].dropna(), dtype=np.uint8)
    y = np.array(df[Take_on_tasks_cols[1]].dropna(), dtype=np.uint8)
    res = scipy.stats.mannwhitneyu(x,y,nan_policy='omit')
    print(Take_on_tasks_cols[0], x.mean().round(decimals = 2), x.std().round(decimals = 2))
    print(Take_on_tasks_cols[1], y.mean().round(decimals = 2), y.std().round(decimals = 2))
    print(res) # Same if not reverse-coded

    df[Take_on_tasks_cols[0]] = df[Take_on_tasks_cols].sum(axis=1)
    df.drop(Take_on_tasks_cols[1], axis=1, inplace = True)

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
    dfM_med = dfM.median()
    dfM_stderr = dfM.std()/np.sqrt(dfM.count())
    dfF_mean = dfF.mean()
    dfF_med = dfF.median()
    dfF_stderr = dfF.std()/np.sqrt(dfF.count())
    dfM_summary = pd.merge(dfM_mean.to_frame(), dfM_stderr.to_frame(), left_index = True, right_index=True)
    dfF_summary = pd.merge(dfF_mean.to_frame(), dfF_stderr.to_frame(), left_index = True, right_index=True)
    dfM_summary.rename(columns={'0_x': 'Mean (male)', '0_y': 'Std.Err. (male)',}, inplace = True)
    dfF_summary.rename(columns={'0_x': 'Mean (female)', '0_y': 'Std.Err. (female)'}, inplace = True)
    dfM_summary['Median (male)'] = dfM_med
    dfF_summary['Median (female)'] = dfF_med
    df_summary = pd.merge(dfM_summary, dfF_summary, left_index = True, right_index=True)
    # significant = np.where( scipy.stats.t.sf( abs((df_summary['Mean (male)']-df_summary['Mean (female)'])/(np.sqrt(df_summary['Std.Err. (male)']**2 + df_summary['Std.Err. (female)']**2) )),df=8) <=0.1)
    # print(df_summary.iloc[significant[0]])
    df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_GenSig.csv', encoding = "utf-8", index=True)
    dfM.to_csv('ExportedFiles/SAGE_M.csv')
    dfF.to_csv('ExportedFiles/SAGE_F.csv')

    # stat, p = scipy.stats.ranksums(df_summary['Median (male)'],df_summary['Median (female)'], nan_policy='omit')
    # print(stat, p)
    # for i in df_summary.index:
    #     stat, p = scipy.stats.ranksums(dfM[i],dfF[i])
    #     print(i, p)

def main():
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete)
    my_file = input('SAGE Filename: ')

    # READ IN DATA FROM SAGE QUALTRICS SURVEY BASED ON THE CSV COLUMN NAMES
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

    # Gender_differences(df) # This needs to go before Prepare_data() because that changes df
    df_norm = Prepare_data(df) # This removes demographic questions, calculates averages and statistics, and combines inversely worded questions into one
    SAGE_validation(df_norm) # Validation factor analysis, exploratory factor analysis, and principal component analysis on only the questions taken from SAGE
    # FullAnalysis(df_norm)

# WHERE THE CODE IS ACTUALLY RUN
try:
    if __name__ == '__main__':
        main()
except Exception as err:
    traceback.print_exc(file=sys.stdout)
    input("Press Enter to exit...")
