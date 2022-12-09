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

# FOR FACTOR ANALYSIS
import factor_analyzer
from factor_analyzer import (ConfirmatoryFactorAnalyzer,ModelSpecificationParser)
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
# from pingouin import cronbach_alpha as al ##Do not work with python3.11
# from advanced_pca import CustomPCA ##Do not work with python3.11

# WHEN I RUN THIS I HAVE A FOLDER WHERE ALL THE CREATED FILES GO CALLED 'ExportedFiles'
image_dir = 'ExportedFiles'

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
    df.rename(columns=Demo_dict, inplace=True)

    # ATTRIBUTE UNIQUE NUMBER TO INTERVENTION
    df_inter = pd.read_excel('Section Intervention Assignments.xlsx', header = None, names=['Unique', 'Day', 'Time', 'Room', 'Intervention Number', 'Intervention'])
    df_inter = df_inter.drop(df_inter.columns[[1,2,3]], axis=1)
    df_inter['Unique']=df_inter['Unique'].astype(str)
    df = df.merge(df_inter, how='inner', on = 'Unique')

    # dfW = df[df['Please select the population group(s) that you most closely identify with (select all that apply).'].str.contains('White', na=False)].copy()
    df_norm = Prepare_data(df) # This removes demographic questions, calculates averages and statistics, and combines inversely worded questions into one
    # Data_statistics(df_norm) # Tabulates counts and calcualtes statistics on responses to each question 
    # Gender_differences(df_norm)
    # Intervention_differences(df_norm)
    # SAGE_validation(df_norm) # Confirmatory factor analysis on questions taken from SAGE
    # EFA(df_norm) # Exploratory factor analysis on questions taken from SAGE
    # PCA(df_norm) # principal component analysis on questions taken from SAGE
    # Specifics(df_norm,'Demo','Column') # Compares the column responses based on the demographic
    Mindset(df_norm) 

# ALLOWS THE USER TO TAB-AUTOCOMPLETE IN COMMANDLINE
def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

def save_fig(fig, figure_name):
    fname = os.path.expanduser(f'{image_dir}/{figure_name}')
    plt.savefig(fname + '.png')
    # plt.savefig(fname + '.svg')

def Prepare_data(df):
    Qs = ['When I work in a group, I do higher quality work.', 'When I work in a group, I end up doing most of the work.', 'The work takes more time to complete when I work with other students.',
            'My group members help explain things that I do not understand.', 'When I work in a group, I am able to share my ideas.', 'My group members make me feel that I am not as smart as they are.',
            'The material is easier to understand when I work with other students.', 'The workload is usually less when I work with other students.', 'My group members respect my opinions.',
            'I feel I am part of what is going on in the group.', 'I prefer when one student regularly takes on a leadership role.', 'I prefer when the leadership role rotates between students.',
            'I do not think a group grade is fair.', 'I try to make sure my group members learn the material.', 'I learn to work with students who are different from me.',
            'My group members do not care about my feelings.', 'I let the other students do most of the work.', 'I feel working in groups is a waste of time.',
            'I have to work with students who are not as smart as I am.', 'When I work with other students the work is divided equally.', 'We cannot complete the assignment unless everyone contributes.',
            'I prefer to take on tasks that I’m already good at.', 'I prefer to take on tasks that will help me better learn the material.', 'I also learn when I teach the material to my group members.',
            'I become frustrated when my group members do not understand the material.', 'Everyone’s ideas are needed if we are going to be successful.',
            'When I work with other students, we spend too much time talking about other things.', 'My group did higher quality work when my group members worked on tasks together.',
            'My group did higher quality work when group members worked on different tasks at the same time.', 'You have a certain amount of physics intelligence, and you can’t really do much to change it.',
            'Your physics intelligence is something about you that you can change.', 'You can learn new things, but you can’t really change your basic physics intelligence.']

    # CONVERT RESPONSES TO NUMERICAL VALUE (I DON'T TRUST QUALTRICS TO DO THIS WELL)
    # 6-point Likert first, then the rest
    # 1 = Fixed Mindset, 6 = Growth Mindset
    for i in ['You have a certain amount of physics intelligence, and you can’t really do much to change it.', 'You can learn new things, but you can’t really change your basic physics intelligence.']:
        df.loc[df[i] == 'Strongly disagree', i] = 6
        df.loc[df[i] == 'Disagree', i] = 5
        df.loc[df[i] == 'Somewhat disagree', i] = 4
        df.loc[df[i] == 'Somewhat agree', i] = 3
        df.loc[df[i] == 'Agree', i] = 2
        df.loc[df[i] == 'Strongly agree', i] = 1

    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Strongly disagree', i] = 1
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Disagree', i] = 2
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Somewhat disagree', i] = 3
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Somewhat agree', i] = 4
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Agree', i] = 5
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Strongly agree', i] = 6

    for i in Qs:
        df.loc[df[i].astype(str).str.contains('Strongly disagree') == True, i] = 1
        df.loc[df[i].astype(str).str.contains('Somewhat disagree') == True, i] = 2
        df.loc[df[i].astype(str).str.contains('Neither agree nor disagree') == True, i] = 3
        df.loc[df[i].astype(str).str.contains('Somewhat agree') == True, i] = 4
        df.loc[df[i].astype(str).str.contains('Strongly agree') == True, i] = 5

    Neg_List = [
    'My group members do not respect my opinions.',
    'I prefer when no one takes on a leadership role.',
    'I do not let the other students do most of the work.']
    for i in Neg_List:
        df.loc[df[i].astype(str).str.contains('Strongly disagree') == True, i] = 5
        df.loc[df[i].astype(str).str.contains('Somewhat disagree') == True, i] = 4
        df.loc[df[i].astype(str).str.contains('Neither agree nor disagree') == True, i] = 3
        df.loc[df[i].astype(str).str.contains('Somewhat agree') == True, i] = 2
        df.loc[df[i].astype(str).str.contains('Strongly agree') == True, i] = 1

    # COMBINE NEGATIVELY WORDED QUESTIONS WITH POSITIVELY WORDED QUESTIONS
    
    # res = scipy.stats.mannwhitneyu(x,y) #,nan_policy='omit'
    # sign = scipy.stats.mood(x,y)
    # med_test = scipy.stats.median_test(x,y)
    # tukey = scipy.stats.tukey_hsd(x,y)
    # rank_sum = scipy.stats.ranksums(x,y)
    # kruskal = scipy.stats.kruskal(x,y)
    # y_short = y[0:len(x)]
    # obs = np.array([x,y_short])
    # chi2, p, dof, expected = scipy.stats.chi2_contingency(obs)
    # c_alpha,_ = al(pd.DataFrame(obs))
    # print('Mann-Whitney U:', res)
    # print('Pearsons Chi-square:', chi2.round(decimals = 2), p.round(decimals = 2))
    # print('Moods sign test (z-score, p-score):', sign)
    # print('Median test (stat, p-score):', med_test[0].round(decimals = 2), med_test[1].round(decimals = 4))
    # print('Tukey hsd test:', tukey)
    # print('Wilcoxon rank-sum:', rank_sum)
    # print('Kruskal-Wallis h test:', kruskal)
    # print('Cronbachs alpha:', c_alpha.round(decimals = 2))

    x = np.array(df['My group members respect my opinions.'].dropna(), dtype=np.uint8)
    y = np.array(df['My group members do not respect my opinions.'].dropna(), dtype=np.uint8)
    # res = scipy.stats.mannwhitneyu(x,y,nan_policy='omit')
    # print('My group members respect my opinions.:', x.mean().round(decimals = 2), x.std().round(decimals = 2))
    # print('My group members do not respect my opinions.:', y.mean().round(decimals = 2), y.std().round(decimals = 2))
    # print('Mann-Whitney U:', res)

    df['My group members respect my opinions.'] = df[['My group members respect my opinions.','My group members do not respect my opinions.']].sum(axis=1)
    df.drop(['My group members do not respect my opinions.'], axis=1, inplace=True)

    x = np.array(df['I prefer when one student regularly takes on a leadership role.'].dropna(), dtype=np.uint8)
    y = np.array(df['I prefer when no one takes on a leadership role.'].dropna(), dtype=np.uint8)
    # res = scipy.stats.mannwhitneyu(x,y,nan_policy='omit')
    # print('I prefer when one student regularly takes on a leadership role.:', x.mean().round(decimals = 2), x.std().round(decimals = 2))
    # print('I prefer when no one takes on a leadership role.:', y.mean().round(decimals = 2), y.std().round(decimals = 2))
    # print('Mann-Whitney U:', res)

    df['I prefer when one student regularly takes on a leadership role.'] = df[['I prefer when one student regularly takes on a leadership role.',
    'I prefer when no one takes on a leadership role.']].sum(axis=1)
    df.drop(['I prefer when no one takes on a leadership role.'], axis=1, inplace=True)

    x = np.array(df['I let the other students do most of the work.'].dropna(), dtype=np.uint8)
    y = np.array(df['I do not let the other students do most of the work.'].dropna(), dtype=np.uint8)
    # res = scipy.stats.mannwhitneyu(x,y,nan_policy='omit')
    # print('I let the other students do most of the work.:', x.mean().round(decimals = 2), x.std().round(decimals = 2))
    # print('I do not let the other students do most of the work.:', y.mean().round(decimals = 2), y.std().round(decimals = 2))
    # print('Mann-Whitney U:', res)

    df['I let the other students do most of the work.'] = df[['I let the other students do most of the work.',
    'I do not let the other students do most of the work.']].sum(axis=1)
    df.drop(['I do not let the other students do most of the work.'], axis=1, inplace=True)

    # REMOVE PARTIAL RESPONSES
    df.dropna(axis=0, how='any', subset = Qs, inplace = True)

    # RENORMALIZE DATA SUCH THAT 5-POINT AND 6-POINT ARE EQUAL
    # Z SCORE
    # df_norm = ((df - df.mean())/df.std()).astype(float)

    # CONVERT 6-POINT TO 5-POINT SCALE
    Phys_Int_Cols = [col for col in df.columns if 'physics intelligence' in col]
    df_norm = df.copy()
    for i in Phys_Int_Cols:
        df_norm[i] = df_norm[i]*5/6

    # MAKE ALL SD+D = -1, N = 0, AND SA+A = 1
    # df_norm = df.copy()
    # my_ist = list(df_norm)
    # col_ist = [x for x in my_ist if x not in Phys_Int_Cols]
    # for i in col_ist:
    #     df_norm[i] = np.select([(df_norm[i] == 1) | (df_norm[i] == 2), (df_norm[i] == 3), (df_norm[i] == 4) | (df_norm[i] == 5)], [-1,0,1])
    # for i in Phys_Int_Cols:
    #     df_norm[i] = np.select([(df_norm[i] == 1) | (df_norm[i] == 2), (df_norm[i] == 3) | (df_norm[i] == 4), (df_norm[i] == 5) | (df_norm[i] == 6)], [-1,0,1])

    df_norm = df_norm.astype(float, errors='ignore')

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df_norm)
    return df_norm

def Data_statistics(df_norm):
    # SAVE RAW DATA
    df_norm.to_csv('ExportedFiles/SAGE_Raw.csv', encoding = "utf-8", index=False)

    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Race or ethnicity', 'Race or ethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    df = df_norm.drop(columns=Demo_Qs, axis=1)

    # CALCULATE MEAN, STD DEV OF EACH COLUMN
    df_mean = df.mean(numeric_only=False)
    df_stddev = df.std(numeric_only=False)
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
        df_summary.at[i,'SD+D'] = s[(s.unique_values.round(2) == 0.83) | (s.unique_values.round(2) == 1.67)].sum()['counts']
        df_summary.at[i,'N'] = s[(s.unique_values == 2.5) | (s.unique_values.round(2) == 3.33)].sum()['counts']
        df_summary.at[i,'SA+A'] = s[(s.unique_values.round(2) == 4.17) | (s.unique_values == 5)].sum()['counts']

    df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_Stats.csv', encoding = "utf-8", index=True)

    total_count = len(df_norm.index)
    intervention_count = df_norm.groupby(['Intervention'])['Intervention'].describe()['count']
    course_count = df_norm.groupby(['Course'])['Course'].describe()['count']
    gender_count = df_norm.groupby(['Gender'])['Gender'].describe()['count']
    raceethnicity_count = df_norm.groupby(['Race or ethnicity'])['Race or ethnicity'].describe()['count']
    education_count = df_norm.groupby(['Education'])['Education'].describe()['count']

    ci_count = df_norm.groupby(['Course','Intervention'])['Course'].describe()['count']
    cg_count = df_norm.groupby(['Course','Gender'])['Course'].describe()['count']
    cre_count = df_norm.groupby(['Course','Race or ethnicity'])['Course'].describe()['count']
    ce_count = df_norm.groupby(['Course','Education'])['Course'].describe()['count']

    ic_count = df_norm.groupby(['Intervention','Course'])['Intervention'].describe()['count']
    ig_count = df_norm.groupby(['Intervention','Gender'])['Intervention'].describe()['count']
    ire_count = df_norm.groupby(['Intervention','Race or ethnicity'])['Intervention'].describe()['count']
    ie_count = df_norm.groupby(['Intervention','Education'])['Intervention'].describe()['count']
    # intervention_count, course_count, gender_count, raceethnicity_count, education_count,
    lists = [ci_count, cg_count, cre_count, ce_count, ic_count, ig_count, ire_count, ie_count]
    full_counts = pd.concat([pd.Series(x) for x in lists])
    full_counts.reset_index().to_csv('ExportedFiles/Sage_Counts.csv', encoding = "utf-8", index=True)

def Gender_differences(df_norm):
    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender - Text', 'Race or ethnicity', 'Race or ethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    df = df_norm.drop(columns=Demo_Qs, axis=1)

    # Split based on gender
    dfM = df[df['Gender'] == 'Male'].copy()
    dfF = df[df['Gender'] == 'Female'].copy()
    dfO = df[~df['Gender'].isin(['Male', 'Female'])].copy()

    # REMOVE THE DEMOGRAPHICS QUESTIONS
    dfM.drop(columns=['Gender'], axis=1, inplace = True)
    dfF.drop(columns=['Gender'], axis=1, inplace = True)
    dfO.drop(columns=['Gender'], axis=1, inplace = True)

    # CALCULATE MEAN, STD DEV OF EACH COLUMN
    dfM_mean = dfM.mean(numeric_only=False)
    dfM_med = dfM.median(numeric_only=False)
    dfM_stderr = dfM.std(numeric_only=False)/np.sqrt(dfM.count())
    dfF_mean = dfF.mean(numeric_only=False)
    dfF_med = dfF.median(numeric_only=False)
    dfF_stderr = dfF.std(numeric_only=False)/np.sqrt(dfF.count())
    dfO_mean = dfO.mean(numeric_only=False)
    dfO_med = dfO.median(numeric_only=False)
    dfO_stderr = dfO.std(numeric_only=False)/np.sqrt(dfO.count())
    dfM_summary = pd.merge(dfM_mean.to_frame(), dfM_stderr.to_frame(), left_index = True, right_index=True)
    dfF_summary = pd.merge(dfF_mean.to_frame(), dfF_stderr.to_frame(), left_index = True, right_index=True)
    dfO_summary = pd.merge(dfO_mean.to_frame(), dfO_stderr.to_frame(), left_index = True, right_index=True)    
    dfM_summary.rename(columns={'0_x': 'Mean (male)', '0_y': 'Std.Err. (male)',}, inplace = True)
    dfF_summary.rename(columns={'0_x': 'Mean (female)', '0_y': 'Std.Err. (female)'}, inplace = True)
    dfO_summary.rename(columns={'0_x': 'Mean (other)', '0_y': 'Std.Err. (other)'}, inplace = True)
    dfM_summary['Median (male)'] = dfM_med
    dfF_summary['Median (female)'] = dfF_med
    dfO_summary['Median (other)'] = dfO_med    
    df_summary = pd.merge(pd.merge(dfM_summary, dfF_summary, left_index=True, right_index=True), dfO_summary, left_index=True, right_index=True)
    df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_GenSig.csv', encoding = "utf-8", index=True)
    # dfM.to_csv('ExportedFiles/SAGE_M.csv')
    # dfF.to_csv('ExportedFiles/SAGE_F.csv')
    # dfO.to_csv('ExportedFiles/SAGE_O.csv')

    # cmap = cm.get_cmap('viridis')
    # colors = cmap(np.linspace(0,1,3))
    # palette ={'F': colors[0], 'M': colors[1], 'Other':colors[2]}
    # gen_counts = data.groupby(['Gender']).count()
    # g = sns.barplot(x='Gender', y='frequency',data=gen_counts,
    #     palette=palette)
    # x_coords = [p.get_x() + 0.5*p.get_width() for p in g.patches]
    # y_coords = [p.get_height() for p in g.patches]
    # g.set_xticklabels(task_labels)
    # g.set_xlabel('')
    # g.set_ylabel('Fraction')
    # plt.title('Preferred Role')
    # plt.tight_layout()
    # save_fig(g,'Preferred_role')
    # plt.close()

    # stat, p = scipy.stats.ranksums(df_summary['Median (male)'],df_summary['Median (female)'], nan_policy='omit')
    # print(stat, p)
    # for i in df_summary.index:
    #     stat, p = scipy.stats.ranksums(dfM[i],dfF[i])
    #     print(i, p)

def Intervention_differences(df_norm):
    Demo_Qs = ['Intervention Number', 'Gender', 'Course', 'Unique', 'Gender - Text', 'Race or ethnicity', 'Race or ethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    df = df_norm.drop(columns=Demo_Qs, axis=1)

    # Split based on gender
    dfPA = df[df['Intervention'] == 'Partner Agreements'].copy()
    dfC = df[df['Intervention'] == 'Control'].copy()
    dfCC = df[df['Intervention'] == 'Collaborative Comparison'].copy()

    # REMOVE THE DEMOGRAPHICS QUESTIONS
    dfPA.drop(columns=['Intervention'], axis=1, inplace = True)
    dfC.drop(columns=['Intervention'], axis=1, inplace = True)
    dfCC.drop(columns=['Intervention'], axis=1, inplace = True)

    # CALCULATE MEAN, STD DEV OF EACH COLUMN
    dfPA_mean = dfPA.mean(numeric_only=False)
    dfPA_med = dfPA.median(numeric_only=False)
    dfPA_stderr = dfPA.std(numeric_only=False)/np.sqrt(dfPA.count())
    dfC_mean = dfC.mean(numeric_only=False)
    dfC_med = dfC.median(numeric_only=False)
    dfC_stderr = dfC.std(numeric_only=False)/np.sqrt(dfC.count())
    dfCC_mean = dfCC.mean(numeric_only=False)
    dfCC_med = dfCC.median(numeric_only=False)
    dfCC_stderr = dfCC.std(numeric_only=False)/np.sqrt(dfCC.count())
    dfPA_summary = pd.merge(dfPA_mean.to_frame(), dfPA_stderr.to_frame(), left_index = True, right_index=True)
    dfC_summary = pd.merge(dfC_mean.to_frame(), dfC_stderr.to_frame(), left_index = True, right_index=True)
    dfCC_summary = pd.merge(dfCC_mean.to_frame(), dfCC_stderr.to_frame(), left_index = True, right_index=True)    
    dfPA_summary.rename(columns={'0_x': 'Mean (Partner Agreements)', '0_y': 'Std.Err. (Partner Agreements)',}, inplace = True)
    dfC_summary.rename(columns={'0_x': 'Mean (Control)', '0_y': 'Std.Err. (Control)'}, inplace = True)
    dfCC_summary.rename(columns={'0_x': 'Mean (Collaborative Comparison)', '0_y': 'Std.Err. (Collaborative Comparison)'}, inplace = True)
    dfPA_summary['Median (Partner Agreements)'] = dfPA_med
    dfC_summary['Median (Control)'] = dfC_med
    dfCC_summary['Median (Collaborative Comparison)'] = dfCC_med    
    df_summary = pd.merge(pd.merge(dfPA_summary, dfC_summary, left_index=True, right_index=True), dfCC_summary, left_index=True, right_index=True)
    df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_IntSig.csv', encoding = "utf-8", index=True)
    # dfM.to_csv('ExportedFiles/SAGE_M.csv')
    # dfF.to_csv('ExportedFiles/SAGE_F.csv')
    # dfO.to_csv('ExportedFiles/SAGE_O.csv')

def SAGE_validation(df_norm):
    # REMOVE DEMOGRAPHIC QUESTIONS
    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Race or ethnicity', 'Race or ethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    df = df_norm.drop(columns=Demo_Qs, axis=1)

    # FIRST VALIDATE THE SAGE QUESTIONS
    not_SAGE = ['My group did higher quality work when my group members worked on tasks together.', 
    'My group did higher quality work when group members worked on different tasks at the same time.', 
    'You have a certain amount of physics intelligence, and you can’t really do much to change it.', 
    'Your physics intelligence is something about you that you can’t change very much.',
    'You can learn new things, but you can’t really change your basic physics intelligence.',
    'I prefer to take on tasks that I’m already good at.',
    'I prefer when one student regularly takes on a leadership role.',
    'I prefer when the leadership role rotates between students.']
    df_SAGE = df.drop(not_SAGE, axis=1)

    not_cfa = ['When I work in a group, I end up doing most of the work.', 'I do not think a group grade is fair.',
    'I try to make sure my group members learn the material.',  'When I work with other students the work is divided equally.']
    df_SAGE_cfa = df_SAGE.drop(not_cfa, axis=1)

    # CONFIRMATORY FACTOR ANALYSIS
    # FIRST DEFINE WHICH QUESTIONS SHOULD READ INTO EACH FACTOR, TAKEN FROM KOUROS AND ABRAMI 2006
    model_dict = {
    'F1': ['When I work in a group I do higher quality work.', 'The material is easier to understand when I work with other students.',
            'My group members help explain things that I do not understand.', 'I feel working in groups is a waste of time.', #'The work takes more time to complete when I work with other students.',
            'The workload is usually less when I work with other students.'], 
            # [1, 12, 8, 30, 5, 16]
    'F2': ['My group members respect my opinions.', 'My group members make me feel that I am not as smart as they are.', 'My group members do not care about my feelings.',
            'I feel I am part of what is going on in the group.', 'When I work in a group, I am able to share my ideas.'], 
            # [6, 11, 26, 17, 10]
    'F3': ['Everyone’s ideas are needed if we are going to be successful.', 'We cannot complete the assignment unless everyone contributes.', 'I let the other students do most of the work.',
            'I also learn when I teach the material to my group members.', 'I learn to work with students who are different from me.'], 
            # [52, 36, 28, 49, 25]
    'F4': ['I become frustrated when my group members do not understand the material.', 'When I work with other students, we spend too much time talking about other things.',
            'I have to work with students who are not as smart as I am.']
            # [50, 53, 33]
    }

    model_spec = ModelSpecificationParser.parse_model_specification_from_dict(df_SAGE_cfa,model_dict)

    cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    cfa.fit(df_SAGE_cfa)

    df_cfa = pd.DataFrame(cfa.loadings_,index=model_spec.variable_names)
    df_cfa.round(decimals = 4).to_csv('ExportedFiles/SAGE_CFA.csv', encoding = "utf-8", index=True)
    
    trunc_cfa = np.ma.masked_where(abs(df_cfa) < 0.0001, df_cfa)
    fig, ax = plt.subplots()
    plt.imshow(trunc_cfa, cmap="viridis", vmin=-1, vmax=1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.colorbar()
    plt.tight_layout()
    save_fig(fig, 'SAGE_CFA')
    plt.clf()

    trunc_cfa = np.ma.masked_where(abs(df_cfa) < 0.4, df_cfa)
    fig, ax = plt.subplots()
    plt.imshow(trunc_cfa, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(fig, 'SAGE_CFA_0.5')
    plt.clf()

def EFA(df_norm):
    # REMOVE DEMOGRAPHIC QUESTIONS
    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Race or ethnicity', 'Race or ethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    df = df_norm.drop(columns=Demo_Qs, axis=1)

    # not_SAGE = ['My group did higher quality work when my group members worked on tasks together.', 
    # 'My group did higher quality work when group members worked on different tasks at the same time.', 
    # 'You have a certain amount of physics intelligence, and you can’t really do much to change it.', 
    # 'Your physics intelligence is something about you that you can change.',
    # 'You can learn new things, but you can’t really change your basic physics intelligence.',
    # 'I prefer to take on tasks that I’m already good at.',
    # 'I prefer when one student regularly takes on a leadership role.',
    # 'I prefer when the leadership role rotates between students.']
    # df_SAGE = df.drop(not_SAGE, axis=1)
    df_SAGE = df.copy()

    # CORRELATION MATRIX
    print('Correlation Matrix')
    corrM = df_SAGE.corr(method='spearman')
    corrM.round(decimals = 4).to_csv('ExportedFiles/SAGE_CorrM.csv', encoding = "utf-8", index=True)

    labels = list(df_SAGE)
    with open('CorrM_labels.txt', 'w') as f:
        original_stdout = sys.stdout # Save a reference to the original standard output
        sys.stdout = f # Change the standard output to the file we created.
        for number, label in enumerate(labels):
            print(number, label)
        sys.stdout = original_stdout # Reset the standard output to its original value

    fig, ax = plt.subplots()
    plt.imshow(corrM, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.title('Correlation Matrix')
    plt.tight_layout()
    save_fig(fig,'SAGE_CorrM')
    plt.clf()

    truncM = corrM[abs(corrM)>=0.5]
    fig, ax = plt.subplots()
    plt.title('Correlation Matrix')
    plt.imshow(truncM, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(fig,'SAGE_CorrM_0.5')
    plt.clf()
    
    print('Statistical Tests')
    # BARTLETT'S TEST
    chi_square_value, p_value = calculate_bartlett_sphericity(df_SAGE)
    print('Bartletts Chi Square =', chi_square_value, '; p-value: ', p_value)

    # KAISER-MEYER-OLKIN MEASURE OF SAMPLING ADEQUACY
    kmo_all, kmo_model = calculate_kmo(df_SAGE)
    print('KMO Measure of Sampling Adequacy: ', kmo_model)

    # # CRONBACH'S ALPHA TEST OF CONSISTENCY (test)
    # print('Cronbachs alpha test of consistency: ', al(data=df_SAGE))

    # EFA
    print('EFA')
    efa = FactorAnalyzer(rotation=None)
    efa.fit(df_SAGE)
    ev, v = efa.get_eigenvalues()

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
    fa = FactorAnalyzer(n_factors=4, rotation='varimax')
    fa.fit(df_SAGE)
    m = pd.DataFrame(fa.loadings_)
    # m.to_csv('ExportedFiles/SAGE_EFA.csv', encoding = "utf-8", index=True)

    fig, ax = plt.subplots()
    plt.imshow(m, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar() 
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(fig, 'SAGE_EFA')
    plt.clf()

    truncm = m[abs(m)>=0.5]
    fig, ax = plt.subplots()
    plt.imshow(truncm, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(fig, 'SAGE_EFA_0.5')
    plt.clf()

def PCA(df_norm):
    # REMOVE DEMOGRAPHIC QUESTIONS
    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Race or ethnicity', 'Race or ethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    df_SAGE = df_norm.drop(columns=Demo_Qs, axis=1)

    # CORRELATION MATRIX
    print('Correlation Matrix')
    corrM = df_SAGE.corr(method='spearman')
    corrM.round(decimals = 4).to_csv('ExportedFiles/SAGE_CorrM.csv', encoding = "utf-8", index=True)

    labels = list(df_SAGE)
    with open('CorrM_labels.txt', 'w') as f:
        original_stdout = sys.stdout # Save a reference to the original standard output
        sys.stdout = f # Change the standard output to the file we created.
        for number, label in enumerate(labels):
            print(number, label)
        sys.stdout = original_stdout # Reset the standard output to its original value

    fig, ax = plt.subplots()
    plt.imshow(corrM, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.title('Correlation Matrix')
    plt.tight_layout()
    save_fig(fig,'SAGE_CorrM')
    plt.clf()

    truncM = corrM[abs(corrM)>=0.5]
    fig, ax = plt.subplots()
    plt.title('Correlation Matrix')
    plt.imshow(truncM, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(fig,'SAGE_CorrM_0.5')
    plt.clf()
    
    # print('Statistical Tests')
    # # BARTLETT'S TEST
    # chi_square_value, p_value = calculate_bartlett_sphericity(df_SAGE)
    # print('Bartletts Chi Square =', chi_square_value, '; p-value: ', p_value)

    # # KAISER-MEYER-OLKIN MEASURE OF SAMPLING ADEQUACY
    # kmo_all, kmo_model = calculate_kmo(df_SAGE)
    # print('KMO Measure of Sampling Adequacy: ', kmo_model)

    # # CRONBACH'S ALPHA TEST OF CONSISTENCY (test)
    # print('Cronbachs alpha test of consistency: ', al(data=df_SAGE))

    # PCA ANALYSIS
    print('PCA')
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

    # Specifying the variance to be >=0.45
    pca = PCA(n_components=0.55)
    pca.fit(df_SAGE)
    pca.transform(df_SAGE)
    print('Number of components for variance >= 0.55:', pca.n_components_)

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
    plt.xlim(-0.5,22)
    plt.ylim(0,5.5)
    plt.ylabel('Eigenvalue')
    save_fig(fig, 'PCA_Scree')
    plt.clf()

    mm = pca.components_.T
    fig, ax = plt.subplots()
    plt.imshow(mm, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(fig,'SAGE_PCA')

    trunc = np.ma.masked_where(abs(mm) < 0.5, mm)
    fig, ax = plt.subplots()
    plt.imshow(trunc, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(fig, 'SAGE_PCA_0.5')
    plt.clf()

    # # fit pca objects with and without rotation with 5 principal components
    # standard_pca5 = CustomPCA(n_components=4).fit(df_SAGE)
    # varimax_pca5 = CustomPCA(n_components=4, rotation='varimax').fit(df_SAGE)

    # # display factor matrices and number of cross loadings
    # print('Factor matrix:\n', standard_pca5.components_.round(1))
    # print(' Number of cross-loadings:', standard_pca5.count_cross_loadings())
    # print('\nRotated factor matrix:\n', varimax_pca5.components_.round(1))
    # print(' Number of cross_loadings:', varimax_pca5.count_cross_loadings())

def Specifics(df_norm,demo,col):
    Demo_Qs = ['Intervention Number', 'Course', 'Gender - Text', 'Race or ethnicity - Text', 'Native', 'Asian - Text', 'Black - Text', 'Latino - Text', 
        'MiddleEast - Text', 'Pacific - Text', 'White - Text', 'Education - Text']
    df = df_norm.drop(Demo_Qs,axis=1)

    df_norm.groupby(['Course', demo])[col].describe()

    values_list = df[col].tolist()
    N_values = len(value_list)

    demo_list = df[demo].tolist()
    N_demo = len(demo_list)

    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0,1,N_demo))
    palette = {demo_list[i]: colors[i] for i in range(N_demo)}

    counts = df_norm.groupby([demo,col]).count()
    freq_per_group = counts.div(counts.groupby(demo).transform('sum')).reset_index()
    g = sns.barplot(x=col, y='frequency', hue=demo,data=freq_per_group, palette=palette)
    freq_per_group = approach_freq_per_group.assign(err=lambda x: (x['frequency']*(1-x['frequency'])/N_demo)**0.5)
    x_coords = [p.get_x() + 0.5*p.get_width() for p in g.patches]
    y_coords = [p.get_height() for p in g.patches]
    # plt.errorbar(x=x_coords, y=y_coords, yerr=approach_freq_per_group['err'], fmt="none", c= "k", capsize=5)
    g.set_xticklabels(values_list)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title(col + 'vs' + demo)
    plt.tight_layout()
    save_fig(g,col + 'vs' + demo)
    plt.clf()

def Mindset(df):
    # This code looks at whethever a student has fixed or growth mindset, then uses linear regression (probit) to see if demos or intervention affect this
    Phys_Int_Cols = [col for col in df.columns if 'physics intelligence' in col]

    # Method 1: For each question, determine mindset of response. Then combine the three questions
    df.insert(df.columns.get_loc(Phys_Int_Cols[-1]), 'Mindset1', 0)
    # THIS IS THE WORST CODE I MAY HAVE EVER WRITTEN
    def mset(row):
        if row[Phys_Int_Cols[0]].round(2) == 0.83 or row[Phys_Int_Cols[0]].round(2) == 1.67:
            x = -1
        if row[Phys_Int_Cols[0]].round(2) == 2.5 or row[Phys_Int_Cols[0]].round(2) == 3.33:
            x = 0
        if row[Phys_Int_Cols[0]].round(2) == 4.17 or row[Phys_Int_Cols[0]].round(2) == 5:
            x = 1
        if row[Phys_Int_Cols[1]].round(2) == 0.83 or row[Phys_Int_Cols[1]].round(2) == 1.67:
            y = -1
        if row[Phys_Int_Cols[1]].round(2) == 2.5 or row[Phys_Int_Cols[1]].round(2) == 3.33:
            y = 0
        if row[Phys_Int_Cols[1]].round(2) == 4.17 or row[Phys_Int_Cols[1]].round(2) == 5:
            y = 1
        if row[Phys_Int_Cols[2]].round(2) == 0.83 or row[Phys_Int_Cols[2]].round(2) == 1.67:
            z = -1
        if row[Phys_Int_Cols[2]].round(2) == 2.5 or row[Phys_Int_Cols[2]].round(2) == 3.33:
            z = 0
        if row[Phys_Int_Cols[2]].round(2) == 4.17 or row[Phys_Int_Cols[2]].round(2) == 5:
            z = 1
        return x+y+z
    df['Mindset1'] = df[Phys_Int_Cols].apply(mset, axis=1)

    # Method 2: Combine all responses into one metric and then use that to determine mindset
    df.insert(df.columns.get_loc('Mindset1'), 'Mindset2', 0)

    df['Mindset2'] = df.apply(lambda x: df[Phys_Int_Cols].sum(axis=1))

    # Method 3: Use Factor Analysis loadings to determine mindset (assuming all 3 questions load to same factor)
    df.insert(df.columns.get_loc('Mindset2'), 'Mindset3', 0)

    print(df['Mindset1'])
    print(df['Mindset2'])

Demo_dict = {'Which course are you currently enrolled in?':'Course',
        "What is your section's unique number?":'Unique',
        'Which gender(s) do you most identify (select all that apply)? - Selected Choice':'Gender',
        'Which gender(s) do you most identify (select all that apply)? - Other (please specify): - Text':'Gender - Text',
        'What is your race or ethnicity (select all that apply)? - Selected Choice': 'Race or ethnicity',
        'What is your race or ethnicity (select all that apply)? - Some other race or ethnicity - Text':'Race or ethnicity - Text',
        'American Indian or Alaska Native - Provide details below.\n\nPrint, for example, Navajo Nation, Blackfeet Tribe, Mayan, Aztec, Native Village of Barrow Inupiat Traditional Government, Tlingit, etc.':'Native',
        'Asian - Provide details below. - Selected Choice':'Asian',
        'Asian - Provide details below. - Some other Asian race or ethnicity\n\nPrint, for example, Pakistani, Cambodian, Hmong, etc. - Text':'Asian - Text',
        'Black or African American - Provide details below. - Selected Choice': 'Black',
        'Black or African American - Provide details below. - Some other Black or African American race or ethnicity\n\nPrint, for example, Ghanaian, South African, Barbadian, etc. - Text': 'Black - Text',
        'Hispanic, Latino, or Spanish - Provide details below. - Selected Choice':'Latino',
        'Hispanic, Latino, or Spanish - Provide details below. - Some other Hispanic, Latino, or Spanish race or ethnicity\n\nPrint, for example, Guatemalan, Spaniard, Ecuadorian, etc. - Text':'Latino - Text',
        'Middle Eastern or North African - Provide details below. - Selected Choice':'MiddleEast',
        'Middle Eastern or North African - Provide details below. - Some other Middle Eastern or North African race or ethnicity\n\nPrint, for example, Algerian, Iraqi, Kurdish, etc.</spa - Text':'MiddleEast - Text',
        'Native Hawaiian or Other Pacific Islander - Provide details below. - Selected Choice':'Pacific',
        'Native Hawaiian or Other Pacific Islander - Provide details below. - Some other Native Hawaiian or Other Pacific Islander race or ethnicity\n\nPrint, for example, Palauan, Tahitian, Chuukese, etc.</spa - Text':'Pacific - Text',
        'White - Provide details below. - Selected Choice':'White',
        'White - Provide details below. - Some other White race or ethnicity\n\nPrint, for example, Scottish, Norwegian, Dutch, etc.</spa - Text':'White - Text',
        'What is the highest level of education either of your parents have achieved? - Selected Choice':'Education',
        'What is the highest level of education either of your parents have achieved? - Other - Text':'Education - Text'
        }

# WHERE THE CODE IS ACTUALLY RUN
try:
    if __name__ == '__main__':
        main()
except Exception as err:
    traceback.print_exc(file=sys.stdout)
    input("Press Enter to exit...")