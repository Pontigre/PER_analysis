# USED FOR ERROR TRACKING
import os
import traceback
import sys
import warnings

# NORMAL PACKAGES
import readline, glob
import pandas as pd
import numpy as np
import scipy
import researchpy as rp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patheffects as path_effects
import seaborn as sns

# FOR FACTOR ANALYSIS
import factor_analyzer
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from patsy import dmatrices

# WHEN I RUN THIS I HAVE A FOLDER WHERE ALL THE CREATED FILES GO CALLED 'ExportedFiles'
image_dir = 'ExportedFiles'

def main():
    my_file = 'SAGEPHY105F22_January3,2023_07.15.csv'

    # READ IN DATA FROM SAGE QUALTRICS SURVEY BASED ON THE CSV COLUMN NAMES
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

    ## IF WE WANT TO REDUCE THE RESPONSES (MAINLY FOR SAGE_validation)
    # dfG = df[df['Gender'] == 'Male'].copy()
    # dfR = df[df['Raceethnicity']== 'White'].copy()

    # df_norm = Prepare_data(df) # Takes the raw csv file and converts the string responses into numbers and combines inversely worded questions into one
    # Prepare_data_for_EFA(df)
    # Data_statistics(df_norm) # Tabulates counts and calcualtes statistics on responses to each question 
    # with warnings.catch_warnings(): # Included because a warning during factor analysis about using a different method of diagnolization is annoying
    #     warnings.simplefilter("ignore")
    #     EFA_alternate(df_norm) # Exploratory factor analysis on questions taken from SAGE ##CFA package doesn't converge, export files to R.
    # with warnings.catch_warnings(): # Included because a warning during factor analysis about using a different method of diagnolization is annoying
    #     warnings.simplefilter("ignore")
    #     Factor_dependences(df_norm) # Performs linear regression and other comparisons for how the demographics affect the factors
    Make_BoxandWhisker_Plots()

# ALLOWS THE USER TO TAB-AUTOCOMPLETE IN COMMANDLINE
def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

def save_fig(fig, figure_name):
    fname = os.path.expanduser(f'{image_dir}/{figure_name}')
    plt.savefig(fname + '.png',dpi=600, bbox_inches='tight')
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

    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Strongly disagree', 'Your physics intelligence is something about you that you can change.'] = 1
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Disagree', 'Your physics intelligence is something about you that you can change.'] = 2
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Somewhat disagree', 'Your physics intelligence is something about you that you can change.'] = 3
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Somewhat agree', 'Your physics intelligence is something about you that you can change.'] = 4
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Agree', 'Your physics intelligence is something about you that you can change.'] = 5
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Strongly agree', 'Your physics intelligence is something about you that you can change.'] = 6

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

    # CONVERT 6-POINT TO 5-POINT SCALE
    Phys_Int_Cols = [col for col in df.columns if 'physics intelligence' in col]
    df_norm = df.copy()

    for i in Phys_Int_Cols:
        df_norm[i] = df_norm[i]*0.8+0.2 

    # MAKE SD = -1, N = 0, AND SA = 1 ...
    for i in Qs:
        df_norm[i] = df_norm[i]*0.5-1.5

    df_norm = df_norm.astype(float, errors='ignore')
    df_norm = df_norm.round(3)
    # SAVE RAW DATA
    df_norm.to_csv('ExportedFiles/SAGE_Raw.csv', encoding = "utf-8", index=False)
    return df_norm

def Prepare_data_for_EFA(df):
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
    # 1 = SD, 6 = SA
    for i in ['You have a certain amount of physics intelligence, and you can’t really do much to change it.', 'You can learn new things, but you can’t really change your basic physics intelligence.']:
        df.loc[df[i] == 'Strongly disagree', i] = 1
        df.loc[df[i] == 'Disagree', i] = 2
        df.loc[df[i] == 'Somewhat disagree', i] = 3
        df.loc[df[i] == 'Somewhat agree', i] = 4
        df.loc[df[i] == 'Agree', i] = 5
        df.loc[df[i] == 'Strongly agree', i] = 6

    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Strongly disagree', 'Your physics intelligence is something about you that you can change.'] = 1
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Disagree', 'Your physics intelligence is something about you that you can change.'] = 2
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Somewhat disagree', 'Your physics intelligence is something about you that you can change.'] = 3
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Somewhat agree', 'Your physics intelligence is something about you that you can change.'] = 4
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Agree', 'Your physics intelligence is something about you that you can change.'] = 5
    df.loc[df['Your physics intelligence is something about you that you can change.'] == 'Strongly agree', 'Your physics intelligence is something about you that you can change.'] = 6

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

    # CONVERT 6-POINT TO 5-POINT SCALE
    Phys_Int_Cols = [col for col in df.columns if 'physics intelligence' in col]
    df_norm = df.copy()

    for i in Phys_Int_Cols:
        df_norm[i] = df_norm[i]*0.8+0.2 

    # MAKE SD = -1, N = 0, AND SA = 1 ...
    for i in Qs:
        df_norm[i] = df_norm[i]*0.5-1.5

    df_norm = df_norm.astype(float, errors='ignore')
    df_norm = df_norm.round(3)
    # SAVE RAW DATA
    df_norm.to_csv('ExportedFiles/SAGE_Raw_EFA.csv', encoding = "utf-8", index=False)

def Data_statistics(df_norm):
    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Raceethnicity', 'Raceethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
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
        df_summary.at[i,'SD+D'] = s[(s.unique_values.round(1) == -1) | (s.unique_values.round(1) == -0.5)].sum()['counts']
        df_summary.at[i,'N'] = s[s.unique_values.round(1) == 0].sum()['counts']
        df_summary.at[i,'SA+A'] = s[(s.unique_values.round(1) == 0.5) | (s.unique_values.round(1) == 1)].sum()['counts']

    for i in Phys_Int_Cols:
        s = df[i].value_counts(normalize=True).sort_index().rename_axis('unique_values').reset_index(name='counts')
        df_summary.at[i,'SD+D'] = s[(s.unique_values.round(1) == -1) | (s.unique_values.round(1) == -0.6)].sum()['counts']
        df_summary.at[i,'N'] = s[(s.unique_values.round(1) == -0.2) | (s.unique_values.round(1) == 0.2)].sum()['counts']
        df_summary.at[i,'SA+A'] = s[(s.unique_values.round(1) == 0.6) | (s.unique_values.round(1) == 1)].sum()['counts']

    df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_Stats.csv', encoding = "utf-8", index=True)

    total_count = len(df_norm.index)
    print(total_count)
    intervention_count = df_norm.groupby(['Intervention'])['Intervention'].describe()['count']
    course_count = df_norm.groupby(['Course'])['Course'].describe()['count']
    gender_count = df_norm.groupby(['Gender'])['Gender'].describe()['count']
    raceethnicity_count = df_norm.groupby(['Raceethnicity'])['Raceethnicity'].describe()['count']
    education_count = df_norm.groupby(['Education'])['Education'].describe()['count']

    ci_count = df_norm.groupby(['Course','Intervention'])['Course'].describe()['count']
    cg_count = df_norm.groupby(['Course','Gender'])['Course'].describe()['count']
    cre_count = df_norm.groupby(['Course','Raceethnicity'])['Course'].describe()['count']
    ce_count = df_norm.groupby(['Course','Education'])['Course'].describe()['count']

    ic_count = df_norm.groupby(['Intervention','Course'])['Intervention'].describe()['count']
    ig_count = df_norm.groupby(['Intervention','Gender'])['Intervention'].describe()['count']
    ire_count = df_norm.groupby(['Intervention','Raceethnicity'])['Intervention'].describe()['count']
    ie_count = df_norm.groupby(['Intervention','Education'])['Intervention'].describe()['count']
    simple_counts = [intervention_count, course_count, gender_count, raceethnicity_count, education_count]
    lists = [ci_count, cg_count, cre_count, ce_count, ic_count, ig_count, ire_count, ie_count]
    top_level_counts = pd.concat([pd.Series(x) for x in simple_counts])
    full_counts = pd.concat([pd.Series(x) for x in lists])
    top_level_counts.reset_index().to_csv('ExportedFiles/Sage_Counts1.csv', encoding = "utf-8", index=True)
    full_counts.reset_index().to_csv('ExportedFiles/Sage_Counts2.csv', encoding = "utf-8", index=True)

def Factor_dependences(df_norm):
    # >0.4 minres factoring
    fit_stats_cfi = []
    fit_stats_aic = [16194, 17402, 17333, 17198, 16534, 18594, 16786]
    fit_stats_rmsea = []
    fit_stats_x = [3,4,5,6,7,8,9]
    sns.set_context("paper")
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 20,
           'legend.fontsize': 20,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'xtick.direction': 'in',
           }
    plt.rcParams.update(params)
    fig, ax = plt.subplots()
    p1, = ax.plot(fit_stats_x, fit_stats_aic, marker='o', ls='None', color = 'k', label='AIC')
    ax.set_ylim(15000, 20000)
    ax.tick_params(axis='both', direction='in', top=True)
    plt.xlabel('Number of factors')
    ax.set_ylabel('Akaike information criterion')
    plt.xticks(np.arange(min(fit_stats_x), max(fit_stats_x)+1, 1.0))
    ax.set_xlim(1.5,9.5)
    ax.tick_params(axis='y', colors=p1.get_color(), direction='in', right=True)
    ax.tick_params(axis='x', direction='in', top=True)

    # fig.subplots_adjust(right=0.75)

    # twin1 = ax.twinx()

    # p1, = ax.plot(fit_stats_x, fit_stats_aic, marker='.', ls='None', color = 'black', label='AIC')
    # p2, = twin1.plot(fit_stats_x, fit_stats_cfi, marker='^', ls='None', color = 'r', label='CFI')
    # ax.set_xlim(1.5,9.5)
    # ax.set_ylim(25000, 38000)
    # twin1.set_ylim(0.85, 1)

    # ax.tick_params(axis='both', direction='in', top=True)
    # twin1.tick_params(axis='y', direction='in')
    # plt.xlabel('Number of factors')
    # ax.set_ylabel('Akaike information criterion')
    # twin1.set_ylabel('Comparative Fit Index')

    # ax.yaxis.label.set_color(p1.get_color())
    # twin1.yaxis.label.set_color(p2.get_color())

    # ax.tick_params(axis='y', colors=p1.get_color(), direction='in')
    # twin1.tick_params(axis='y', colors=p2.get_color(), direction='in')
    # ax.tick_params(axis='x', direction='in', top=True)
    # ax.legend(handles=[p1,p2])

    plt.tight_layout()
    save_fig(fig, 'fit_stats')
    plt.clf()

def Make_BoxandWhisker_Plots():
    sns.set_context("paper")
    sns.set(style="whitegrid")
    palette1 = sns.color_palette("colorblind")
    palette2 = sns.color_palette("colorblind")
    palette2[0] = "#fdb863"
    palette2[1] = "#b2abd2"
    palette2[2] = "#a3ffb7"
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 10,
           'legend.fontsize': 10,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8,
           'xtick.bottom': True,
           'xtick.direction': 'in',
           'font.weight': 'normal',
           'axes.labelweight': 'normal'
           }
    plt.rcParams.update(params)
    goldenRatioInverse = ((5**.5 - 1) / 2)
    
    #READ IN DATA FROM R
    my_file = 'ExportedFiles/R_scores.csv'
    df1 = pd.read_csv(my_file, encoding = "utf-8")
    print(df1)

    # Determine mindset of each student
    df1.insert(df1.columns.get_loc('f2'), 'Mindset_C', 0)
    df1['Mindset_C'] = ['Growth' if x > 0.4 else 'Fixed' if x < -0.4 else 'Neutral' for x in df1['f2']]

    # Phys_Int_Cols = [col for col in df1.columns if 'physics intelligence' in col]
    # df1.insert(df1.columns.get_loc('Mindset'), 'Mindset_C2', 0)
    # conditions = [df1[df1[Phys_Int_Cols]<-0.5].count(axis=1) > 1,  df1[df1[Phys_Int_Cols]>0.5].count(axis=1) > 1]
    # choices = ['Fixed','Growth']
    # df1['Mindset_C2'] = np.select(conditions, choices, default='Neutral')

    fs = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
    Factorlabels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']

    # Plot (box and whisker) averages for each factor by course
    df_bnw = df1.drop(['Mindset_C','Intervention','Gender_group','Raceethnicity_group','Education_group'],axis=1)
    course_count =  df_bnw.groupby(['Course']).count()
    course_list = []
    course_count_list = []
    courselabels = ['Physics I Lab', 'Physics II Lab']
    for i in list(course_count.index):
        course_list.append(i)
        string = i + ' (n = ' + str(course_count.loc[i]['f2']) + ')'
        course_count_list.append(string)
    palette = {course_list[0]: palette2[0], course_list[1]: palette2[1]}
    hue_order = [course_list[0],course_list[1]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Course'], value_vars=fs, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Course', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3, fancybox=True, shadow=False)
    L.set_title('Course')
    for t, l in zip(L.get_texts(), courselabels):
        t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_course')
    plt.clf()

    # Plot (box and whisker) averages for each factor by intervention
    df_bnw = df1.drop(['Mindset_C','Course','Gender_group','Raceethnicity_group','Education_group'],axis=1)
    intervention_count =  df_bnw.groupby(['Intervention']).count()
    intervention_list = []
    intervention_count_list = []
    for i in list(intervention_count.index):
        intervention_list.append(i)
        string = i + ' (n = ' + str(intervention_count.loc[i]['f2']) + ')'
        intervention_count_list.append(string)
    palette ={intervention_list[1]: palette2[0], intervention_list[2]: palette2[1]}#,  intervention_list[0]: palette2[2]}
    hue_order = [intervention_list[1],intervention_list[2]]#,intervention_list[0]]
    interventionlabels = ['Control','Partner Agreements']

    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Intervention'], value_vars=fs, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Intervention', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
    L.set_title('Intervention')
    # for t, l in zip(L.get_texts(), interventionlabels):
    #     t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_intervention')
    plt.clf()

    # Plot (box and whisker) averages for each factor by gender
    df_bnw = df1.drop(['Mindset_C','Course','Intervention','Raceethnicity_group','Education_group'],axis=1)
    df_bnw.drop(df_bnw.loc[df_bnw['Gender_group']=='Prefer not to disclose'].index, inplace=True)
    gender_count =  df_bnw.groupby(['Gender_group']).count()
    gender_list = []
    gender_count_list = []
    genderlabels = ['Woman', 'Man', 'Non-binary/Other']
    for i in list(gender_count.index):
        gender_list.append(i)
        string = i + ' (n = ' + str(gender_count.loc[i]['f2']) + ')'
        gender_count_list.append(string)
    palette ={gender_list[0]: palette2[1], gender_list[1]: palette2[0], gender_list[2]: palette2[2]}
    hue_order = [gender_list[2],gender_list[0], gender_list[1]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Gender_group'], value_vars=fs, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Gender_group', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3, fancybox=True, shadow=False)
    L.set_title('Gender')
    for t, l in zip(L.get_texts(), genderlabels):
        t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_gender')
    plt.clf()

    # Plot (box and whisker) averages for each factor by race and ethnicity
    df_bnw = df1.drop(['Mindset_C','Course','Intervention','Gender_group','Education_group'],axis=1)
    raceethnicity_count =  df_bnw.groupby(['Raceethnicity_group']).count()
    raceethnicity_list = []
    raceethnicity_count_list = []
    raceethnicitylabels = ['Yes', 'No']
    for i in list(raceethnicity_count.index):
        raceethnicity_list.append(i)
        string = i + ' (n = ' + str(raceethnicity_count.loc[i]['f2']) + ')'
        raceethnicity_count_list.append(string)
    palette ={raceethnicity_list[2]: palette2[0], raceethnicity_list[1]: palette2[1]}
    hue_order = [raceethnicity_list[2],raceethnicity_list[1]]

    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Raceethnicity_group'], value_vars=fs, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Raceethnicity_group', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
    L.set_title('Systematically Non Dominant?')
    for t, l in zip(L.get_texts(), raceethnicitylabels):
        t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_raceethnicity')
    plt.clf()

    # Plot (box and whisker) averages for each factor by education
    df_bnw = df1.drop(['Mindset_C','Course','Intervention','Gender_group','Raceethnicity_group'],axis=1)
    education_count =  df_bnw.groupby(['Education_group']).count()
    education_list = []
    education_count_list = []
    educationlabels = ['Yes', 'No']
    for i in list(education_count.index):
        education_list.append(i)
        string = i + ' (n = ' + str(education_count.loc[i]['f2']) + ')'
        education_count_list.append(string)
    palette ={education_list[0]: palette2[0], education_list[1]: palette2[1]}
    hue_order = [education_list[0],education_list[1]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Education_group'], value_vars=fs, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Education_group', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols = 2, fancybox=True, shadow=False)
    L.set_title('1st generation student?')
    for t, l in zip(L.get_texts(), educationlabels):
        t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_education')
    plt.clf()

    # Plot (box and whisker) averages for each factor by mindset
    df_bnw = df1.drop(['Education_group','Course','Intervention','Gender_group','Raceethnicity_group'],axis=1)
    mindset_count = df_bnw.groupby(['Mindset_C']).count()
    mindset_list = []
    mindset_count_list = []
    for i in list(mindset_count.index):
        mindset_list.append(i)
        string = i + ' (n = ' + str(mindset_count.loc[i]['f2']) + ')'
        mindset_count_list.append(string)
    palette ={mindset_list[0]: palette2[0], mindset_list[1]: palette2[1]}
    hue_order = [mindset_list[0],mindset_list[1]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Mindset_C'], value_vars=fs, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Mindset_C', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
    L.set_title('Mindset')
    for t, l in zip(L.get_texts(), mindset_list):
        t.set_text(l)
    plt.tight_layout()

    save_fig(g,'factor_ratings_mindset')
    plt.clf()

    params = {'axes.labelsize': 14,
           'legend.fontsize': 10,
           'xtick.labelsize': 12,
           'ytick.labelsize': 12,
           'xtick.bottom': True,
           'xtick.direction': 'in',
           'font.weight': 'normal',
           'axes.labelweight': 'normal'
           }
    plt.rcParams.update(params)

    # SPLIT THE BOXPLOTS BY INTERVENTION
    # Plot (box and whisker) averages for each factor by gender
    df_bnw = df1.drop(['Course','Raceethnicity_group','Education_group'],axis=1)
    gender_count =  df_bnw.groupby(['Gender_group']).count()
    gender_list = []
    gender_count_list = []
    genderlabels = ['Woman', 'Man', 'Non-binary/Other']
    for i in list(gender_count.index):
        gender_list.append(i)
        string = i + ' (n = ' + str(gender_count.loc[i]['f2']) + ')'
        gender_count_list.append(string)
    palette ={gender_list[0]: palette2[1], gender_list[1]: palette2[0], gender_list[2]: palette2[2]}
    hue_order = [gender_list[2],gender_list[0], gender_list[1]]
    ax = plt.figure(figsize=(7.16, 3.404*1.5))
    melt_df = df_bnw.melt(id_vars=['Gender_group','Intervention'], value_vars=fs, var_name='Factor', value_name='Rating')
    g = sns.catplot(data=melt_df, x='Factor', y='Rating', hue='Gender_group', row='Intervention', row_order=['Control', 'Partner Agreements'], hue_order=hue_order, palette=palette, kind='box', legend=False)
    g.add_legend(legend_data={key: value for key, value in zip(genderlabels, g._legend_data.values())}, loc='lower center', bbox_to_anchor=(0.5, 1), ncols=3, frameon=True, fancybox=True, shadow=False)
    axes = g.axes.flatten()
    axes[0].set_title("Control",fontweight='normal')
    axes[1].set_title("Partner Agreements",fontweight='normal')
    axes[0].set_xticklabels(Factorlabels)
    g.fig.tight_layout()
    save_fig(g,'factor_ratings_genderbyintervention2')
    plt.clf()

    # NOW SIDE BY SIDE
    ax = plt.figure(figsize=(7.057*1.5, 7.057*1.5*((5**.5 - 1) / 2)))
    melt_df = df_bnw.melt(id_vars=['Gender_group','Intervention'], value_vars=fs, var_name='Factor', value_name='Rating')
    g = sns.catplot(data=melt_df, x='Factor', y='Rating', hue='Gender_group', col='Intervention', col_order=['Control', 'Partner Agreements'], hue_order=hue_order, palette=palette, saturation=1, kind='box', legend=False)
    g.add_legend(legend_data={key: value for key, value in zip(genderlabels, g._legend_data.values())}, loc='upper center', bbox_to_anchor=(0.4, 1.05), ncols=3, frameon=True, fancybox=True, shadow=False)
    plt.subplots_adjust(hspace=0, wspace=0.02)
    axes = g.axes.flatten()

    # hatches = ['', '', '...']
    # for ax in g.axes.flat:
    #     # select the correct patches
    #     patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
    #     # the number of patches should be evenly divisible by the number of hatches
    #     h = hatches * (len(patches) // len(hatches))
    #     # iterate through the patches for each subplot
    #     for patch, hatch in zip(patches, h):
    #         patch.set_hatch(hatch)
    #         fc = patch.get_facecolor()
    #         # patch.set_edgecolor(fc)
    # for lp, hatch in zip(g.legend.get_patches(), hatches):
    #     lp.set_hatch(hatch)
    #     fc = lp.get_facecolor()
    #     # lp.set_edgecolor(fc)
    #     #lp.set_facecolor('none')
    axes[0].set_title("Control",fontweight='normal')
    axes[1].set_title("Partner Agreements",fontweight='normal')
    axes[0].set_xticklabels(Factorlabels)
    sns.despine(right=False, top=False)
    save_fig(g,'factor_ratings_genderbyintervention3')
    plt.clf()

def OldFunctionRepository():
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

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df_norm)
    def EFA(df_norm):
        # REMOVE DEMOGRAPHIC QUESTIONS
        Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Race', 'Raceethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
            'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
        df_SAGE = df_norm.drop(columns=Demo_Qs, axis=1).astype(float)
        df_SAGE.apply(pd.to_numeric)

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

        truncM = corrM[abs(corrM)>=min_loadings]
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

        # KAISER-MEYER-OLKIN MEASURE OF SAMPLING ADEQUACY
        kmo_all, kmo_model = calculate_kmo(df_SAGE)
        print('KMO Measure of Sampling Adequacy: ', kmo_model)
        print(kmo_all)

        # BARTLETT'S TEST
        chi_square_value, p_value = calculate_bartlett_sphericity(df_SAGE)
        print('Bartletts Chi Square =', chi_square_value, '; p-value: {0:.2E}'.format(p_value))

        # EFA
        print('EFA')
        efa = FactorAnalyzer(rotation=None)
        efa.fit(df_SAGE)
        ev, v = efa.get_eigenvalues()
        print(pd.DataFrame(efa.get_communalities(),index=df_SAGE.columns,columns=['Communalities']))

        fig, ax = plt.subplots()
        plt.plot(ev, '.-', linewidth=2, color='blue')
        plt.hlines(1, 0, 22, linestyle='dashed')
        plt.title('Factor Analysis Scree Plot')
        plt.xlabel('Factor')
        plt.ylabel('Eigenvalue')
        plt.xlim(-0.5,22)
        plt.ylim(0,5.5)
        plt.xticks(range(0,20))
        save_fig(fig, 'SAGE_Scree')
        plt.clf()

        for i in range(2,8):
            # Based on the scree plot and Kaiser criterion, n=6 (or 7)
            fa = FactorAnalyzer(n_factors=i, rotation='varimax')
            fa.fit(df_SAGE)
            m = pd.DataFrame(fa.loadings_)
            # m.to_csv('ExportedFiles/SAGE_EFA.csv', encoding = "utf-8", index=True)

            fig, ax = plt.subplots()
            plt.imshow(m, cmap="viridis", vmin=-1, vmax=1)
            plt.colorbar() 
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            plt.tight_layout()
            file_string = 'SAGE_EFA_n=' + str(i)
            save_fig(fig, file_string)
            plt.clf()

            truncm = m[abs(m)>=min_loadings]
            fig, ax = plt.subplots()
            plt.imshow(truncm, cmap="viridis", vmin=-1, vmax=1)
            plt.colorbar()
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            plt.tight_layout()
            file_string2 = 'SAGE_EFA_0.5_n=' + str(i)
            save_fig(fig, file_string2)
            plt.clf()

    def PCA(df_norm):
        # REMOVE DEMOGRAPHIC QUESTIONS
        Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Raceethnicity', 'Raceethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
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

        # Specifying the variance to be >=0.55
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

    def Gender_differences(df_norm):
        Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender - Text', 'Raceethnicity', 'Raceethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
            'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
        df = df_norm.drop(columns=Demo_Qs, axis=1)

        # CALCULATE STATISTICS OF EACH
        df_summary = rp.summary_cont(df.groupby('Intervention'))
        df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_GenSig.csv', encoding = "utf-8", index=True)

        # SEPARATE RESPONSES BASED ON GENDER
        df_M = df[df['Gender'] == 'Male'].drop(columns=['Gender'], axis=1)
        df_F = df[df['Gender'] == 'Female'].drop(columns=['Gender'], axis=1)
        df_O = df[~df['Gender'].isin(['Male', 'Female'])].drop(columns=['Gender'], axis=1)

        # ANOVA TEST ON THE RESPONSES
        F, p = scipy.stats.f_oneway(df_M, df_F, df_O)
        print(F, p)

        # GET FACTOR SCORES FOR EACH RESPONSE 
        fs, model = factor_scores(df_norm,6)

        # SEPARATE FACTOR SCORES BASED ON GENDER
        fs['Gender'] = df['Gender']
        fs_M = fs[fs['Intervention'] == 'Male'].drop(columns=['Gender'], axis=1)
        fs_F = fs[fs['Intervention'] == 'Female'].drop(columns=['Gender'], axis=1)
        fs_O = fs[~fs['Gender'].isin(['Male', 'Female'])].drop(columns=['Gender'], axis=1)

        # ANOVA TEST ON THE FACTOR SCORES
        F, p = scipy.stats.f_oneway(fs_M, fs_F, fs_O)
        print(F, p)

    def Intervention_differences(df_norm):
        Demo_Qs = ['Intervention Number', 'Gender', 'Course', 'Unique', 'Gender - Text', 'Raceethnicity', 'Raceethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
            'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
        df = df_norm.drop(columns=Demo_Qs, axis=1)

        # CALCULATE STATS OF EACH INTERVENTION
        df_summary = rp.summary_cont(df.groupby('Intervention'))
        df_summary.round(decimals = 4).to_csv('ExportedFiles/SAGE_IntSig.csv', encoding = "utf-8", index=True)

        # SEPARATE RESPONSES BASED ON INTERVENTION
        df_C = df[df['Intervention'] == 'Control'].drop(columns=['Intervention'], axis=1)
        df_CC = df[df['Intervention'] == 'Collaborative Comparison'].drop(columns=['Intervention'], axis=1)
        df_PA = df[df['Intervention'] == 'Partner Agreements'].drop(columns=['Intervention'], axis=1)

        # ANOVA TEST ON THE RESPONSES
        F, p = scipy.stats.f_oneway(df_C, df_CC, df_PA)
        print(F, p)

        # GET FACTOR SCORES FOR EACH RESPONSE 
        fs, model = factor_scores(df_norm,6)

        # SEPARATE FACTOR SCORES BASED ON INTERVENTION
        fs['Intervention'] = df['Intervention']
        fs_C = fs[fs['Intervention'] == 'Control'].drop(columns=['Intervention'], axis=1)
        fs_CC = fs[fs['Intervention'] == 'Collaborative Comparison'].drop(columns=['Intervention'], axis=1)
        fs_PA = fs[fs['Intervention'] == 'Partner Agreements'].drop(columns=['Intervention'], axis=1)

        # ANOVA TEST ON THE FACTOR SCORES
        F, p = scipy.stats.f_oneway(fs_C, fs_CC, fs_PA)
        print(F, p)

    def Specifics(df_norm,demo,col):
        Demo_Qs = ['Intervention Number', 'Course', 'Gender - Text', 'Raceethnicity - Text', 'Native', 'Asian - Text', 'Black - Text', 'Latino - Text', 
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

    def Mindset(df_norm):
        Phys_Int_Cols = [col for col in df_norm.columns if 'physics intelligence' in col]
        df = df_norm.copy()
        fs, model = factor_scores(df,8)

        # -1 = Fixed Mindset, 1 = Growth Mindset

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
            return (x+y+z)/3
        df['Mindset1'] = df[Phys_Int_Cols].apply(mset, axis=1)

        # Method 2: Combine all responses into one metric and then scale to [-1,1]
        df.insert(df.columns.get_loc('Mindset1')+1, 'Mindset2', 0)
        x_min = 0.83*3
        x_max = 15
        scale_min = -1
        scale_max = 1
        df['Mindset2'] = (df[Phys_Int_Cols].sum(axis=1) - x_min)/(x_max - x_min) * (scale_max - scale_min) + scale_min 
        
        # Method 3: Use Factor Analysis loadings to determine mindset
        df.insert(df.columns.get_loc('Mindset2')+1, 'Mindset3',0)
        df['Mindset3'] = fs[3].values
        # df['Mindset3'] = (fs[3].values - fs[3].min())/(fs[3].max() - fs[3].min()) * (scale_max - scale_min) + scale_min

        mindset = [col for col in df.columns if 'Mindset' in col]
        df[Phys_Int_Cols+mindset].round(decimals = 4).to_csv('ExportedFiles/SAGE_Mindset.csv', encoding = "utf-8", index=True)
        
        Demo_Qs = ['Intervention', 'Course', 'Gender', 'Raceethnicity', 'Education']
        df = df[mindset+Demo_Qs]

        # Linear regression
        mod = smf.ols(formula='Mindset1 ~ C(Intervention) + Course + Gender + C(Raceethnicity) + Education', data=df)
        res = mod.fit()
        print(res.summary().as_latex())

    def add_median_labels(ax, fmt='.2f'):
        lines = ax.get_lines()
        boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
        lines_per_box = int(len(lines) / len(boxes))
        for median in lines[4:len(lines):lines_per_box]:
            x, y = (data.mean() for data in median.get_data())
            text = ax.text(x, y, f'{y:{fmt}}', ha='center', va='center', fontsize=8,
                           fontweight='bold', color='white')
            # create median-colored border around white text for contrast
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ])
    def Make_BoxandWhisker_Plots_old(df1,fs):
        sns.set_context("paper")
        sns.set(style="whitegrid")
        palette1 = sns.color_palette("colorblind")
        palette2 = sns.color_palette("colorblind")
        palette2[0] = "#fdb863"
        palette2[1] = "#b2abd2"
        palette2[2] = "#a3ffb7"
        plt.rcParams['font.family'] = 'serif'
        params = {'axes.labelsize': 10,
               'legend.fontsize': 10,
               'xtick.labelsize': 8,
               'ytick.labelsize': 8,
               'xtick.bottom': True,
               'xtick.direction': 'in',
               'font.weight': 'normal',
               'axes.labelweight': 'normal'
               }
        plt.rcParams.update(params)
        goldenRatioInverse = ((5**.5 - 1) / 2)
        Factorlabels=['QP','CL','IB','M','II','F']

        # # Plot (box and whisker) averages for each factor by course
        # df_bnw = df1.drop(['Mindset_C','Mindset_C2','Intervention','Gender_C','Raceethnicity_C','Education_C'],axis=1)
        # course_count =  df_bnw.groupby(['Course']).count()
        # course_list = []
        # course_count_list = []
        # courselabels = ['Physics I Lab', 'Physics II Lab']
        # for i in list(course_count.index):
        #     course_list.append(i)
        #     string = i + ' (n = ' + str(course_count.loc[i]['Mindset']) + ')'
        #     course_count_list.append(string)
        # palette = {course_list[0]: palette2[0], course_list[1]: palette2[1]}
        # hue_order = [course_list[0],course_list[1]]
        # ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
        # g = sns.boxplot(data=df_bnw.melt(id_vars=['Course'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
        #                 x='Rating', y='Factor', hue='Course', hue_order=hue_order, palette=palette)
        # plt.xlabel('Factor')
        # plt.ylabel('Rating')
        # g.set_xticklabels(Factorlabels)
        # L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3, fancybox=True, shadow=False)
        # for t, l in zip(L.get_texts(), courselabels):
        #     t.set_text(l)
        # plt.tight_layout()
        # save_fig(g,'factor_ratings_course')
        # plt.clf()

        # # Plot (box and whisker) averages for each factor by intervention
        # df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Gender_C','Raceethnicity_C','Education_C'],axis=1)
        # intervention_count =  df_bnw.groupby(['Intervention']).count()
        # intervention_list = []
        # intervention_count_list = []
        # for i in list(intervention_count.index):
        #     intervention_list.append(i)
        #     string = i + ' (n = ' + str(intervention_count.loc[i]['Mindset']) + ')'
        #     intervention_count_list.append(string)
        # palette ={intervention_list[0]: palette2[0], intervention_list[1]: palette2[1]}
        # hue_order = [intervention_list[0],intervention_list[1]]
        # interventionlabels = ['Control','Partner Agreements']

        # ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
        # g = sns.boxplot(data=df_bnw.melt(id_vars=['Intervention'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
        #                 x='Rating', y='Factor', hue='Intervention', hue_order=hue_order, palette=palette)
        # plt.xlabel('Factor')
        # plt.ylabel('Rating')
        # g.set_xticklabels(Factorlabels)
        # L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
        # for t, l in zip(L.get_texts(), interventionlabels):
        #     t.set_text(l)
        # plt.tight_layout()
        # save_fig(g,'factor_ratings_intervention')
        # plt.clf()

        # # Plot (box and whisker) averages for each factor by gender
        # df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Intervention','Raceethnicity_C','Education_C'],axis=1)
        # df_bnw.drop(df_bnw.loc[df_bnw['Gender_C']=='Prefer not to disclose'].index, inplace=True)
        # gender_count =  df_bnw.groupby(['Gender_C']).count()
        # gender_list = []
        # gender_count_list = []
        # genderlabels = ['Women', 'Men', 'Non-binary/Other']
        # for i in list(gender_count.index):
        #     gender_list.append(i)
        #     string = i + ' (n = ' + str(gender_count.loc[i]['Mindset']) + ')'
        #     gender_count_list.append(string)
        # palette ={gender_list[0]: palette2[1], gender_list[1]: palette2[0], gender_list[2]: palette2[2]}
        # hue_order = [gender_list[0],gender_list[1], gender_list[2]]
        # ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
        # g = sns.boxplot(data=df_bnw.melt(id_vars=['Gender_C'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
        #                 x='Rating', y='Factor', hue='Gender_C', hue_order=hue_order, palette=palette)
        # plt.xlabel('Factor')
        # plt.ylabel('Rating')
        # g.set_xticklabels(Factorlabels)
        # L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3, fancybox=True, shadow=False)
        # for t, l in zip(L.get_texts(), genderlabels):
        #     t.set_text(l)
        # plt.tight_layout()
        # save_fig(g,'factor_ratings_gender')
        # plt.clf()

        # # Plot (box and whisker) averages for each factor by race and ethnicity
        # df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Intervention','Gender_C','Education_C'],axis=1)
        # df_bnw.drop(df_bnw.loc[df_bnw['Raceethnicity_C']=='Prefer not to disclose'].index, inplace=True)
        # raceethnicity_count =  df_bnw.groupby(['Raceethnicity_C']).count()
        # raceethnicity_list = []
        # raceethnicity_count_list = []
        # raceethnicitylabels = ['No', 'Yes']
        # for i in list(raceethnicity_count.index):
        #     raceethnicity_list.append(i)
        #     string = i + ' (n = ' + str(raceethnicity_count.loc[i]['Mindset']) + ')'
        #     raceethnicity_count_list.append(string)
        # palette ={raceethnicity_list[2]: palette2[0], raceethnicity_list[1]: palette2[1]}
        # hue_order = [raceethnicity_list[2],raceethnicity_list[1]]

        # ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
        # g = sns.boxplot(data=df_bnw.melt(id_vars=['Raceethnicity_C'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
        #                 x='Rating', y='Factor', hue='Raceethnicity_C', hue_order=hue_order, palette=palette)
        # plt.xlabel('Factor')
        # plt.ylabel('Rating')
        # g.set_xticklabels(Factorlabels)
        # L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
        # L.set_title('URM status?')
        # for t, l in zip(L.get_texts(), raceethnicitylabels):
        #     t.set_text(l)
        # plt.tight_layout()
        # save_fig(g,'factor_ratings_raceethnicity')
        # plt.clf()

        # # Plot (box and whisker) averages for each factor by education
        # df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Intervention','Gender_C','Raceethnicity_C'],axis=1)
        # df_bnw.drop(df_bnw.loc[df_bnw['Education_C']=='Prefer not to answer'].index, inplace=True)
        # education_count =  df_bnw.groupby(['Education_C']).count()
        # education_list = []
        # education_count_list = []
        # educationlabels = ['Yes', 'No']
        # for i in list(education_count.index):
        #     education_list.append(i)
        #     string = i + ' (n = ' + str(education_count.loc[i]['Mindset']) + ')'
        #     education_count_list.append(string)
        # palette ={education_list[0]: palette2[0], education_list[1]: palette2[1]}
        # hue_order = [education_list[0],education_list[1]]
        # ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
        # g = sns.boxplot(data=df_bnw.melt(id_vars=['Education_C'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
        #                 x='Rating', y='Factor', hue='Education_C', hue_order=hue_order, palette=palette)
        # plt.xlabel('Factor')
        # plt.ylabel('Rating')
        # g.set_xticklabels(Factorlabels)
        # L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols = 2, fancybox=True, shadow=False)
        # L.set_title('1st generation student?')
        # for t, l in zip(L.get_texts(), educationlabels):
        #     t.set_text(l)
        # plt.tight_layout()
        # save_fig(g,'factor_ratings_education')
        # plt.clf()

        # # Plot (box and whisker) averages for each factor by mindset (method 1)
        # df_bnw = df1.drop(['Mindset_C2','Education_C','Course','Intervention','Gender_C','Raceethnicity_C'],axis=1)
        # mindset_count =  df_bnw.groupby(['Mindset_C']).count()
        # mindset_list = []
        # mindset_count_list = []
        # for i in list(mindset_count.index):
        #     mindset_list.append(i)
        #     string = i + ' (n = ' + str(mindset_count.loc[i]['Mindset']) + ')'
        #     mindset_count_list.append(string)
        # palette ={mindset_list[0]: palette2[0], mindset_list[1]: palette2[1]}
        # hue_order = [mindset_list[0],mindset_list[1]]
        # ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
        # g = sns.boxplot(data=df_bnw.melt(id_vars=['Mindset_C'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
        #                 x='Rating', y='Factor', hue='Mindset_C', hue_order=hue_order, palette=palette)
        # plt.xlabel('Factor')
        # plt.ylabel('Rating')
        # g.set_xticklabels(Factorlabels)
        # L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
        # for t, l in zip(L.get_texts(), mindset_list):
        #     t.set_text(l)
        # plt.tight_layout()

        # save_fig(g,'factor_ratings_mindset')
        # plt.clf()

        # # Plot (box and whisker) averages for each factor by mindset (method 2)
        # df_bnw = df1.drop(['Mindset_C','Education_C','Course','Intervention','Gender_C','Raceethnicity_C'],axis=1)
        # mindset_count =  df_bnw.groupby(['Mindset_C2']).count()
        # mindset_list = []
        # mindset_count_list = []
        # for i in list(mindset_count.index):
        #     mindset_list.append(i)
        #     string = i + ' (n = ' + str(mindset_count.loc[i]['Mindset']) + ')'
        #     mindset_count_list.append(string)
        # palette ={mindset_list[0]: palette2[0], mindset_list[1]: palette2[1]}
        # hue_order = [mindset_list[0],mindset_list[1]]
        # ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
        # g = sns.boxplot(data=df_bnw.melt(id_vars=['Mindset_C2'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
        #                 x='Rating', y='Factor', hue='Mindset_C2', hue_order=hue_order, palette=palette)
        # plt.xlabel('Factor')
        # plt.ylabel('Rating')
        # g.set_xticklabels(Factorlabels)
        # L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
        # for t, l in zip(L.get_texts(), mindset_list):
        #     t.set_text(l)
        # plt.tight_layout()

        # save_fig(g,'factor_ratings_mindset2')
        # plt.clf()

        params = {'axes.labelsize': 14,
               'legend.fontsize': 10,
               'xtick.labelsize': 12,
               'ytick.labelsize': 12,
               'xtick.bottom': True,
               'xtick.direction': 'in',
               'font.weight': 'normal',
               'axes.labelweight': 'normal'
               }
        plt.rcParams.update(params)

        # SPLIT THE BOXPLOTS BY INTERVENTION
        # Plot (box and whisker) averages for each factor by gender
        df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Raceethnicity_C','Education_C'],axis=1)
        gender_count =  df_bnw.groupby(['Gender_C']).count()
        gender_list = []
        gender_count_list = []
        genderlabels = ['Men', 'Women', 'Non-binary/Other']
        for i in list(gender_count.index):
            gender_list.append(i)
            string = i + ' (n = ' + str(gender_count.loc[i]['Mindset']) + ')'
            gender_count_list.append(string)
        palette ={gender_list[0]: palette2[0], gender_list[3]: palette2[1], gender_list[1]: palette2[2]}
        hue_order = [gender_list[0],gender_list[3], gender_list[1]]
        ax = plt.figure(figsize=(7.16, 3.404*1.5))
        melt_df = df_bnw.melt(id_vars=['Gender_C','Intervention'], value_vars=fs.columns, var_name='Factor', value_name='Rating')
        g = sns.catplot(data=melt_df, x='Factor', y='Rating', hue='Gender_C', row='Intervention', row_order=['Control', 'Partner Agreements'], hue_order=hue_order, palette=palette, kind='box', legend=False)
        g.add_legend(legend_data={key: value for key, value in zip(genderlabels, g._legend_data.values())}, loc='lower center', bbox_to_anchor=(0.5, 1), ncols=3, frameon=True, fancybox=True, shadow=False)
        axes = g.axes.flatten()
        axes[0].set_title("Control",fontweight='normal')
        axes[1].set_title("Partner Agreements",fontweight='normal')
        axes[0].set_xticklabels(Factorlabels)
        g.fig.tight_layout()
        save_fig(g,'factor_ratings_genderbyintervention2')
        plt.clf()

        # NOW SIDE BY SIDE
        ax = plt.figure(figsize=(7.057*1.5, 7.057*1.5*((5**.5 - 1) / 2)))
        melt_df = df_bnw.melt(id_vars=['Gender_C','Intervention'], value_vars=fs.columns, var_name='Factor', value_name='Rating')
        g = sns.catplot(data=melt_df, x='Factor', y='Rating', hue='Gender_C', col='Intervention', row_order=['Control', 'Partner Agreements'], hue_order=hue_order, palette=palette, saturation=1, kind='box', legend=False)
        g.add_legend(legend_data={key: value for key, value in zip(genderlabels, g._legend_data.values())}, loc='upper center', bbox_to_anchor=(0.4, 1.05), ncols=3, frameon=True, fancybox=True, shadow=False)
        plt.subplots_adjust(hspace=0, wspace=0.02)
        axes = g.axes.flatten()

        # hatches = ['', '', '...']
        # for ax in g.axes.flat:
        #     # select the correct patches
        #     patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
        #     # the number of patches should be evenly divisible by the number of hatches
        #     h = hatches * (len(patches) // len(hatches))
        #     # iterate through the patches for each subplot
        #     for patch, hatch in zip(patches, h):
        #         patch.set_hatch(hatch)
        #         fc = patch.get_facecolor()
        #         # patch.set_edgecolor(fc)
        # for lp, hatch in zip(g.legend.get_patches(), hatches):
        #     lp.set_hatch(hatch)
        #     fc = lp.get_facecolor()
        #     # lp.set_edgecolor(fc)
        #     #lp.set_facecolor('none')
        axes[0].set_title("Control",fontweight='normal')
        axes[1].set_title("Partner Agreements",fontweight='normal')
        axes[0].set_xticklabels(Factorlabels)
        sns.despine(right=False, top=False)
        save_fig(g,'factor_ratings_genderbyintervention3')
        plt.clf()

Demo_dict = {'Which course are you currently enrolled in?':'Course',
        "What is your section's unique number?":'Unique',
        'Which gender(s) do you most identify (select all that apply)? - Selected Choice':'Gender',
        'Which gender(s) do you most identify (select all that apply)? - Other (please specify): - Text':'Gender - Text',
        'What is your race or ethnicity (select all that apply)? - Selected Choice': 'Raceethnicity',
        'What is your race or ethnicity (select all that apply)? - Some other race or ethnicity - Text':'Raceethnicity - Text',
        'American Indian or Alaska Native - Provide details below.\r\n\r\nPrint, for example, Navajo Nation, Blackfeet Tribe, Mayan, Aztec, Native Village of Barrow Inupiat Traditional Government, Tlingit, etc.':'Native',
        'Asian - Provide details below. - Selected Choice':'Asian',
        'Asian - Provide details below. - Some other Asian race or ethnicity\r\n\r\nPrint, for example, Pakistani, Cambodian, Hmong, etc. - Text':'Asian - Text',
        'Black or African American - Provide details below. - Selected Choice': 'Black',
        'Black or African American - Provide details below. - Some other Black or African American race or ethnicity\r\n\r\nPrint, for example, Ghanaian, South African, Barbadian, etc. - Text': 'Black - Text',
        'Hispanic, Latino, or Spanish - Provide details below. - Selected Choice':'Latino',
        'Hispanic, Latino, or Spanish - Provide details below. - Some other Hispanic, Latino, or Spanish race or ethnicity\r\n\r\nPrint, for example, Guatemalan, Spaniard, Ecuadorian, etc. - Text':'Latino - Text',
        'Middle Eastern or North African - Provide details below. - Selected Choice':'MiddleEast',
        'Middle Eastern or North African - Provide details below. - Some other Middle Eastern or North African race or ethnicity\r\n\r\nPrint, for example, Algerian, Iraqi, Kurdish, etc.</spa - Text':'MiddleEast - Text',
        'Native Hawaiian or Other Pacific Islander - Provide details below. - Selected Choice':'Pacific',
        'Native Hawaiian or Other Pacific Islander - Provide details below. - Some other Native Hawaiian or Other Pacific Islander race or ethnicity\r\n\r\nPrint, for example, Palauan, Tahitian, Chuukese, etc.</spa - Text':'Pacific - Text',
        'White - Provide details below. - Selected Choice':'White',
        'White - Provide details below. - Some other White race or ethnicity\r\n\r\nPrint, for example, Scottish, Norwegian, Dutch, etc.</spa - Text':'White - Text',
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