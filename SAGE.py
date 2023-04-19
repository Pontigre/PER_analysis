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

    df_norm = Prepare_data(df) # Takes the raw csv file and converts the string responses into numbers and combines inversely worded questions into one
    # Data_statistics(df_norm) # Tabulates counts and calcualtes statistics on responses to each question 
    # SAGE_validation(df_norm) # Prepares files for Confirmatory factor analysis on questions taken from SAGE run in R
    # with warnings.catch_warnings(): # Included because a warning during factor analysis about using a different method of diagnolization is annoying
    #     warnings.simplefilter("ignore")
    #     EFA_alternate(df_norm) # Exploratory factor analysis on questions taken from SAGE ##CFA package doesn't converge, export files to R.
    with warnings.catch_warnings(): # Included because a warning during factor analysis about using a different method of diagnolization is annoying
        warnings.simplefilter("ignore")
        Factor_dependences(df_norm) # Performs linear regression and other comparisons for how the demographics affect the factors

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

def SAGE_validation(df_norm):
    # REMOVE DEMOGRAPHIC QUESTIONS
    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Raceethnicity', 'Raceethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    df = df_norm.drop(columns=Demo_Qs, axis=1)

    # FIRST VALIDATE THE SAGE QUESTIONS
    not_SAGE = ['My group did higher quality work when my group members worked on tasks together.', 
    'My group did higher quality work when group members worked on different tasks at the same time.', 
    'You have a certain amount of physics intelligence, and you can’t really do much to change it.', 
    'Your physics intelligence is something about you that you can change.',
    'You can learn new things, but you can’t really change your basic physics intelligence.',
    'I prefer to take on tasks that I’m already good at.',
    'I prefer to take on tasks that will help me better learn the material.',
    'I prefer when one student regularly takes on a leadership role.',
    'I prefer when the leadership role rotates between students.']
    df_SAGE = df.drop(not_SAGE, axis=1)

    not_cfa = ['When I work in a group, I end up doing most of the work.', 'I do not think a group grade is fair.',
    'I try to make sure my group members learn the material.',  'When I work with other students the work is divided equally.']
    df_SAGE_cfa = df_SAGE.drop(not_cfa, axis=1)

    # EXPORTS FILE FOR R
    df_SAGE_cfa.to_csv('ExportedFiles/CFA_file.csv', encoding = "utf-8", index=False)

    # CONFIRMATORY FACTOR ANALYSIS
    # FIRST DEFINE WHICH QUESTIONS SHOULD READ INTO EACH FACTOR, TAKEN FROM KOUROS AND ABRAMI 2006

    model_dict = {
    'F1': ['When I work in a group I do higher quality work.', 'The material is easier to understand when I work with other students.',
            'My group members help explain things that I do not understand.', 'I feel working in groups is a waste of time.', 'The work takes more time to complete when I work with other students.',
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

def EFA_alternate(df_norm):
    min_kmo = 0.6
    min_communalities = 0.2
    min_loadings = 0.4

    # REMOVE DEMOGRAPHIC QUESTIONS
    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Raceethnicity', 'Raceethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    df_SAGE = df_norm.drop(columns=Demo_Qs, axis=1).astype(float)
    df_SAGE.apply(pd.to_numeric)
    df_SAGE.to_csv('ExportedFiles/CFA_full.csv', encoding = "utf-8", index=True)
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
    save_fig(fig,'SAGE_CorrM_0.4')
    plt.clf()

    # KAISER-MEYER-OLKIN MEASURE OF SAMPLING ADEQUACY
    kmo_all, kmo_model = calculate_kmo(df_SAGE)
    print('KMO Measure of Sampling Adequacy: ', kmo_model)
    print(kmo_all)

    # BARTLETT'S TEST
    chi_square_value, p_value = calculate_bartlett_sphericity(df_SAGE)
    print('Bartletts Chi Square =', chi_square_value, '; p-value: {0:.3E}'.format(p_value))
    print('Degrees of freedom =', len(labels)*(len(labels)-1)/2)

    # Scree Plot
    print('Scree Plot')
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df_SAGE)

    ev, v = fa.get_eigenvalues()
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 10,
           'legend.fontsize': 10,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8,
           'xtick.bottom': True,
           'xtick.top': True,
           'ytick.left': True,
           'ytick.right': True,
           'xtick.direction': 'in',
           'ytick.direction': 'in',
           'font.weight': 'bold',
           'axes.labelweight': 'bold'
           }
    plt.rcParams.update(params)

    sns.set_context("paper")

    fig, ax = plt.subplots()
    plt.plot(ev, '.-', linewidth=2, color='blue')
    plt.hlines(1, 0, 22, linestyle='dashed')
    plt.xlabel('Factor')
    plt.ylabel('Eigenvalue')
    plt.xlim(-0.5,15)
    plt.ylim(0,8)
    plt.xticks(range(0,20))
    save_fig(fig, 'SAGE_Scree')
    plt.clf()

    print('EFA')
    # Follows Section II.A from Eaton et al. 2019
    # 1. Calculate the Kaiser-Meyer-Olkin (KMO) values for every item. If any items have a KMO below the cutoff value, then the item with the lowest value is removed and the step is repeated. KMO values above 0.6 are kept, though above 0.8 are preferred.
    # 2. Check whether the items can be factored using Bartlett's test of sphericity. A low p-score indicates that factor analysis can be performed.
    # 3. Calculate the EFA model using factoring and a specified number of factors.
    # 4. Calculate the commonalities, which are the proportion of the item's variance explained by the factors. If any item is below the cutoff (<0.4), then the item with the lowest value is dropped and then restart at Step 1.
    # 5. Calculate the item loadings. If there are items that fail to load to any factor, then remove the item with the smallest max loading and then restart at Step 1.
    # 6. Create a model by placing each item onto the factor that contains the item's largest loading. If any items load equally onto more than one factor, then add to all factors where this is the case.
    # 7. Fit this model to the original data using CFA and extract a fit statistic (confirmatory fit index, Akaike information criterion, or similar).
    # 8. Change the number of factors and repeat the above steps.
    # 9. Plot the fit statistic vs the number of factors. The model with the local minimum index is the preferred model.

    fit_stats_x = []
    fit_stats_y = []
    for n in range(2,3):
        print('Number of factors:', n)
        # fit_stats_x.append(n)
        # Create a copy of the data so that we don't remove data when dropping columns
        dfn = df_SAGE.copy()
        dropped = []
        efa=0

        # 5. Loadings loop
        loadings_test = True
        while loadings_test:
            # 4. Communalities loop
            communs_test = True
            while communs_test:
                # 1. KMO loop
                kmo_test = True
                while kmo_test:
                    kmo_all, kmo_model = calculate_kmo(dfn)

                    if abs(kmo_all).min() < min_kmo:
                        print('Lowest KMO:', abs(kmo_all).min())
                        dropped.append(dfn.columns[abs(kmo_all).argmin()])
                        dfn.drop(dfn.columns[abs(kmo_all).argmin()], axis=1, inplace=True)
                    else:
                        kmo_test = False
                # print('KMO Measure of Sampling Adequacy: ', kmo_model)
                # print(kmo_all)

                # 2. BARTLETT
                chi_square_value, p_value = calculate_bartlett_sphericity(dfn)
                if p_value > 0.2:
                    print("Bartlett's Test of Sphericity failed with p-value", p_value)
                # else:
                #     print("Bartlett's Chi Square =", chi_square_value, "; p-value: {0:.3E}".format(p_value))

                # 3. EFA
                efa = FactorAnalyzer(n_factors=n, method='minres', rotation='varimax')
                efa.fit(dfn)

                # 4. Communalities
                communs = efa.get_communalities()

                if abs(communs).min() < min_communalities:
                    print('Lowest communality:', abs(communs).min())
                    dropped.append(dfn.columns[abs(communs).argmin()])
                    dfn.drop(dfn.columns[abs(communs).argmin()], axis=1, inplace=True)
                else:
                    communs_test = False
            # print(communs)

            # 5. Item loading
            drop_list = np.array([abs(i).max() for i in efa.loadings_])
            if drop_list.min() < min_loadings:
                print('Lowest item loading:', drop_list.min())
                dropped.append(dfn.columns[drop_list.argmin()])
                dfn.drop(dfn.columns[drop_list.argmin()], axis=1, inplace=True)
            else:
                loadings_test = False

        # 6. Place loadings into model
        df_loadings = pd.DataFrame(efa.loadings_)
        df_loadings_1 = pd.DataFrame(efa.loadings_, index=list(dfn))
        df_loadings1 = df_loadings_1.reindex(index=list(df_SAGE), fill_value=0)

        truncm = df_loadings1[abs(df_loadings1)>=0.001]
        fig, ax = plt.subplots()
        plt.imshow(truncm, cmap="viridis", vmin=-1, vmax=1)
        plt.colorbar() 
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        file_string1 = 'SAGE_EFA_n=' + str(n)
        save_fig(fig, file_string1)
        plt.clf()

        # For each factor create an empty list to populate with items
        lists = [[] for _ in range(n)]
        # Add the Factor name to the list of lists
        numb = 1
        for i in lists:
            i.append('F'+str(numb))
            numb += 1
        # For each item, find the factor that it loaded into (>0.4)
        # Add that item to the correct factor list

        loads = pd.DataFrame(efa.loadings_)
        loads[abs(loads) < min_loadings] = 0
        # loads.iloc[np.where(loads.ne(0).sum(axis=1) > 1)[0]] = 0

        for index, row in loads.iterrows():
            if abs(row).max() > min_loadings:
                lists[abs(row).argmax()].append(dfn.columns[index])
        # Convert the lists into a dictionary
        model_dict = {i[0]:i[1:] for i in lists}

        # Uncomment to export Factors to R
        for i in lists:
            test_str = '+'.join(i[1:]) 
            test_str1 = test_str.replace(' ','.')
            test_str2 = test_str1.replace(',','.')
            test_str3 = test_str2.replace("'",".")
            print(test_str3)

        file_string = image_dir + '/EFA_factors_n=' + str(n) + '.txt'
        with open(file_string, 'w') as f:
            original_stdout = sys.stdout # Save a reference to the original standard output
            sys.stdout = f # Change the standard output to the file we created.
            for keys, values in model_dict.items():
                print(values)
            sys.stdout = original_stdout # Reset the standard output to its original value

        loads_1 = pd.DataFrame(efa.loadings_, index=list(dfn))
        loads1 = loads_1.reindex(index=list(df_SAGE), fill_value=0)
        loads1[abs(loads1) < min_loadings] = 0
        # loads1.iloc[np.where(loads1.ne(0).sum(axis=1) > 1)[0]] = 0

        truncm = loads1[abs(loads1)>=0.001]
        fig, ax = plt.subplots()
        plt.imshow(truncm, cmap="viridis", vmin=-1, vmax=1)
        plt.colorbar()
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        file_string2 = 'SAGE_EFA_0.4_n=' + str(n)
        save_fig(fig, file_string2)
        plt.clf()

        # 7. Fit model using CFA and extract fit statistic
        # Export to R and just do CFA there instead

    # 9. Plot fit statistic vs number of factors
    fit_stats_x = [2,3,4,5,6,7,8,9,10,11,12,13]

    #>0.4 dropped multi loadings
    # fit_stats_cfi = [0.749, 0.866, 0.887, 0.915, 0.903, 0.891, 0.900, 0.892, 0.917, 0.886, 0.903, 0.920]
    # fit_stats_aic = [26984.664, 28609.788, 29635.652, 30486.064, 31500.537, 33498.759, 33146.545, 32774.801, 26922.132, 35217.072, 33630.630, 31440.705]
    # fit_stats_rmsea = [0.101, 0.070, 0.063, 0.053, 0.058, 0.059, 0.058, 0.059, 0.058, 0.059, 0.055, 0.054]

    # # >0.4 minres factoring
    fit_stats_cfi = [0.737, 0.852, 0.868, 0.876, 0.893, 0.887, 0.895, 0.873, 0.912, 0.876, 0.876, 0.907]
    fit_stats_aic = [28363.591, 30992.473, 32137.870, 33494.822, 32211.446, 34176.652, 33811.093, 34864.240, 30362.595, 35926.155, 35926.155, 33983.052]
    fit_stats_rmsea = [0.102, 0.072, 0.066, 0.063, 0.060, 0.059, 0.058, 0.062, 0.057, 0.061, 0.061, 0.057]

    # >0.45
    # fit_stats_cfi = [0.735, 0.870, 0.902, 0.895, 0.891, 0.893, 0.892, 0.882, 0.909, 0.893, 0.879, 0.929]
    # fit_stats_aic = [26238.613, 25641.484, 24265.990, 27286.532, 26800.021, 25915.938, 24075.208, 28043.943, 19382.200, 27742.007, 34306.088, 21131.969]
    # fit_stats_rmsea = [0.108, 0.075, 0.067, 0.067, 0.068, 0.071, 0.075, 0.067, 0.070, 0.066, 0.062, 0.062]

    # >0.5
    # fit_stats_cfi = [0.744, 0.919, 0.923, 0.912, 0.952, 0, 0.920, 0.926, 0.909, 0.909, 0.926, 0.959]
    # fit_stats_aic = [18421.318, 19339.369, 20325.490, 22875.203, 21216.331, 0, 22075.066, 21556.533, 19382.200, 25617.265, 19049.394, 9880.362]
    # fit_stats_rmsea = [0.132, 0.072, 0.068, 0.067, 0.052, 0, 0.070, 0.065, 0.070, 0.063, 0.072, 0.078]

    # >0.4 principal axis factoring
    # fit_stats_cfi = [0.744, 0.853, 0.848, 0.853, 0.869, 0.889, 0.876, 0.900, 0.915, 0.895, 0.889, 0.901]
    # fit_stats_aic = [36785.799, 33048.111, 38921.059, 37827.959, 36387.090, 34932.983, 42099.834, 38007.856, 35414.681, 39775.987, 42159.633, 42614.275]
    # fit_stats_rmsea = [0.084, 0.066, 0.061, 0.059, 0.059, 0.055, 0.053, 0.049, 0.048, 0.052, 0.052, 0.050]

    # # >0.4 principal axis factoring, oblimin rotation
    # fit_stats_cfi = [0.742, 0.851, 0.854, 0.844, 0.859, 0.843, 0.886, 0.887, 0.877, 0.893, 0.887, 0.903]
    # fit_stats_aic = [34963.108, 32820.964, 38354.429, 40450.617, 39938.855, 43865.028, 34667.657, 37298.599, 37465.952, 40850.335, 43815.261, 41599.902]
    # fit_stats_rmsea = [0.087, 0.068, 0.060, 0.060, 0.058, 0.059, 0.057, 0.054, 0.059, 0.052, 0.052, 0.051]

    # # >0.4 minres factoring, oblimin rotation
    # fit_stats_cfi = [0.735, 0.876, 0.907, 0.895, 0.913, 0.981, 0.886, 0.928, 0.957, 0.933, 0.934, 0.989]
    # fit_stats_aic = [26238.613, 26125.849, 24205.310, 28121.677, 28345.578, 12278.196, 34667.657, 22316.908, 17478.230, 14912.115, 25411.775, 8695.152]
    # fit_stats_rmsea = [0.108, 0.071, 0.065, 0.062, 0.059, 0.054, 0.057, 0.070, 0.058, 0.076, 0.070, 0.036]

    # >0.4 minres factoring, promax rotation
    # fit_stats_cfi = [0.735, 0.884, 0.883, 0.912, 0.904, 0.908, 0.894, 0.893, 0.905, 0.928, 0.951, 0.957]
    # fit_stats_aic = [26238.613, 26506.597, 29128.450, 27683.023, 29791.958, 25347.142, 30469.837, 32697.601, 30499.639, 25531.259, 26400.484, 23966.355]
    # fit_stats_rmsea = [0.108, 0.068, 0.064, 0.056, 0.059, 0.067, 0.059, 0.060, 0.061, 0.061, 0.052, 0.057]

    # # >0.4 principal axis factoring, promax rotation
    # fit_stats_cfi = [0.747, 0.846, 0.854, 0.856, 0.855, 0.850, 0.878, 0.877, 0.907, 0.904, 0.895, 0.957]
    # fit_stats_aic = [32235.822, 34019.724, 38354.429, 37811.160, 38971.628, 39655.241, 39770.818, 39468.707, 39308.744, 39498.471, 41458.399, 23966.355]
    # fit_stats_rmsea = [0.090, 0.068, 0.060, 0.059, 0.060, 0.061, 0.054, 0.056, 0.047, 0.050, 0.051, 0.057]

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()

    p1, = ax.plot(fit_stats_x, fit_stats_aic, marker='.', ls='None', color = 'black', label='AIC')
    p2, = twin1.plot(fit_stats_x, fit_stats_cfi, marker='^', ls='None', color = 'r', label='CFI')
    ax.set_xlim(1.8,9.2)
    ax.set_ylim(20000, 35000)
    twin1.set_ylim(0.85, 1)

    ax.tick_params(axis='both', direction='in', top=True)
    twin1.tick_params(axis='y', direction='in')
    plt.xlabel('Number of factors')
    ax.set_ylabel('Akaike information criterion')
    twin1.set_ylabel('Comparative Fit Index')

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())

    ax.tick_params(axis='y', colors=p1.get_color(), direction='in')
    twin1.tick_params(axis='y', colors=p2.get_color(), direction='in')
    ax.tick_params(axis='x', direction='in', top=True)
    ax.legend(handles=[p1,p2])

    plt.tight_layout()
    save_fig(fig, 'fit_stats_0.4')
    plt.clf()

    # Combined scree and AIC plot
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df_SAGE)
    ev, v = fa.get_eigenvalues()

    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 10,
           'legend.fontsize': 10,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8,
           'xtick.bottom': True,
           'xtick.top': True,
           'ytick.left': True,
           'ytick.right': True,
           'xtick.direction': 'in',
           'ytick.direction': 'in',
           'font.weight': 'bold',
           'axes.labelweight': 'bold'
           }
    plt.rcParams.update(params)

    sns.set_context("paper")
    f, (ax1, ax2) = plt.subplots(2, sharex=True)#, sharey=True)
    p1, = ax1.plot(ev, marker='o', linestyle='solid', linewidth=2, color='black', markeredgecolor='blue', markerfacecolor='blue')
    p2, = ax2.plot(fit_stats_x, fit_stats_aic, marker='o', ls='None', color = 'black', label='AIC')
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    ax1.hlines(1, 0, 22, linestyle='dashed')

    plt.xlabel('Number of factors')
    # ax1.yaxis.label.set_color(p1.get_color())
    # ax1.tick_params(axis='y', colors=p1.get_color(), direction='in')
    ax1.set_ylabel(r'Eigenvalues')
    ax2.set_ylabel('AIC')

    plt.xticks(np.arange(0, 22, 1.0))
    plt.xlim(-0.5,13.5)
    ax1.set_ylim(0.1,8)
    ax2.set_ylim(25000, 39000)

    save_fig(fig, 'Scree_AIC')
    plt.clf()

def factor_scores(df_norm,number_of_factors):
    # Simplified version of EFA_alternate to get factor scores for a single number of factors
    min_kmo = 0.6
    min_communalities = 0.2
    min_loadings = 0.4

    # REMOVE DEMOGRAPHIC QUESTIONS
    Demo_Qs = ['Intervention Number', 'Intervention', 'Course', 'Unique', 'Gender', 'Gender - Text', 'Raceethnicity', 'Raceethnicity - Text', 'Native', 'Asian', 'Asian - Text', 'Black', 'Black - Text', 'Latino', 'Latino - Text', 
        'MiddleEast', 'MiddleEast - Text', 'Pacific', 'Pacific - Text', 'White', 'White - Text', 'Education', 'Education - Text']
    dfn = df_norm.drop(columns=Demo_Qs, axis=1).astype(float)
    dfn.apply(pd.to_numeric)

    loadings_test = True
    while loadings_test:
        communs_test = True
        while communs_test:
            kmo_test = True
            while kmo_test:
                kmo_all, kmo_model = calculate_kmo(dfn)
                if abs(kmo_all).min() < min_kmo:
                    dfn.drop(dfn.columns[abs(kmo_all).argmin()], axis=1, inplace=True)
                else:
                    kmo_test = False
            chi_square_value, p_value = calculate_bartlett_sphericity(dfn)
            if p_value > 0.2:
                print("Bartlett's Test of Sphericity failed with p-value", p_value)

            efa = FactorAnalyzer(n_factors=number_of_factors, method='minres', rotation='varimax')
            efa.fit(dfn)
            df_loadings = pd.DataFrame(efa.loadings_)

            communs = efa.get_communalities()
            if abs(communs).min() < min_communalities:
                dfn.drop(dfn.columns[abs(communs).argmin()], axis=1, inplace=True)
            else:
                communs_test = False

        drop_list = np.array([abs(i).max() for i in efa.loadings_])
        if drop_list.min() < min_loadings:
            dfn.drop(dfn.columns[drop_list.argmin()], axis=1, inplace=True)
        else:
            loadings_test = False

    lists = [[] for _ in range(number_of_factors)]
    item_list = []
    numb = 1
    for i in lists:
        i.append('F'+str(numb))
        numb += 1

    loads = pd.DataFrame(efa.loadings_)
    loads[abs(loads) < min_loadings] = 0
    # loads.iloc[np.where(loads.ne(0).sum(axis=1) > 1)[0]] = 0

    loads_1 = pd.DataFrame(efa.loadings_, index=list(dfn))
    loads1 = loads_1.reindex(index=list(df_norm), fill_value=0).drop(Demo_Qs)
    loads1[abs(loads1) < min_loadings] = 0
    # loads1.iloc[np.where(loads1.ne(0).sum(axis=1) > 1)[0]] = 0

    truncm = loads1[abs(loads1)>=0.001]
    fig, ax = plt.subplots()
    plt.imshow(truncm, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(fig, 'SAGE_EFA')
    plt.clf()
    loads1.round(decimals = 4).to_csv('ExportedFiles/SAGE_factors.csv', encoding = "utf-8", index=True)

    for index, row in loads.iterrows():
        if abs(row).max() > min_loadings:
            lists[abs(row).argmax()].append(dfn.columns[index])
            item_list.append(dfn.columns[index])
    model_dict = {i[0]:i[1:] for i in lists}

    # GET FACTOR SCORES
    scores = pd.DataFrame(np.dot(dfn,loads))
    norm_scores = scores/(loads.abs().sum())

    return norm_scores, model_dict

def Factor_dependences(df_norm):
    # >0.4 minres factoring
    fit_stats_cfi = [0.737, 0.852, 0.868, 0.876, 0.893, 0.887, 0.895, 0.873, 0.912, 0.876, 0.876, 0.907]
    fit_stats_aic = [28363.591, 30992.473, 32137.870, 33494.822, 32211.446, 34176.652, 33811.093, 34864.240, 30362.595, 35926.155, 35926.155, 33983.052]
    fit_stats_rmsea = [0.102, 0.072, 0.066, 0.063, 0.060, 0.059, 0.058, 0.062, 0.057, 0.061, 0.061, 0.057]
    fit_stats_x = [2,3,4,5,6,7,8,9,10,11,12,13]
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
    p1, = ax.plot(fit_stats_x, fit_stats_aic, marker='o', ls='None', color = 'black', label='AIC')
    ax.set_ylim(25000, 38000)
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

    # This code looks at the loading of each student onto each factor, then uses linear regression to see if demos or intervention affect these
    df = df_norm.copy()
    fs, model = factor_scores(df,6)
    fs.columns =['Quality_of_process', 'Collective_Learning', 'Individual_Belonging', 'Mindset', 'Impact_on_Individual', 'Frustrations']

    # Create a dataframe that has the factor scores and the demographics of each student
    Demo_Qs = ['Intervention', 'Course', 'Gender', 'Raceethnicity', 'Education']
    df1 = pd.concat([fs,df[Demo_Qs].set_index(fs.index)], axis=1)

    ##Condenses demographics
    # Gender -> Male, Female, Other
    df1.insert(df1.columns.get_loc('Gender'), 'Gender_C', 0)
    df1['Gender_C'] = ['Male' if x == 'Male' else 'Female' if x == 'Female' else 'Prefer not to disclose' if 'Prefer not' in str(x) else 'Other' for x in df['Gender']]
    df1.drop(columns=['Gender'], axis=1, inplace = True)

    # Raceethnicity -> Wellrepresented (white, asian), underrepresented
    df1.insert(df1.columns.get_loc('Raceethnicity'), 'Raceethnicity_C', 0)
    conditions = [(df1['Raceethnicity'] == 'Asian') | (df1['Raceethnicity'] == 'White') | (df1['Raceethnicity'] == 'Asian,White'),
                ((~df1['Raceethnicity'].str.contains('Asian')) & (~df1['Raceethnicity'].str.contains('White')) & ~df1['Raceethnicity'].str.contains('Prefer not')),
                (df1['Raceethnicity'].str.contains('Prefer not'))]
    choices = ['Wellrepresented','Underrepresented','Prefer not to disclose']
    df1['Raceethnicity_C'] = np.select(conditions, choices, default='Mixed')
    df1.drop(columns=['Raceethnicity'], axis=1, inplace = True)

    # Education -> 1st gen, not 1st gen
    df1.insert(df1.columns.get_loc('Education')+1, 'Education_C', 0)
    df1['Education_C'] = ['1stGen' if (x == 'Other') | (x == 'High school') | (x == 'Some college but no degree') | (x == "Associate's or technical degree") else 'Prefer not to answer' if 'Prefer not' in str(x) else 'Not1stGen' for x in df['Education']]
    df1.drop(columns=['Education'], axis=1, inplace = True)

    # Determine mindset of each student
    df1.insert(df1.columns.get_loc('Mindset'), 'Mindset_C', 0)
    df1['Mindset_C'] = ['Growth' if x > 0.4 else 'Fixed' if x < -0.4 else 'Neutral' for x in df1['Mindset']]

    Phys_Int_Cols = [col for col in df.columns if 'physics intelligence' in col]
    df1.insert(df1.columns.get_loc('Mindset'), 'Mindset_C2', 0)
    conditions = [df[df[Phys_Int_Cols]<-0.5].count(axis=1) > 1,  df[df[Phys_Int_Cols]>0.5].count(axis=1) > 1]
    choices = ['Fixed','Growth']
    df1['Mindset_C2'] = np.select(conditions, choices, default='Neutral')
    df1 = df1[df1['Intervention']!='Collaborative Comparison']
    df1.to_csv('ExportedFiles/StudentRatings.csv', encoding = "utf-8", index=False)

    # total_count = len(df1.index)
    # course_count = df1.groupby(['Course'])['Course'].describe()['count']
    # intervention_count = df1.groupby(['Intervention'])['Intervention'].describe()['count']
    # gender_count = df1.groupby(['Gender_C'])['Gender_C'].describe()['count']
    # raceethnicity_count = df1.groupby(['Raceethnicity_C'])['Raceethnicity_C'].describe()['count']
    # education_count = df1.groupby(['Education_C'])['Education_C'].describe()['count']
    # print('Total Count:', total_count)
    # print('Course Count:', course_count)
    # print('Intervention Count:', intervention_count)
    # print('Gender Count:', gender_count)
    # print('Race/Ethnicity Count:', raceethnicity_count)
    # print('Education Count:', education_count)

    Make_BoxandWhisker_Plots(df1,fs) # I got tired of scrolling past it all

    # Linear regression
    for i in list(fs):
        res = smf.ols((str(i) + "~ C(Intervention, Treatment(reference='Control')) + C(Course) + C(Gender_C, Treatment(reference='Female')) + C(Raceethnicity_C, Treatment(reference='Wellrepresented')) + C(Education_C, Treatment(reference='Not1stGen'))"), data=df1).fit()
        with open(('ExportedFiles/LinReg' + str(i) +'.txt'), 'w') as fh:
            fh.write(res.summary2().as_text())

    dfM = df1[df1['Course'] == 'PHY105M']
    for i in list(fs):
        res = smf.ols((str(i) + "~ C(Intervention, Treatment(reference='Control')) + C(Course) + C(Gender_C, Treatment(reference='Female')) + C(Raceethnicity_C, Treatment(reference='Wellrepresented')) + C(Education_C, Treatment(reference='Not1stGen'))"), data=dfM).fit()
        with open(('ExportedFiles/LinReg' + str(i) +'M.txt'), 'w') as fh:
            fh.write(res.summary2().as_text())
        res = smf.ols((str(i) + "~ C(Course) + C(Gender_C, Treatment(reference='Female')) + C(Raceethnicity_C, Treatment(reference='Wellrepresented')) + C(Education_C, Treatment(reference='Not1stGen'))"), data=dfM[dfM['Intervention'] == 'Control']).fit()
        with open(('ExportedFiles/LinReg' + str(i) +'ControlM.txt'), 'w') as fh:
            fh.write(res.summary2().as_text())
        res = smf.ols((str(i) + "~ C(Course) + C(Gender_C, Treatment(reference='Female')) + C(Raceethnicity_C, Treatment(reference='Wellrepresented')) + C(Education_C, Treatment(reference='Not1stGen'))"), data=dfM[dfM['Intervention'] == 'Partner Agreements']).fit()
        with open(('ExportedFiles/LinReg' + str(i) +'PartnerAgreementsM.txt'), 'w') as fh:
            fh.write(res.summary2().as_text())

    dfN = df1[df1['Course'] == 'PHY105N']
    for i in list(fs):
        res = smf.ols((str(i) + "~ C(Intervention, Treatment(reference='Control')) + C(Course) + C(Gender_C, Treatment(reference='Female')) + C(Raceethnicity_C, Treatment(reference='Wellrepresented')) + C(Education_C, Treatment(reference='Not1stGen'))"), data=dfN).fit()
        with open(('ExportedFiles/LinReg' + str(i) +'N.txt'), 'w') as fh:
            fh.write(res.summary2().as_text())
        res = smf.ols((str(i) + "~ C(Course) + C(Gender_C, Treatment(reference='Female')) + C(Raceethnicity_C, Treatment(reference='Wellrepresented')) + C(Education_C, Treatment(reference='Not1stGen'))"), data=dfN[dfN['Intervention'] == 'Control']).fit()
        with open(('ExportedFiles/LinReg' + str(i) +'ControlN.txt'), 'w') as fh:
            fh.write(res.summary2().as_text())
        res = smf.ols((str(i) + "~ C(Course) + C(Gender_C, Treatment(reference='Female')) + C(Raceethnicity_C, Treatment(reference='Wellrepresented')) + C(Education_C, Treatment(reference='Not1stGen'))"), data=dfN[dfN['Intervention'] == 'Partner Agreements']).fit()
        with open(('ExportedFiles/LinReg' + str(i) +'PartnerAgreementsN.txt'), 'w') as fh:
            fh.write(res.summary2().as_text())

def Make_BoxandWhisker_Plots(df1,fs):
    sns.set_context("paper")
    sns.set(style="whitegrid")
    palette1 = sns.color_palette("colorblind")
    palette2 = sns.color_palette("colorblind")
    palette1[0] = "#1f78b4"
    palette1[1] = "#33a02c"
    palette2[0] = "#a6cee3"
    palette2[1] = "#b2df8a"
    palette2[2] = "#fb9a99"
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 10,
           'legend.fontsize': 10,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8,
           'xtick.bottom': True,
           'xtick.direction': 'in',
           'font.weight': 'bold',
           'axes.labelweight': 'bold'
           }
    plt.rcParams.update(params)
    goldenRatioInverse = ((5**.5 - 1) / 2)
    Factorlabels=['QP','CL','IB','M','II','F']

    # Plot (box and whisker) averages for each factor by course
    df_bnw = df1.drop(['Mindset_C','Mindset_C2','Intervention','Gender_C','Raceethnicity_C','Education_C'],axis=1)
    course_count =  df_bnw.groupby(['Course']).count()
    course_list = []
    course_count_list = []
    courselabels = ['Physics I Lab', 'Physics II Lab']
    for i in list(course_count.index):
        course_list.append(i)
        string = i + ' (n = ' + str(course_count.loc[i]['Mindset']) + ')'
        course_count_list.append(string)
    palette = {course_list[0]: palette2[0], course_list[1]: palette2[1]}
    hue_order = [course_list[0],course_list[1]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Course'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Course', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3, fancybox=True, shadow=False)
    for t, l in zip(L.get_texts(), courselabels):
        t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_course')
    plt.clf()

    # Plot (box and whisker) averages for each factor by intervention
    df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Gender_C','Raceethnicity_C','Education_C'],axis=1)
    intervention_count =  df_bnw.groupby(['Intervention']).count()
    intervention_list = []
    intervention_count_list = []
    for i in list(intervention_count.index):
        intervention_list.append(i)
        string = i + ' (n = ' + str(intervention_count.loc[i]['Mindset']) + ')'
        intervention_count_list.append(string)
    palette ={intervention_list[0]: palette2[0], intervention_list[1]: palette2[1]}
    hue_order = [intervention_list[0],intervention_list[1]]
    interventionlabels = ['Control','Partner Agreements']

    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Intervention'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Intervention', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
    for t, l in zip(L.get_texts(), interventionlabels):
        t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_intervention')
    plt.clf()

    # Plot (box and whisker) averages for each factor by gender
    df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Intervention','Raceethnicity_C','Education_C'],axis=1)
    df_bnw.drop(df_bnw.loc[df_bnw['Gender_C']=='Prefer not to disclose'].index, inplace=True)
    gender_count =  df_bnw.groupby(['Gender_C']).count()
    gender_list = []
    gender_count_list = []
    genderlabels = ['Female', 'Male', 'Non-binary/Other']
    for i in list(gender_count.index):
        gender_list.append(i)
        string = i + ' (n = ' + str(gender_count.loc[i]['Mindset']) + ')'
        gender_count_list.append(string)
    palette ={gender_list[0]: palette2[1], gender_list[1]: palette2[0], gender_list[2]: palette2[2]}
    hue_order = [gender_list[0],gender_list[1], gender_list[2]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Gender_C'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Gender_C', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3, fancybox=True, shadow=False)
    for t, l in zip(L.get_texts(), genderlabels):
        t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_gender')
    plt.clf()

    # Plot (box and whisker) averages for each factor by race and ethnicity
    df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Intervention','Gender_C','Education_C'],axis=1)
    df_bnw.drop(df_bnw.loc[df_bnw['Raceethnicity_C']=='Prefer not to disclose'].index, inplace=True)
    raceethnicity_count =  df_bnw.groupby(['Raceethnicity_C']).count()
    raceethnicity_list = []
    raceethnicity_count_list = []
    raceethnicitylabels = ['No', 'Yes']
    for i in list(raceethnicity_count.index):
        raceethnicity_list.append(i)
        string = i + ' (n = ' + str(raceethnicity_count.loc[i]['Mindset']) + ')'
        raceethnicity_count_list.append(string)
    palette ={raceethnicity_list[2]: palette2[0], raceethnicity_list[1]: palette2[1]}
    hue_order = [raceethnicity_list[2],raceethnicity_list[1]]

    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Raceethnicity_C'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Raceethnicity_C', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
    L.set_title('URM status?')
    for t, l in zip(L.get_texts(), raceethnicitylabels):
        t.set_text(l)
    plt.tight_layout()
    save_fig(g,'factor_ratings_raceethnicity')
    plt.clf()

    # Plot (box and whisker) averages for each factor by education
    df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Intervention','Gender_C','Raceethnicity_C'],axis=1)
    df_bnw.drop(df_bnw.loc[df_bnw['Education_C']=='Prefer not to answer'].index, inplace=True)
    education_count =  df_bnw.groupby(['Education_C']).count()
    education_list = []
    education_count_list = []
    educationlabels = ['Yes', 'No']
    for i in list(education_count.index):
        education_list.append(i)
        string = i + ' (n = ' + str(education_count.loc[i]['Mindset']) + ')'
        education_count_list.append(string)
    palette ={education_list[0]: palette2[0], education_list[1]: palette2[1]}
    hue_order = [education_list[0],education_list[1]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Education_C'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Education_C', hue_order=hue_order, palette=palette)
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

    # Plot (box and whisker) averages for each factor by mindset (method 1)
    df_bnw = df1.drop(['Mindset_C2','Education_C','Course','Intervention','Gender_C','Raceethnicity_C'],axis=1)
    mindset_count =  df_bnw.groupby(['Mindset_C']).count()
    mindset_list = []
    mindset_count_list = []
    for i in list(mindset_count.index):
        mindset_list.append(i)
        string = i + ' (n = ' + str(mindset_count.loc[i]['Mindset']) + ')'
        mindset_count_list.append(string)
    palette ={mindset_list[0]: palette2[0], mindset_list[1]: palette2[1]}
    hue_order = [mindset_list[0],mindset_list[1]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Mindset_C'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Mindset_C', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
    for t, l in zip(L.get_texts(), mindset_list):
        t.set_text(l)
    plt.tight_layout()

    save_fig(g,'factor_ratings_mindset')
    plt.clf()

    # Plot (box and whisker) averages for each factor by mindset (method 2)
    df_bnw = df1.drop(['Mindset_C','Education_C','Course','Intervention','Gender_C','Raceethnicity_C'],axis=1)
    mindset_count =  df_bnw.groupby(['Mindset_C2']).count()
    mindset_list = []
    mindset_count_list = []
    for i in list(mindset_count.index):
        mindset_list.append(i)
        string = i + ' (n = ' + str(mindset_count.loc[i]['Mindset']) + ')'
        mindset_count_list.append(string)
    palette ={mindset_list[0]: palette2[0], mindset_list[1]: palette2[1]}
    hue_order = [mindset_list[0],mindset_list[1]]
    ax = plt.figure(figsize=(3.404*1.5, 3.404*1.5*goldenRatioInverse))
    g = sns.boxplot(data=df_bnw.melt(id_vars=['Mindset_C2'], value_vars=fs.columns, var_name='Rating', value_name='Factor'), 
                    x='Rating', y='Factor', hue='Mindset_C2', hue_order=hue_order, palette=palette)
    plt.xlabel('Factor')
    plt.ylabel('Rating')
    g.set_xticklabels(Factorlabels)
    L = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, shadow=False)
    for t, l in zip(L.get_texts(), mindset_list):
        t.set_text(l)
    plt.tight_layout()

    save_fig(g,'factor_ratings_mindset2')
    plt.clf()

    params = {'axes.labelsize': 14,
           'legend.fontsize': 10,
           'xtick.labelsize': 12,
           'ytick.labelsize': 12,
           'xtick.bottom': True,
           'xtick.direction': 'in',
           'font.weight': 'bold',
           'axes.labelweight': 'bold'
           }
    plt.rcParams.update(params)

    # SPLIT THE BOXPLOTS BY INTERVENTION
    # Plot (box and whisker) averages for each factor by gender
    df_bnw = df1.drop(['Mindset_C','Mindset_C2','Course','Raceethnicity_C','Education_C'],axis=1)
    palette ={gender_list[0]: palette2[1], gender_list[1]: palette2[0], gender_list[2]: palette2[2]}
    hue_order = [gender_list[0],gender_list[1], gender_list[2]]
    ax = plt.figure(figsize=(7.16, 3.404*1.5))
    melt_df = df_bnw.melt(id_vars=['Gender_C','Intervention'], value_vars=fs.columns, var_name='Factor', value_name='Rating')
    g = sns.catplot(data=melt_df, x='Factor', y='Rating', hue='Gender_C', row='Intervention', row_order=['Control', 'Partner Agreements'], hue_order=hue_order, palette=palette, kind='box', legend=False)
    g.add_legend(legend_data={key: value for key, value in zip(genderlabels, g._legend_data.values())}, loc='lower center', bbox_to_anchor=(0.5, 1), ncols=3, frameon=True, fancybox=True, shadow=False)
    axes = g.axes.flatten()
    axes[0].set_title("Control",fontweight='bold')
    axes[1].set_title("Partner Agreements",fontweight='bold')
    axes[0].set_xticklabels(Factorlabels)
    g.fig.tight_layout()
    save_fig(g,'factor_ratings_genderbyintervention2')
    plt.clf()

    # NOW SIDE BY SIDE
    ax = plt.figure(figsize=(7.16, 3.404*1.5))
    melt_df = df_bnw.melt(id_vars=['Gender_C','Intervention'], value_vars=fs.columns, var_name='Factor', value_name='Rating')
    g = sns.catplot(data=melt_df, x='Factor', y='Rating', hue='Gender_C', col='Intervention', row_order=['Control', 'Partner Agreements'], hue_order=hue_order, palette=palette, kind='box', legend=False)
    g.add_legend(legend_data={key: value for key, value in zip(genderlabels, g._legend_data.values())}, loc='lower center', bbox_to_anchor=(0.5, 1), ncols=3, frameon=True, fancybox=True, shadow=False)
    plt.subplots_adjust(hspace=-0.4, wspace=0)
    axes = g.axes.flatten()
    axes[0].set_title("Control",fontweight='bold')
    axes[1].set_title("Partner Agreements",fontweight='bold')
    axes[0].set_xticklabels(Factorlabels)
    g.fig.tight_layout()
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