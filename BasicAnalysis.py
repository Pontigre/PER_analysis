# USED FOR ERROR TRACKING
import os
import traceback
import sys

# NORMAL PACKAGES
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# I LIEK FOLDERS
image_dir = 'Diagrams'

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

def Holmes_plots(data):
    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0,1,2))
    palette ={'F': colors[0], 'M': colors[1]}
    hue_order = ['F','M']
    task_order=['Setting up the apparatus and collecting data.',
        'Writing up the lab methods and conclusions.',
        'Analyzing data and making graphs.',
        'Managing the group progress.',
        'I prefer to be involved in all of the above.',
        'No preference or none of the above.']
    task_labels=['Equipment','Notes','Analysis','Managing','All','No preference']
    approach_order=['One where everyone takes turns with each task.',
        'One where each person has a different task.',
        'One where everyone works on each task together.',
        'No preference.',
        'Something else.']
    approach_labels=['Take turns','Different tasks','Work together','No preference','Other']
    leadership_order=['One where no one takes on the leadership role.',
        'One where the leadership role rotates between students.',
        'One where one student regularly takes on the leadership role.',
        'No preference.',
        'Something else.']
    leadership_labels=['No leader','Take turns','One leader','No preference','Other']
    data['frequency'] = 0 # a dummy column to refer to
    Nn = int(data['Sex'].count())

    # PLOT PREFERENCE OF ROLE VS GENDER (HOLMES FIG 1a)
    fig, axes = plt.subplots()
    task_counts = data.groupby(['Tasks','Sex']).count()
    task_freq_per_group = task_counts.div(task_counts.groupby('Sex').transform('sum')).reset_index()
    task_freq_per_group = task_freq_per_group.assign(err=lambda x: (x['frequency']*(1-x['frequency'])/Nn)**0.5)
    g = sns.barplot(x='Tasks', y='frequency', hue='Sex',data=task_freq_per_group,
        order=task_order,hue_order=hue_order,palette=palette)
    x_coords = [p.get_x() + 0.5*p.get_width() for p in g.patches]
    y_coords = [p.get_height() for p in g.patches]
    # plt.errorbar(x=x_coords, y=y_coords, yerr=task_freq_per_group['err'], fmt="none", c= "k", capsize=5)
    g.set_xticklabels(task_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Preferred Role')
    plt.tight_layout()
    save_fig(g,'Preferred_role')
    plt.close()

    # PLOT PREFERENCE OF ROLE DISTRIBUTION VS GENDER (HOLMES FIG 1b)
    fig, ax = plt.subplots()
    approach_counts = data.groupby(['Approaches','Sex']).count()
    approach_freq_per_group = approach_counts.div(approach_counts.groupby('Sex').transform('sum')).reset_index()
    g = sns.barplot(x='Approaches', y='frequency', hue='Sex',data=approach_freq_per_group, 
        order=approach_order,hue_order=hue_order,palette=palette)
    approach_freq_per_group = approach_freq_per_group.assign(err=lambda x: (x['frequency']*(1-x['frequency'])/Nn)**0.5)
    x_coords = [p.get_x() + 0.5*p.get_width() for p in g.patches]
    y_coords = [p.get_height() for p in g.patches]
    # plt.errorbar(x=x_coords, y=y_coords, yerr=approach_freq_per_group['err'], fmt="none", c= "k", capsize=5)
    g.set_xticklabels(approach_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Role Distribution')
    plt.tight_layout()
    save_fig(g,'Role_distribution')
    plt.close()

    # PLOT LEADERSHIP PREFERENCE VS GENDER (HOLMES FIG 2)
    fig, ax = plt.subplots()
    leadership_counts = data.groupby(['Leadership','Sex']).count()
    leadership_freq_per_group = leadership_counts.div(leadership_counts.groupby('Sex').transform('sum')).reset_index()
    leadership_freq_per_group = leadership_freq_per_group.assign(err=lambda x: (x['frequency']*(1-x['frequency'])/Nn)**0.5)
    g = sns.barplot(x='Leadership', y='frequency', hue='Sex',data=leadership_freq_per_group, 
        order=leadership_order,hue_order=hue_order,palette=palette)
    x_coords = [p.get_x() + 0.5*p.get_width() for p in g.patches]
    y_coords = [p.get_height() for p in g.patches]
    # plt.errorbar(x=x_coords, y=y_coords, yerr=leadership_freq_per_group['err'], fmt="none", c= "k", capsize=5)
    g.set_xticklabels(leadership_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Leadership Preference')
    plt.tight_layout()
    save_fig(g,'Leadership_preference')
    plt.close()

def bySchool(data):
    data = data[data.Sch.str.contains('6|N|P|S')==False ] # DROPS ANY SCHOOL <2 STUDENTS
    data['Sch'] = data['Sch'].replace({'2' : 'Business (N=16)', '3' : 'Education (N=50)', '4' : 'Engineering (N=517)', '5' : 'Fine Arts (N=10)', '6' : 'Graduate School (N=1)', '9' : 'Architecture (N=10)', 
        'C' : 'Communication (N=7)', 'E' : 'Natural Sciences (N=924)', 'L' : 'Liberal Arts (N=173)', 'J' : 'Jackson (N=17)', 'N' : 'Nursing (N=0)',
        'P' : 'Pharmacy (N=1)', 'S' : 'Social Work (N=0)', 'U' : 'Undeclared  (N=59)', 'D' : 'Double Major (N=77)'})
    # Sch_Coding = {'2' : 'Business', '3' : 'Education', '4' : 'Engineering', '5' : 'Fine Arts', '6' : 'Graduate School', '9' : 'Architecture', 
        # 'C' : 'Communication', 'E' : 'Natural Sciences', 'L' : 'Liberal Arts', 'J' : 'Jackson School of Geosciences', 'N' : 'Nursing',
        # 'P' : 'Pharmacy', 'S' : 'Social Work', 'U' : 'Undergraduate Studies', 'D' : 'Double Major'}

    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0,1,11))
    palette ={'Architecture (N=10)' : colors[0], 'Business (N=16)' : colors[1], 'Communication (N=7)' : colors[2], 'Education (N=50)' : colors[3],
        'Engineering (N=517)' : colors[4], 'Fine Arts (N=10)' : colors[5], 'Jackson (N=17)' : colors[6], 'Liberal Arts (N=173)' : colors[7],
        'Natural Sciences (N=924)' : colors[8], 'Undeclared  (N=59)' : colors[9], 'Double Major (N=77)' : colors[10]}
    hue_order=['Architecture (N=10)', 'Business (N=16)', 'Communication (N=7)', 'Education (N=50)', 'Engineering (N=517)', 'Fine Arts (N=10)', 
        'Jackson (N=17)', 'Liberal Arts (N=173)', 'Natural Sciences (N=924)', 'Undeclared  (N=59)', 'Double Major (N=77)']
    task_order=['Setting up the apparatus and collecting data.',
        'Writing up the lab methods and conclusions.',
        'Analyzing data and making graphs.',
        'Managing the group progress.',
        'I prefer to be involved in all of the above.',
        'No preference or none of the above.']
    task_labels=['Equipment','Notes','Analysis','Managing','All','No preference']
    approach_order=['One where everyone takes turns with each task.',
        'One where each person has a different task.',
        'One where everyone works on each task together.',
        'No preference.',
        'Something else.']
    approach_labels=['Take turns','Different tasks','Work together','No preference','Other']
    leadership_order=['One where no one takes on the leadership role.',
        'One where the leadership role rotates between students.',
        'One where one student regularly takes on the leadership role.',
        'No preference.',
        'Something else.']
    leadership_labels=['No leader','Take turns','One leader','No preference','Other']
    data['frequency'] = 0 # a dummy column to refer to
    Nn = int(data['Sch'].count())

    # PLOT PREFERENCE OF ROLE VS SCHOOL
    fig, axes = plt.subplots()
    task_counts = data.groupby(['Tasks','Sch']).count()
    task_freq_per_group = task_counts.div(task_counts.groupby('Sch').transform('sum')).reset_index()
    g = sns.barplot(x='Tasks', y='frequency', hue='Sch',data=task_freq_per_group, 
        order=task_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(task_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Preferred Role by School')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Preferred_role_bySch')
    plt.close()

    # PLOT PREFERENCE OF ROLE DISTRIBUTION VS SCHOOL
    fig, ax = plt.subplots()
    approach_counts = data.groupby(['Approaches','Sch']).count()
    approach_freq_per_group = approach_counts.div(approach_counts.groupby('Sch').transform('sum')).reset_index()
    g = sns.barplot(x='Approaches', y='frequency', hue='Sch',data=approach_freq_per_group, 
        order=approach_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(approach_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Role Distribution by School')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Role_distribution_bySch')
    plt.close()

    # PLOT LEADERSHIP PREFERENCE VS SCHOOL
    fig, ax = plt.subplots()
    leadership_counts = data.groupby(['Leadership','Sch']).count()
    leadership_freq_per_group = leadership_counts.div(leadership_counts.groupby('Sch').transform('sum')).reset_index()
    g = sns.barplot(x='Leadership', y='frequency', hue='Sch',data=leadership_freq_per_group, 
        order=leadership_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(leadership_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Leadership Preference by School')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Leadership_preference_bySch')
    plt.close()

def byMaj(data):
    # GET MAJORS WITH >50 STUDENTS
    # print(data.groupby(['Maj']).count().sort_values('Student',ascending=False).head(10))
    x = data.loc[((data['Maj'] == '10400') | (data['Maj'] == '100') | (data['Maj'] == '11100') | (data['Maj'] == '66100') |
        (data['Maj'] == '10300') | (data['Maj'] == '13600') | (data['Maj'] == '84000') | (data['Maj'] == '21700'))]

    # '10400' : 'BIOLOGY (BSA) [N=152]'
    # '100' : 'UNDECLARED [N=107]'
    # '11100' : 'NEUROSCI (BSA) [N=104]'
    # '66100' : 'MECHANICAL ENGR [N=95]'
    # '10300' : 'BIOCHEM (BSA) [N=93]'
    # '13600' : 'BIOCHEM (BSBIOCHEM) [N=65]'
    # '84000' : 'PSYCH (BSPSY) [N=64]'
    # '21700' : 'CIVIL ENGR [N=56]'

    x['Maj'] = x['Maj'].replace({'10400' : 'BIOLOGY (BSA) [N=152]', '100' : 'UNDECLARED [N=107]', '11100' : 'NEUROSCI (BSA) [N=104]', 
        '66100' : 'MECHANICAL ENGR [N=95]', '10300' : 'BIOCHEM (BSA) [N=93]', '13600' : 'BIOCHEM (BSBIOCHEM) [N=65]',
        '84000' : 'PSYCH (BSPSY) [N=64]', '21700' : 'CIVIL ENGR [N=56]'})

    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0,1,8))
    palette ={'UNDECLARED [N=107]' : colors[0], 'BIOLOGY (BSA) [N=152]' : colors[1], 'BIOCHEM (BSA) [N=93]' : colors[2], 'BIOCHEM (BSBIOCHEM) [N=65]' : colors[3],
     'NEUROSCI (BSA) [N=104]' : colors[4], 'PSYCH (BSPSY) [N=64]' : colors[5], 'MECHANICAL ENGR [N=95]' : colors[6], 'CIVIL ENGR [N=56]' : colors[7]}
    hue_order=['UNDECLARED [N=107]', 'BIOLOGY (BSA) [N=152]', 'BIOCHEM (BSA) [N=93]', 'BIOCHEM (BSBIOCHEM) [N=65]',
     'NEUROSCI (BSA) [N=104]', 'PSYCH (BSPSY) [N=64]', 'MECHANICAL ENGR [N=95]', 'CIVIL ENGR [N=56]']
    task_order=['Setting up the apparatus and collecting data.',
        'Writing up the lab methods and conclusions.',
        'Analyzing data and making graphs.',
        'Managing the group progress.',
        'I prefer to be involved in all of the above.',
        'No preference or none of the above.']
    task_labels=['Equipment','Notes','Analysis','Managing','All','No preference']
    approach_order=['One where everyone takes turns with each task.',
        'One where each person has a different task.',
        'One where everyone works on each task together.',
        'No preference.',
        'Something else.']
    approach_labels=['Take turns','Different tasks','Work together','No preference','Other']
    leadership_order=['One where no one takes on the leadership role.',
        'One where the leadership role rotates between students.',
        'One where one student regularly takes on the leadership role.',
        'No preference.',
        'Something else.']
    leadership_labels=['No leader','Take turns','One leader','No preference','Other']
    x['frequency'] = 0 # a dummy column to refer to
    Nn = int(data['Maj'].count())

    # PLOT PREFERENCE OF ROLE VS MAJOR
    fig, axes = plt.subplots()
    task_counts = x.groupby(['Tasks','Maj']).count()
    task_freq_per_group = task_counts.div(task_counts.groupby('Maj').transform('sum')).reset_index()
    g = sns.barplot(x='Tasks', y='frequency', hue='Maj',data=task_freq_per_group, 
        order=task_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(task_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Preferred Role by Major')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Preferred_role_byMaj')
    plt.close()

    # PLOT PREFERENCE OF ROLE DISTRIBUTION VS MAJOR
    fig, ax = plt.subplots()
    approach_counts = x.groupby(['Approaches','Maj']).count()
    approach_freq_per_group = approach_counts.div(approach_counts.groupby('Maj').transform('sum')).reset_index()
    g = sns.barplot(x='Approaches', y='frequency', hue='Maj',data=approach_freq_per_group, 
        order=approach_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(approach_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Role Distribution by Major')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Role_distribution_byMaj')
    plt.close()

    # PLOT LEADERSHIP PREFERENCE VS MAJOR
    fig, ax = plt.subplots()
    leadership_counts = x.groupby(['Leadership','Maj']).count()
    leadership_freq_per_group = leadership_counts.div(leadership_counts.groupby('Maj').transform('sum')).reset_index()
    g = sns.barplot(x='Leadership', y='frequency', hue='Maj',data=leadership_freq_per_group, 
        order=leadership_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(leadership_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Leadership Preference by Major')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Leadership_preference_byMaj')
    plt.close()

def byClass(data):
    data = data[data.Class != 5.0]
    data['Class'] = data['Class'].replace({1.0 : 'Freshman (N=315)', 2.0 : 'Sophomore (N=367)', 3.0 : 'Junior (N=537)', 4.0 : 'Senior (N=641)'}) #, 5.0 : 'Super Senior (N=1)'})

    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0,1,4))
    palette ={'Freshman (N=315)' : colors[0], 'Sophomore (N=367)' : colors[1], 'Junior (N=537)' : colors[2], 'Senior (N=641)' : colors[3]}
    hue_order=['Freshman (N=315)', 'Sophomore (N=367)', 'Junior (N=537)', 'Senior (N=641)']
    task_order=['Setting up the apparatus and collecting data.',
        'Writing up the lab methods and conclusions.',
        'Analyzing data and making graphs.',
        'Managing the group progress.',
        'I prefer to be involved in all of the above.',
        'No preference or none of the above.']
    task_labels=['Equipment','Notes','Analysis','Managing','All','No preference']
    approach_order=['One where everyone takes turns with each task.',
        'One where each person has a different task.',
        'One where everyone works on each task together.',
        'No preference.',
        'Something else.']
    approach_labels=['Take turns','Different tasks','Work together','No preference','Other']
    leadership_order=['One where no one takes on the leadership role.',
        'One where the leadership role rotates between students.',
        'One where one student regularly takes on the leadership role.',
        'No preference.',
        'Something else.']
    leadership_labels=['No leader','Take turns','One leader','No preference','Other']
    data['frequency'] = 0 # a dummy column to refer to
    Nn = int(data['Class'].count())

    # PLOT PREFERENCE OF ROLE VS CLASS
    fig, axes = plt.subplots()
    task_counts = data.groupby(['Tasks','Class']).count()
    task_freq_per_group = task_counts.div(task_counts.groupby('Class').transform('sum')).reset_index()
    # task_err = np.sqrt(task_counts*((task_counts.groupby('Class').transform('sum')) - task_counts) / (task_counts.groupby('Class').transform('sum'))) / (task_counts.groupby('Class').transform('sum'))
    g = sns.barplot(x='Tasks', y='frequency', hue='Class',data=task_freq_per_group, 
        order=task_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(task_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Preferred Role by Class')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Preferred_role_byCls')
    plt.close()

    # PLOT PREFERENCE OF ROLE DISTRIBUTION VS CLASS
    fig, ax = plt.subplots()
    approach_counts = data.groupby(['Approaches','Class']).count()
    approach_freq_per_group = approach_counts.div(approach_counts.groupby('Class').transform('sum')).reset_index()
    g = sns.barplot(x='Approaches', y='frequency', hue='Class',data=approach_freq_per_group, 
        order=approach_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(approach_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Role Distribution by Class')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Role_distribution_byCls')
    plt.close()

    # PLOT LEADERSHIP PREFERENCE VS CLASS
    fig, ax = plt.subplots()
    leadership_counts = data.groupby(['Leadership','Class']).count()
    leadership_freq_per_group = leadership_counts.div(leadership_counts.groupby('Class').transform('sum')).reset_index()
    g = sns.barplot(x='Leadership', y='frequency', hue='Class',data=leadership_freq_per_group, 
        order=leadership_order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(leadership_labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Leadership Preference by Class')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
    plt.tight_layout()
    save_fig(g,'Leadership_preference_byCls')
    plt.close()

def group_comps(data):
    # READ IN ADDITIONAL GROUP INFO AND MERGE WITH OTHER DAATA
    df4 = pd.read_csv('GroupsS22.csv', encoding = "utf-8", usecols= ['Student EID','Group #'])
    df4 = df4.rename(columns={'Student EID':'UTEid'})
    data = pd.merge(data,df4,on='UTEid')
    # RECAST BOTH UNIQUE AND GROUP #S AS INTEGERS FOR COMPARISON. EMPTY GROUPS GIVEN SOMETHING OUT OF RANGE (0)
    data['Group #'] = data['Group #'].fillna(0).astype(np.int64)
    data['Unique'] = data['Unique'].astype(np.int64)

    group_data = pd.DataFrame(columns = ['Course', 'Unique', 'Group #', 'Comp'])
    for i in range(55375,55790):    # ITERATE OVER ALL POSSIBLE SECTIONS
        for j in range(1,11):       # ITERATE OVER ALL POSSIBLE GROUPS
            x = data.loc[(data['Unique']==i) & (data['Group #']==j)]    # CHECK IF STUDENTS ARE IN THE SAME GROUP
            if x.empty:
                pass
            else:
                if (x['Sex'] == 'M').all():
                    group_data = group_data.append({'Course': x['Course'].iloc[0], 'Unique': i, 'Group #': j, 'Comp': 'M'},ignore_index=True)
                elif (x['Sex'] == 'F').all():
                    group_data = group_data.append({'Course': x['Course'].iloc[0], 'Unique': i, 'Group #': j, 'Comp': 'F'},ignore_index=True)
                else:
                    group_data = group_data.append({'Course': x['Course'].iloc[0], 'Unique': i, 'Group #': j, 'Comp': 'Mixed'},ignore_index=True)

    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0,1,2))
    palette ={'105M': colors[0], '105N': colors[1]}
    hue_order = ['105M','105N']
    order=['F','M','Mixed']
    labels=['All female','All male','Mixed']

    group_data['frequency'] = 0 # a dummy column to refer to
    fig, ax = plt.subplots()
    # g = sns.catplot(x='Comp', hue='Course',data=group_data, kind='count')
    counts = group_data.groupby(['Comp','Course']).count()
    freq_per_group = counts.div(counts.groupby('Course').transform('sum')).reset_index()
    g = sns.barplot(x='Comp', y='frequency', hue='Course',data=freq_per_group,
        order=order,hue_order=hue_order,palette=palette)
    g.set_xticklabels(labels)
    g.set_xlabel('')
    g.set_ylabel('Fraction')
    plt.title('Group Composition')
    plt.tight_layout()
    save_fig(g,'Groups')
    plt.close()

def save_fig(fig, figure_name):
    fname = os.path.expanduser(f'{image_dir}/{figure_name}')
    plt.savefig(fname + '.png')
    # plt.savefig(fname + '.svg')

def main():
    # READ IN THE RELEVANT DATA BASED ON THE CSV COLUMN NAMES
    df1 = pd.read_csv('CLIPS_data.csv', encoding = "utf-8", usecols= ['Student','UTEid','Sex','Sch','Maj','Desc','Class','Unique','Course'])
    df1 = df1.astype({'Maj': str}) #RECASTING Maj AS A STRING, NOT AN INT
    df2 = pd.read_csv('Prelab0M.csv', encoding = "utf-8", usecols= ['sis_id','28247129: Which of the following experimental tasks do you prefer taking on?','28247130: Which of the following approaches to group tasks do you prefer?','28247131: Which of the following approaches to leadership do you prefer?'])
    df2 = df2.rename(columns={'sis_id':'UTEid','28247129: Which of the following experimental tasks do you prefer taking on?':'Tasks','28247130: Which of the following approaches to group tasks do you prefer?':'Approaches','28247131: Which of the following approaches to leadership do you prefer?':'Leadership'})
    df3 = pd.read_csv('Prelab0N.csv', encoding = "utf-8", usecols= ['sis_id','28247187: Which of the following experimental tasks do you prefer taking on?','28247191: Which of the following approaches to group tasks do you prefer?','28247194: Which of the following approaches to leadership do you prefer?'])
    df3 = df3.rename(columns={'sis_id':'UTEid','28247187: Which of the following experimental tasks do you prefer taking on?':'Tasks','28247191: Which of the following approaches to group tasks do you prefer?':'Approaches','28247194: Which of the following approaches to leadership do you prefer?':'Leadership'})
    concatenated = pd.concat([df2, df3])

    # COMBINE DOUBLE MAJOR STUDENTS' DATA INTO A SINGLE ROW
    index_nan, = np.where(pd.isnull(df1['Student']))
    for i in index_nan:
        # df1.at[i-1,'Sch'] = 'D' # Define Double Major Students as School 'D' (Used in classifying by Schools)
        df1.at[i-1,'Sch'] = str(df1.at[i-1,'Sch'] + ', ' + df1.at[i,'Sch'])
        df1.at[i-1,'Maj'] = str(df1.at[i-1,'Maj'] + ', ' + df1.at[i,'Maj'])
        df1.at[i-1,'Desc'] = str(df1.at[i-1,'Desc'] + ', ' + df1.at[i,'Desc'])
    df1 = df1.drop(index_nan)

    # THERE ARE DUPLICATES IN THE PRELAB DATA AND THEN REMOVING THE DUPLICATES DROPS BELOW THE INITIAL STUDENT COUNT 
    # (1915 initial vs 1926 prelab vs 1861 final)
    data = pd.merge(df1,concatenated,on='UTEid')
    data = data.drop_duplicates(subset=['UTEid'],keep='last')

    # SEPARATE BY COURSE
    data_M = data[data['Course'] == '105M']
    data_N = data[data['Course'] == '105N']

    # print(data)
    # print(data_M)
    # print(data_N)
    # data.to_csv('data.csv',index=False)

    # NEED TO INCLUDE
    # ERR = SQRT(p(1-p)/N) where p is fraction of pop and N is total pop

    # CHANGE WHAT DATA TO FEED THE PLOTS
    data = data.loc[data['Sch'].str.contains('J')]

    # # PRINT ALL ROWS IN A DATAFRAME
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(data)

    # HERE IS WHERE WE MAKE NICE PLOTS
    Holmes_plots(data)
    # group_comps(data)
    # bySchool(data)
    # byMaj(data)
    # byClass(data)   

try:
    if __name__ == '__main__':
        main()
except Exception as err:
    traceback.print_exc(file=sys.stdout)
    input("Press Enter to continue...")