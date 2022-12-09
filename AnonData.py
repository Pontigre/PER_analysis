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

# WHEN I RUN THIS I HAVE A FOLDER WHERE ALL THE CREATED FILES GO CALLED 'ExportedFiles'
image_dir = 'ExportedFiles'

def main():
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete)
    my_file = input('CLIPS Filename: ')
    my_file2 = input('PHY105M Prelab 0:')
    my_file3 = input('PHY105N Prelab 0:')
    
    # READ IN DATA
    clips = pd.read_csv(my_file, encoding = "utf-8", usecols= ['Ctr','Student','UTEid','Sex','Sch','Maj','Desc','Class','Unique','Course'])
    MPrelab = pd.read_csv(my_file2, encoding = "utf-8", usecols= ['sis_id','28247129: Which of the following experimental tasks do you prefer taking on?',
        '28247130: Which of the following approaches to group tasks do you prefer?','28247131: Which of the following approaches to leadership do you prefer?'])
    NPrelab = pd.read_csv(my_file3, encoding = "utf-8", usecols= ['sis_id','28247187: Which of the following experimental tasks do you prefer taking on?',
        '28247191: Which of the following approaches to group tasks do you prefer?','28247194: Which of the following approaches to leadership do you prefer?'])

    ExcludedHeaders = ['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)',
    'Finished', 'RecordedDate', 'ResponseId', 'RecipientLastName', 'RecipientFirstName',
    'RecipientEmail', 'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'DistributionChannel', 'UserLanguage']
    headers = [*pd.read_csv(my_file2, nrows=1)]
    SAGE = pd.read_csv(my_file2, encoding = "utf-8", usecols=lambda x: x not in ExcludedHeaders)

    # CLIPS DATA PROCESSING
    clips = clips.astype({'Maj': str}) #RECASTING Maj AS A STRING, NOT AN INT
    # COMBINE DOUBLE MAJOR STUDENTS' DATA INTO A SINGLE ROW
    index_nan, = np.where(pd.isnull(clips['Student']))
    for i in index_nan:
        if clips.at[i-1,'Sch'] != clips.at[i,'Sch']:
            clips.at[i-1,'Sch'] = 'D' # Define Double Major Students as School 'D' (Used in classifying by Schools)
            # df1.at[i-1,'Sch'] = str(df1.at[i-1,'Sch'] + ', ' + df1.at[i,'Sch'])
        clips.at[i-1,'Maj'] = str(clips.at[i-1,'Maj'] + ', ' + clips.at[i,'Maj'])
        clips.at[i-1,'Desc'] = str(clips.at[i-1,'Desc'] + ', ' + clips.at[i,'Desc'])
    clips = clips.drop(index_nan)

    # PRELAB 0 PROCESSING
    MPrelab = MPrelab.rename(columns={'sis_id':'UTEid','28247129: Which of the following experimental tasks do you prefer taking on?':'Tasks',
        '28247130: Which of the following approaches to group tasks do you prefer?':'Approaches',
        '28247131: Which of the following approaches to leadership do you prefer?':'Leadership'})
    NPrelab = NPrelab.rename(columns={'sis_id':'UTEid','28247187: Which of the following experimental tasks do you prefer taking on?':'Tasks',
        '28247191: Which of the following approaches to group tasks do you prefer?':'Approaches',
        '28247194: Which of the following approaches to leadership do you prefer?':'Leadership'})
    prelabs = pd.concat([MPrelab, NPrelab])

    data = pd.merge(clips,prelabs,on='UTEid')
    # REMOVING DUPLICATES OF EIDS FROM STUDENTS SUBMITTING MULTIPLE PRELABS
    data = data.drop_duplicates(subset=['UTEid'],keep='last')

    # CREATE FILE FOR STUDENT REFERENCE AND REMOVE STUDENT INFORMATION
    df3 = data[['Ctr','Student','UTEid']]
    df1 = data.drop(columns=['Student', 'UTEid'])

    # EXPORT TO ANONYMIZED CSVS
    df1.to_csv('ExportedFiles/CLIPSPreferences_anon.csv', index=False)
    df3.to_csv('ExportedFiles/StudentReference.csv', index=False)
    print('Exported anonymized files')

def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

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

try:
    if __name__ == '__main__':
        main()
except Exception as err:
    traceback.print_exc(file=sys.stdout)
    input("Press Enter to exit...")