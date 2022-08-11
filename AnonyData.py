# USED FOR ERROR TRACKING
import os
import traceback
import sys

# NORMAL PACKAGES
import readline, glob
import pandas as pd
import numpy as np

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
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete)
    my_file = input('CLIPS Filename: ')
    my_file2 = input('SAGE Filename: ')

    # READ IN DATA FROM CLIPS BASED ON THE CSV COLUMN NAMES
    df1 = pd.read_csv(my_file, encoding = "utf-8", usecols= ['Ctr','Student','UTEid','Sex','Sch','Maj','Desc','Class','Unique','Course'])
    df1 = df1.astype({'Maj': str}) #RECASTING Maj AS A STRING, NOT AN INT
    # COMBINE DOUBLE MAJOR STUDENTS' DATA INTO A SINGLE ROW
    index_nan, = np.where(pd.isnull(df1['Student']))
    for i in index_nan:
        # df1.at[i-1,'Sch'] = 'D' # Define Double Major Students as School 'D' (Used in classifying by Schools)
        df1.at[i-1,'Sch'] = str(df1.at[i-1,'Sch'] + ', ' + df1.at[i,'Sch'])
        df1.at[i-1,'Maj'] = str(df1.at[i-1,'Maj'] + ', ' + df1.at[i,'Maj'])
        df1.at[i-1,'Desc'] = str(df1.at[i-1,'Desc'] + ', ' + df1.at[i,'Desc'])
    df1 = df1.drop(index_nan)

    # CREATE FILE FOR STUDENT REFERENCE
    df3 = df1[['Ctr','Student','UTEid']]
    df1= df1.drop(columns=['Student', 'UTEid'])

    # READ IN DATA FROM SAGE BASED ON THE CSV COLUMN NAMES
    headers = [*pd.read_csv(my_file2, nrows=1)]
    ExcludedHeaders = ['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)',
    'Finished', 'RecordedDate', 'ResponseId', 'RecipientLastName', 'RecipientFirstName',
    'RecipientEmail', 'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'DistributionChannel', 'UserLanguage']
    df2 = pd.read_csv(my_file2, encoding = "utf-8", usecols=lambda x: x not in ExcludedHeaders)
    df2 = df2.drop([1])

    # EXPORT TO ANONYMIZED CSVS
    df1.to_csv('ExportedFiles/CLIPS_anony.csv', index=False)
    df2.to_csv('ExportedFiles/SAGE_anony.csv', index=False)
    df3.to_csv('ExportedFiles/StudentReference.csv', index=False)
    print('Exported anonymized files')

try:
    if __name__ == '__main__':
        main()
except Exception as err:
    traceback.print_exc(file=sys.stdout)
    input("Press Enter to exit...")