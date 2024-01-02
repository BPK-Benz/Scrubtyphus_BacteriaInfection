import json, os, glob, rbo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
import numpy as np
from sklearn.metrics import precision_score, recall_score

def convert_longtowide(df_long, suffix):
    df_wide = df_long.pivot(index=df_long.columns[0], columns=df_long.columns[1], values=df_long.columns[2])
    df_wide = df_wide.reset_index()

    # Add the suffix to the last three columns
    df_wide.columns = [col if i < len(df_wide.columns) - 3 else col + suffix for i, col in enumerate(df_wide.columns)]

    return df_wide

def order_model(dataset):
    def model_to_number(model_name):
        if 'CascadeMask' in model_name:
            return 1
        elif 'MaskRCNN' in model_name:
            return 2
        elif 'FasterRCNN' in model_name:
            return 3
        elif 'Retina' in model_name:
            return 4
        else:
            return 0  # Default case if none of the above
    def model_to_number2(model_name):
        if 'R101_fp16' in model_name:
            return 2
        elif 'R101' in model_name:
            return 1
        elif 'R50_fp16' in model_name:
            return 4
        elif 'R50' in model_name:
            return 3
        else:
            return 0  # Default case if none of the above

    dataset['order'] = dataset['Techniques'].apply(model_to_number)
    dataset['order2'] = dataset['Techniques'].apply(model_to_number2)
    dataset = dataset.sort_values(by=['order','order2'])
    return dataset

def jaccard_similarity(list1, list2):
    """Calculate the Jaccard Similarity between two lists."""
    # Convert both lists to sets to remove duplicates and allow set operations
    set1 = set(list1)
    set2 = set(list2)

    # Find the intersection (common elements) between the two sets
    intersection = set1.intersection(set2)
    # print(intersection)

    # Find the union of the two sets
    union = set1.union(set2)
    # print(union)

    # Calculate the Jaccard Similarity
    similarity = len(intersection) / len(union)

    return similarity , len(intersection)


########## 1. Load all data results##########
# keep image information
df_annotations = pd.read_csv(glob.glob(os.getcwd()+'/ImageID')[0]).iloc[:,1:]
name_mapping = {
    'Plate_03': 'Plate_01',
    'Plate_04': 'Plate_02',
    'Plate_07': 'Plate_03'
}
df_annotations['plate_name'] = [_.split('/')[3] for _ in df_annotations['file_name']]
df_annotations['plate_name'] = df_annotations['plate_name'].replace(name_mapping)
df_annotations['cell_type'] = [_.split('/')[4] for _ in df_annotations['file_name']]
df_annotations['image'] = [_.split('/')[5][:6] for _ in df_annotations['file_name']]

# keep groundtruth from json file
df_long = pd.read_csv(glob.glob(os.getcwd()+'/Json*')[0]).iloc[:,1:]
df_wide = convert_longtowide(df_long, '_Ground Truth')

merged_df = pd.merge(df_annotations, df_wide, on='file_name')

# keep image processing results
ip_df = pd.read_csv(glob.glob(os.getcwd()+'/IP_Infect*')[0]).iloc[:,1:]
ip_df.columns = ['Gene','Border', 'Divide', 'Uninfected cell_Image Processing','Infected cell_Image Processing']
ip_df['Irrelevant cell_Image Processing'] = ip_df['Border']+ip_df['Divide']

ip_df['plate_name'] = [_.split('/')[0] for _ in ip_df['Gene']]
ip_df['plate_name'] = ip_df['plate_name'].replace(name_mapping)
ip_df['cell_type'] = [_.split('/')[1] for _ in ip_df['Gene']]
ip_df['image'] = [_.split('/')[2][:6] for _ in ip_df['Gene']]
merged_df = pd.merge(merged_df, ip_df.iloc[:,[-1,-2,-3,-5,-4,-6]], on=['plate_name','cell_type','image'])

# keep model prediction results
keep_model = []
for _ in sorted(glob.glob(os.getcwd()+'/*Infect/my_cell_det_output.bbox.json')):
    model = _.split('/')[-2][:-7]
    keep_model += [model]
    
    # keep model prediction results
    with open(_, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    category_mapping = {
        1: 'Infected cell',
        2: 'Uninfected cell',
        3: 'Irrelevant cell'
    }

    df['category_id'] = df['category_id'].replace(category_mapping)
    df_long = pd.DataFrame(df[(df['score'] >= 0.6)].groupby(['image_id', 'category_id']).count()['score']).reset_index()
    df_wide = convert_longtowide(df_long, '_'+model)
    df_wide 


    merged_df = pd.merge(merged_df, df_wide, on='image_id')

keep_model = ['Ground Truth', 'Image Processing'] + keep_model  
merged_df.columns = [item.replace('Casecade', 'Cascade') for item in merged_df.columns.tolist()]

# counting performance evaluation
count_pergene = merged_df.iloc[:,2:].groupby(['plate_name','cell_type','image']).sum().reset_index()
infect = pd.concat([count_pergene.iloc[:,:3],count_pergene.iloc[:,3::3]],axis=1)
irrelevant = pd.concat([count_pergene.iloc[:,:3],count_pergene.iloc[:,4::3]],axis=1)
uninfect = pd.concat([count_pergene.iloc[:,:3],count_pergene.iloc[:,5::3]],axis=1)

###### convert long format to wide format
wide_infect = pd.melt(infect, id_vars=['plate_name', 'cell_type', 'image'], var_name='Techniques', value_name='Cell_numbers')
wide_infect['Techniques'] = [_[14:] for _ in wide_infect['Techniques']]


wide_irrelevant = pd.melt(irrelevant, id_vars=['plate_name', 'cell_type', 'image'], var_name='Techniques', value_name='Cell_numbers')
wide_irrelevant['Techniques'] = [_[16:] for _ in wide_irrelevant['Techniques']]

wide_uninfect = pd.melt(uninfect, id_vars=['plate_name', 'cell_type', 'image'], var_name='Techniques', value_name='Cell_numbers')
wide_uninfect['Techniques'] = [_[16:] for _ in wide_uninfect['Techniques']]


###### 2 Counting cell assessment ################
###### 2.1 plot results in gene level for counting of cell's type ##########
# Set style
sns.set_style('whitegrid')

# Create a figure with a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)  # 3 rows, 3 columns

# Datasets
datasets = [wide_infect, wide_uninfect, wide_irrelevant]
dataset_names = ['Infected', 'Uninfected', 'Irrelevant']

# Plates
plates = ['Plate_01', 'Plate_02', 'Plate_03']

# Colors and hatches
colors = ['#F9B360', 'lightgray'] + ['#FCFA9F']*4 + ['#baffc9']*4 + ['lightpink']*4 + ['#bae1ff']*4
hatches = ['', ''] + ['', '..', 'O', 'XX'] * 4


# Y-Axis limits and star positions for each dataset
ymax_values = {'Infected': 900, 'Uninfected': 2700, 'Irrelevant': 1200}
star_positions = {'Infected': 850, 'Uninfected': 2600, 'Irrelevant': 1100}

# Iterate over each dataset and plate
for i, dataset in enumerate(datasets):
    dataset = order_model(dataset)
    dataset_name = dataset_names[i]
    for j, plate in enumerate(plates):
        ax = axes[i, j]
        num_techniques = len(dataset['Techniques'].unique())
        sns.boxplot(data=dataset[dataset['plate_name'] == plate], x='Techniques', y='Cell_numbers', palette=colors, ax=ax)

        # Set subplot title and ylabel
        ax.set_title(plate if i == 0 else '')
        ax.set_ylabel(f'{dataset_name} Cell Numbers' if j == 0 else '')

        # Set y-axis limit
        ax.set_ylim(0, ymax_values[dataset_name])

        # Adding hatch patterns and red stars
        for k, bar in enumerate(ax.patches):
            hatch_index = k % len(hatches)
            bar.set_hatch(hatches[hatch_index])
            bar.set_edgecolor('gray')

        ax.tick_params(axis='x', rotation=90)
        
        
# Adjust layout
plt.tight_layout()
plt.savefig('InfectCountingPerformance_NoStar.png', dpi=300)
plt.show()


######## 2.2 calculate wilcoxon sign-rank test

# Datasets
datasets = [infect, uninfect, irrelevant]
dataset_names = ['Infected', 'Uninfected', 'Irrelevant']

# Plates
plates = ['Plate_01', 'Plate_02', 'Plate_03']
# Iterate over each dataset and plate
keep_whole = pd.DataFrame()
for ii, dataset in enumerate(datasets):
    keep_datasets = pd.DataFrame()
    for j, plate in enumerate(plates):
        data=dataset[dataset['plate_name'] == plate]
        data1 = data[data.columns[3]]
        keep_p = []
        for i in range(4, data.shape[1]):
            data2 = data.iloc[:,i]
            w, p = wilcoxon(data1, data2)
            if p < 0.05:
                outcome = "Sig."
            else:
                outcome = "No Sig."

            keep_p += [[infect.columns[i], p, outcome]]
            
        p_df = pd.DataFrame(keep_p, columns=['model_'+plate+dataset_names[ii],'p-value_'+plate+dataset_names[ii],'interpret_'+plate+dataset_names[ii]])
        keep_datasets = pd.concat([keep_datasets, p_df], axis=1)
        
    keep_whole = pd.concat([keep_whole, keep_datasets], axis=1)
    
# read confusion matrix at 0.6 score and 0.8 iou from all models ########   
keep_modelcm = []
for _ in sorted(glob.glob(os.getcwd()+'/*/confusion_matrix_score_0.6_iou_0.8.csv')):
    temp = pd.read_csv(_)
    keep_cm = []
    for r in range(temp.shape[0]):
        # Confusion matrix
        cm = [list(map(float, row.replace('[', '').replace(']', '').split())) for row in temp.iloc[r,2].split('\n')]
        keep_cm += [cm]
    temp['CM'] = keep_cm
    
    # Initialize the sum with the first value
    sum_result = temp.iloc[0, 3]

    # Iteratively add the next values
    for i in range(1, temp.shape[0]):  # Assuming you want to add the first 4 elements in the column
        sum_result = np.add(sum_result, temp.iloc[i, 3])

    keep_modelcm += [[_.split('New_OCT')[1].split('/')[1],sum_result]]
    
pr_model = pd.DataFrame(keep_modelcm, columns=['model','confusion matrix'])
keep_sums =[]
for _ in range(pr_model.shape[0]):
    array = pr_model['confusion matrix'][_]
    # Summing specific elements
    sums = [array[0, 0] + array[0, 3],  # 2773 + 731
            array[1, 1] + array[1, 3],  # 10581 + 1844
            array[2, 2] + array[2, 3]]  # 5000 + 1497
    keep_sums += [sums]
pr_model['check'] = keep_sums
pr_model['model'] =[item.replace('Casecade', 'Cascade') for item in pr_model['model'].tolist()]

# merge all model results
nets = [pd.DataFrame() for _ in range(4)]

for _ in range(pr_model.shape[0]):
    confusion_matrix = pr_model['confusion matrix'][_][:-1, :-1]

    # Calculating precision, recall, and F1 score for each class
    precision_per_class = np.round(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0),4)
    recall_per_class = np.round(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1),4)
    f1_score_per_class = np.round(2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class),4)

    # Calculating weighted average of precision, recall, and F1 score
    total_instances = np.sum(confusion_matrix)
    weighted_precision = np.round(np.sum(precision_per_class * np.sum(confusion_matrix, axis=1) / total_instances),4)
    weighted_recall = np.round(np.sum(recall_per_class * np.sum(confusion_matrix, axis=1) / total_instances),4)
    weighted_f1_score = np.round(np.sum(f1_score_per_class * np.sum(confusion_matrix, axis=1) / total_instances),4)

    # Creating a DataFrame
    metrics_df = pd.DataFrame({
        'Class': ['Infected cells', 'Uninfected cells', 'Irrelevant cells', 'Weighted Average'],
        'Precision': list(precision_per_class) + [weighted_precision],
        'Recall': list(recall_per_class) + [weighted_recall],
        'F1 Score': list(f1_score_per_class) + [weighted_f1_score]
    })

    # Adding the number of instances for each class to the DataFrame
    num_instances_per_class = np.sum(confusion_matrix, axis=1).astype(int)
    metrics_df['Support'] = list(num_instances_per_class) + [total_instances]
        
        
    # Determine which net DataFrame to append to based on the index
    net_index = _ // 4
    nets[net_index] = pd.concat([nets[net_index], metrics_df], axis=1)

###### 3 identify significant gene (FC of infection ratio)  ################
# Generate ratio of infection = infected cells/sum(infected cells+uninfected cells)
for _ in range(3,57,3):
    infect_ratio = count_pergene.iloc[:,_]/(count_pergene.iloc[:,_]+count_pergene.iloc[:,_+2])
    name_col = 'Infection ratio_'+list(set([_.split('cell_')[1] for _ in count_pergene.iloc[:,_:_+3].columns]))[0]
    count_pergene[name_col] = infect_ratio
    
# Select only ratio of infection    
infectRatio = pd.concat([count_pergene.iloc[:,:3],count_pergene.iloc[:,-18:]], axis=1)

# Change lone df to wide df
wide_infectRatio = pd.melt(infectRatio, id_vars=['plate_name', 'cell_type', 'image'], var_name='Techniques', value_name='Cell_numbers')
wide_infectRatio['Techniques'] = [_[16:] for _ in wide_infectRatio['Techniques']]

# Determine mean of scramble and keep knockdown gene
scramble_infectRatio = wide_infectRatio[wide_infectRatio['cell_type']=='Scramble'].iloc[:,[0,-2,-1]].groupby(['plate_name','Techniques']).mean().round(3).reset_index()
gene_infectRatio = wide_infectRatio[wide_infectRatio['cell_type']!='Scramble']

# Determin the effect of knockdown gene to scramble gene in each plate
keep_infectRatio = []
for p in sorted(set(wide_infectRatio['plate_name'])):
    for t in sorted(set(wide_infectRatio['Techniques'])):
        scramble = scramble_infectRatio[(scramble_infectRatio['plate_name']==p)&(scramble_infectRatio['Techniques']==t)]['Cell_numbers'].tolist()[0]
        gene = gene_infectRatio[(gene_infectRatio['plate_name']==p)&(gene_infectRatio['Techniques']==t)]['Cell_numbers']
        gene_name = gene_infectRatio[(gene_infectRatio['plate_name']==p)&(gene_infectRatio['Techniques']==t)]['image']
        for i, ii in zip(gene_name, gene):
            keep_infectRatio += [[p,t,i, round(ii/scramble,3)]]
            
# Keep dataframe of the effect of knockdown gene           
ratio_infectRatio = pd.DataFrame(keep_infectRatio, columns=['plate_name','Techniques','image','ratio'])

keep_value = []
for p,t,image in zip(gene_infectRatio['plate_name'],gene_infectRatio['Techniques'],gene_infectRatio['image']):
    ratio = ratio_infectRatio[(ratio_infectRatio['plate_name']==p)&
                         (ratio_infectRatio['Techniques']==t)&
                         (ratio_infectRatio['image']==image)]['ratio'].tolist()[0]
    keep_value += [ratio]
            
gene_infectRatio['ratio'] = keep_value

# Sorted the effect of gene in each plate 
list_gene_infectRatio = []
for p in sorted(set(gene_infectRatio['plate_name'])):
    for t in sorted(set(gene_infectRatio['Techniques'])):
        temp = gene_infectRatio[(gene_infectRatio['plate_name']==p)&
                             (gene_infectRatio['Techniques']==t)].sort_values(by=['ratio'])
        list_gene_infectRatio += [[p,t,temp['image'].tolist()]]
        
list_genedf_infectRatio = pd.DataFrame(list_gene_infectRatio, columns=['plate_name','Techniques','list gene'])
list_genedf_infectRatio = order_model(list_genedf_infectRatio)

# Merged all knockdown gene in all three plates (one techniques has one list knockdown genes)
merged_list = []
for t in set(list_genedf_infectRatio['Techniques']):
    t_list = []
    for _ in list_genedf_infectRatio[list_genedf_infectRatio['Techniques']==t]['list gene']:
        t_list += [_]
    merged_list += [[t,_]]
    
bio = pd.DataFrame(merged_list, columns= ['Techniques','list gene'])

# Calculate RBO (rank biased overlap) RBO compares two ranked lists, and returns a numeric value between zero and one to quantify their similarity. 
# A RBO value of zero indicates the lists are completely different, and a RBO of one means completely identical. 
# The terms 'different' and 'identical' require a little more clarification.
gt = bio[bio['Techniques']=='Ground Truth']['list gene'].tolist()[0]
keep_rbo = []
keep_hit = []
for _ in bio['Techniques'].tolist():
    tech = bio[bio['Techniques']==_]['list gene'].tolist()[0]

    rbo_score = np.round(rbo.RankingSimilarity(gt, tech).rbo(),3)
    keep_rbo += [rbo_score]
    keep_hit += [tech[-5:]]

bio['rbo score'] = keep_rbo
bio['hit genes'] = keep_hit
order_model(bio).sort_values(by=['order','order2','Techniques'])

keep_generbo = []
for _ in set(gene_infectRatio['Techniques']):
    g1 = gene_infectRatio[gene_infectRatio['Techniques']== 'Ground Truth'].sort_values(by='ratio')['Knockdown genes'].tolist()
    g2 = gene_infectRatio[gene_infectRatio['Techniques']== _].sort_values(by='ratio')['Knockdown genes'].tolist()
    keep_generbo += [[_, np.round(rbo.RankingSimilarity(g1, g2).rbo(),3)]]

geneRBO = pd.DataFrame(keep_generbo, columns=['Techniques','RBO']).sort_values(by='RBO', ascending=False).reset_index(drop=True)
gene_infectRatio.iloc[:,[3,4,-2]].pivot(index='Knockdown genes', columns='Techniques', values='ratio').sort_values(by='Ground Truth')



###### 4 identify significant gene (2 Threshold critera)  ################
vis = gene_infectRatio[(gene_infectRatio['color']== 'Ground Truth') | 
                 (gene_infectRatio['color']== 'Image Processing')
                | (gene_infectRatio['Techniques']== 'FasterRCNN_R50_fp16')
                | (gene_infectRatio['Techniques']== 'CascadeMask_R50')]
colors_vis = ['red','gray','green','#836EF9','green','blue']

plt.figure(figsize=(10, 6))
line_styles = ["-", "--", "-.", ":"]

# Plot each technique separately
unique_techniques = vis['Techniques'].unique()
for i, technique in enumerate(unique_techniques):
    subset = vis[vis['Techniques'] == technique]
    sns.lineplot(data=subset, x='Knockdown genes', y='ratio', color=colors_vis[i], 
                 marker='o', linestyle=line_styles[i], label=technique)

plt.title('Identifying significant genes by fold change of bacteria infection')
plt.xlabel('Knockdown gene')
plt.ylabel('Fold change of bacteria infection')
plt.xticks(rotation=90)
plt.legend(title='Techniques', loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig('FoldChange_RBO.png', dpi=300)
plt.show()

# Generate ratio infection = infected cells/sum(infected cells+uninfected cells)
for _ in range(3,57,3):
    totalcount = (count_pergene.iloc[:,_]+count_pergene.iloc[:,_+2])
    name_col = 'Total_'+list(set([_.split('cell_')[1] for _ in count_pergene.iloc[:,_:_+3].columns]))[0]
    count_pergene[name_col] = totalcount

cri_gene = count_pergene[count_pergene['cell_type']!='Scramble'].set_index('plate_name').iloc[:,2:].reset_index()
for method in ['Ground Truth','Image Processing','FasterRCNN_R101', 'CasecadeMask_R101']:
    temp = cri_gene.loc[:,cri_gene.columns.str.endswith(method)]
    temp = pd.concat([cri_gene['plate_name'],temp], axis=1)

    cri_scramble = count_pergene[count_pergene['cell_type']=='Scramble'].set_index('plate_name').iloc[:,2:].reset_index()
    mean_scramble  = cri_scramble.groupby('plate_name').mean()
    mean_scramble = mean_scramble.loc[:,mean_scramble.columns.str.endswith(method)]*0.7
    mean_scramble

    # Create a single figure with three subplots, one for each 'plate_name' category
    plt.figure(figsize=(15, 4))  # Adjust the size of the figure as needed

    unique_plates = temp['plate_name'].unique()
    for i, plate in enumerate(unique_plates, 1):
        plt.subplot(1, len(unique_plates), i)  # Creating subplots in a row
        sns.scatterplot(data=temp[temp['plate_name'] == plate], x='Total_'+method, 
                        y='Infected cell_'+method, color='gray')
        t = mean_scramble.loc[plate,'Total_'+method].round(1)
        plt.vlines(x=t, ymax=400, ymin=0, linestyles='--', color='red')
        inf = mean_scramble.loc[plate,'Infected cell_'+method].round(1)
        plt.hlines(y=inf, xmax=2300, xmin=0, linestyles='--', color='red')
        plt.title(f"{plate}")

    plt.tight_layout()
    plt.savefig('Criteria_'+method+'.png', dpi=300)
    plt.show()
    
    keep_dataDF = pd.DataFrame()
    
for _ in order_model(bio).sort_values(by=['order','order2','Techniques'])['Techniques'].tolist():
    col = count_pergene.loc[:, count_pergene.columns.str.endswith(_)]
    
    temp = cri_gene.loc[:,cri_gene.columns.str.endswith(_)]
    temp = pd.concat([cri_gene['plate_name'],temp], axis=1)

    cri_scramble = count_pergene[count_pergene['cell_type']=='Scramble'].set_index('plate_name').iloc[:,2:].reset_index()
    mean_scramble  = cri_scramble.groupby('plate_name').mean()
    mean_scramble = mean_scramble.loc[:,mean_scramble.columns.str.endswith(_)]*0.7

    unique_plates = temp['plate_name'].unique()
    keep_data = pd.DataFrame()
    for i, plate in enumerate(unique_plates, 1):
        data = temp[temp['plate_name'] == plate]

        inf = mean_scramble.loc[plate,col.columns[0]].round(1)
        data['cri_inf_'+_] = ['pass' if x > inf else 'no' for x in data[col.columns[0]]]

        t = mean_scramble.loc[plate,col.columns[4]].round(1)
        data['cri_total_'+_] = ['pass' if x > t else 'no' for x in data[col.columns[4]]]

        keep_data = pd.concat([keep_data, data], axis=0)
    keep_dataDF = pd.concat([keep_dataDF,keep_data], axis=1)


gene_image = count_pergene[count_pergene['cell_type']!='Scramble'].reset_index(drop=True).iloc[:,:3]
cri = pd.concat([gene_image,keep_dataDF], axis=1)


keep_jaccard = []
for _ in jaccard.columns[4:].tolist():
    g1 = jaccard[jaccard['Ground Truth'] == 'hit']['Knockdown gene'].tolist()
    g2 = jaccard[jaccard[_] == 'hit']['Knockdown gene'].tolist()
    keep_jaccard += [[_, len(g2), np.round(jaccard_similarity(g1, g2)[1],0), np.round(jaccard_similarity(g1, g2)[0],3)]]

jaccard_score = pd.DataFrame(keep_jaccard, columns=['Techniques', 'Passed gene', 'Similar gene','Jaccard']).sort_values(by='Jaccard', ascending=False).reset_index(drop=True)
jaccard_score