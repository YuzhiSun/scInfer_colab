import pandas as pd
from torch.utils import data
import torch
from sklearn.model_selection import train_test_split
import anndata as ad
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def make_paired_samples(rna, protein, sample_nums=15000, celltype='celltype'):
    rna_ids = rna.obs[celltype]
    rna_ids = rna_ids.reset_index()
    protein_ids = protein.obs[celltype].reset_index()

    print(f"Counts of all celltype:\n {protein_ids.groupby(celltype).count().rename(columns={'index': 'count'})}")

    if (len(protein_ids[protein_ids.duplicated()]) != 0) | (len(rna_ids[rna_ids.duplicated()]) != 0):
        print("There are duplicated ids in protein or rna!!!")

    combine_df = pd.merge(rna_ids, protein_ids, how='cross')
    combine_df['label'] = (combine_df[f'{celltype}_x'].astype(str) == combine_df[f'{celltype}_y'].astype(str)).astype(int)
    combine_df_pos = combine_df[combine_df['label'] == 1].groupby([f'{celltype}_y'], group_keys=False).apply(
        lambda x: x.sample(min(sample_nums, len(x))))
    combine_df_neg = combine_df[combine_df['label'] == 0].groupby([f'{celltype}_y'], group_keys=False).apply(
        lambda x: x.sample(min(sample_nums, len(x))))
    print(f"neg_samples:{combine_df_neg.shape[0]}, pos_samples:{combine_df_pos.shape[0]}")
    sample_df = pd.concat([combine_df_pos, combine_df_neg])
    sample_df = sample_df[['index_x', 'index_y', 'label']]

    return sample_df

def make_rna_prt_index(rna, protein, celltype='celltype'):
    rna_ids = rna.obs[celltype]
    rna_ids = rna_ids.reset_index()
    protein_ids = protein.obs[celltype].reset_index()
    print(
        f"Counts of all celltype in rna:\n {rna_ids.groupby(celltype).count().rename(columns={'index': 'count'})}")
    print(f"Counts of all celltype in protein:\n {protein_ids.groupby(celltype).count().rename(columns={'index': 'count'})}")
    return rna_ids, protein_ids

def make_train_test_id(ids, by='celltype', test_frac = 0.3):
    def split_group(group, fraction=0.3):
        # 打乱每个分组的数据
        group = group.sample(frac=1, random_state=1).reset_index(drop=True)
        # 计算分割点
        split_point = int(len(group) * fraction)
        # 分割数据
        group_test = group[:split_point]
        group_train = group[split_point:]
        return group_test, group_train
    grouped = ids.groupby(by)
    group_test_lst, group_train_lst = [], []
    for name, group in grouped:
        g_test, g_train = split_group(group, test_frac)
        group_test_lst.append(g_test)
        group_train_lst.append(g_train)
    test_id = pd.concat(group_test_lst).reset_index(drop=True)
    train_id = pd.concat(group_train_lst).reset_index(drop=True)
    return train_id, test_id

def make_paired_samples_for_benchmark(rna_ids, protein_ids, celltype, sample_nums):
    combine_df = pd.merge(rna_ids, protein_ids, how='cross')
    combine_df['label'] = (combine_df[f'{celltype}_x'].astype(str) == combine_df[f'{celltype}_y'].astype(str)).astype(int)
    combine_df_pos = combine_df[combine_df['label'] == 1].groupby([f'{celltype}_y'], group_keys=False).apply(
        lambda x: x.sample(min(sample_nums, len(x))))
    combine_df_neg = combine_df[combine_df['label'] == 0].groupby([f'{celltype}_y'], group_keys=False).apply(
        lambda x: x.sample(min(sample_nums, len(x))))
    print(f"neg_samples:{combine_df_neg.shape[0]}, pos_samples:{combine_df_pos.shape[0]}")
    sample_df = pd.concat([combine_df_pos, combine_df_neg])
    sample_df = sample_df[['index_x', 'index_y', 'label']]

    return sample_df

class scDataset(data.Dataset):
    def __init__(self, protein_df, rna_df, relations):
        self.protein_data = protein_df
        self.rna_data = rna_df
        self.rna_id = relations['index_x'].tolist()
        self.protein_id = relations['index_y'].tolist()
        self.label = relations['label'].tolist()

    def __getitem__(self, item):
        protein_id = self.protein_id[item]
        rna_id = self.rna_id[item]
        protein_tensor = torch.tensor(self.protein_data.loc[protein_id, :].values, dtype=torch.float)
        rna_tensor = torch.tensor(self.rna_data.loc[rna_id, :].values, dtype=torch.float)

        label = torch.tensor(self.label[item], dtype=torch.int)
        return protein_tensor, rna_tensor, label

    def __len__(self):
        return len(self.label)

    def getid(self, item):
        protein_id = self.protein_id[item]
        rna_id = self.rna_id[item]
        return protein_id, rna_id

    def get_size(self):
        return {'protein': self.protein_data.shape[1],
                'rna': self.rna_data.shape[1]}

def min_max(df):
    return (df - df.min()) / (df.max() - df.min())

def make_scInfer_dataset(rna, protein, sample_df, rna_vars='highly_variable', protein_vars=None, test_size=0.3, random_state=2024):
    train_relation, valid_relation = train_test_split(sample_df, test_size=test_size, random_state=random_state)
    rna_df = rna[:, rna.var[rna_vars]].to_df()

    if protein_vars:
        protein_df = protein[:, protein.var[protein_vars] > 0].to_df()
    else:
        protein_df = protein.to_df()

    protein_df = min_max(protein_df)
    rna_df = min_max(rna_df)
    train_dataset = scDataset(protein_df, rna_df, train_relation)
    valid_dataset = scDataset(protein_df, rna_df, valid_relation)
    return train_dataset, valid_dataset, rna_df, protein_df

def make_scInfer_dataframe(rna, protein, rna_vars='highly_variable', protein_vars=None):
    rna_df = rna[:, rna.var[rna_vars]].to_df()
    if protein_vars:
        protein_df = protein[:, protein.var[protein_vars] > 0].to_df()
    else:
        protein_df = protein.to_df()

    protein_df = min_max(protein_df)
    rna_df = min_max(rna_df)
    protein_df.fillna(0, inplace=True)
    rna_df.fillna(0, inplace=True)
    return rna_df, protein_df

def make_unlabeled_dataframe(rna, unlabeled_rna, rna_vars='highly_variable'):
    unlabeled_rna_df = unlabeled_rna[:, rna.var[rna_vars]].to_df()
    unlabeled_rna_df = min_max(unlabeled_rna_df)
    unlabeled_rna_df.fillna(0, inplace=True)
    return unlabeled_rna_df

class scDatasetPred(data.Dataset):
    def __init__(self, scms_df):
        self.scms_data = scms_df

    def __getitem__(self, item):
        scms_tensor = torch.tensor(self.scms_data.iloc[item, :].values, dtype=torch.float)
        return scms_tensor

    def __len__(self):
        return len(self.scms_data)

def train_embeddings(num_epochs, optimizer, device, model, criterion, train_loader, test_loader, patience=10, decline_rate = 0.9, plot_line=False):
    train_loss_lst, test_loss_lst = [], []
    pbar_epochs = tqdm(range(num_epochs), desc="Training", unit="epoch")
    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in pbar_epochs:
        train_loss = 0
        if epoch % 10 == 0:
            optimizer.param_groups[0]['lr'] *= decline_rate
            # print(f'lr: {optimizer.param_groups[0]["lr"]:.6f}')
        for protein_tmp, rna_tmp, label in train_loader:
            protein_tmp, rna_tmp, label = protein_tmp.to(device), rna_tmp.to(device), label.to(device)

            protein_tmp_embedding, rna_tmp_embedding = model(protein_tmp, rna_tmp)
            loss = criterion(protein_tmp_embedding, rna_tmp_embedding, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        model.eval()
        test_loss = 0
        for protein_tmp, rna_tmp, label in test_loader:
            protein_tmp, rna_tmp, label = protein_tmp.to(device), rna_tmp.to(device), label.to(device)
            protein_tmp_embedding, rna_tmp_embedding = model(protein_tmp, rna_tmp)
            loss = criterion(protein_tmp_embedding, rna_tmp_embedding, label)


            test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_test_loss = test_loss / len(test_loader.dataset)
        # print(
        #     f'epoch:{epoch}| Train Loss: {train_loss / len(train_loader.dataset)} | Test Loss: {test_loss / len(test_loader.dataset)}', )
        train_loss_lst.append(train_loss / len(train_loader.dataset))
        test_loss_lst.append(test_loss / len(test_loader.dataset))
        # 更新进度条显示信息
        pbar_epochs.set_postfix({
            'train_loss': f'{avg_train_loss:.10f}',
            'test_loss': f'{avg_test_loss:.10f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}. Best Test Loss: {best_test_loss:.6f}')
                break
        model.train()
    pbar_epochs.close()
    model.eval()
    loss_df = pd.DataFrame(train_loss_lst, columns=['train_loss'])
    loss_df['test_loss'] = test_loss_lst
    if plot_line:
        loss_df.plot()
    return model



def make_embeddings(model, device, rna_df, protein_df, rna, protein, celltype = 'celltype'):
    protein_ds = scDatasetPred(protein_df)
    protein_loader = torch.utils.data.DataLoader(
        protein_ds,
        batch_size=2, shuffle=False)
    protein_em_lst = []
    for protein_piece in protein_loader:
        protein_piece = protein_piece.to(device)
        protein_em_tmp = model.protein_branch(protein_piece).cpu().detach()
        protein_em_lst.append(protein_em_tmp)
    protein_em = torch.vstack(protein_em_lst)
    rna_ds = scDatasetPred(rna_df)
    rna_loader = torch.utils.data.DataLoader(
        rna_ds,
        batch_size=2, shuffle=False)
    rna_em_lst = []
    for rna_piece in rna_loader:
        rna_piece = rna_piece.to(device)
        rna_em_tmp = model.rna_branch(rna_piece).cpu().detach()
        rna_em_lst.append(rna_em_tmp)
    rna_em = torch.vstack(rna_em_lst)

    protein_em_ann = ad.AnnData(X=protein_em.cpu().detach().numpy())
    protein_em_ann.obs_names = protein.obs_names.tolist()
    protein_em_ann.obs[celltype] = protein.obs[celltype].tolist()

    rna_em_ann = ad.AnnData(X=rna_em.cpu().detach().numpy())
    rna_em_ann.obs_names = rna.obs_names.tolist()
    rna_em_ann.obs[celltype] = rna.obs[celltype].tolist()

    return rna_em_ann, protein_em_ann

def screen_candidate(rna_em, protein_em, threshold, screen_num, min_prt_num):
    mtx_smcs = protein_em.to_df().values
    mtx_rna = rna_em.to_df().values
    cos_sim = cosine_similarity(mtx_rna, mtx_smcs)
    cos_sim_df = pd.DataFrame(cos_sim, columns=protein_em.obs_names, index=rna_em.obs_names)
    cos_sim_ann = ad.AnnData(cos_sim_df, obs=rna_em.obs, var=protein_em.obs)

    def screen_prt(row):
        selected_prt = row[row > threshold].nlargest(screen_num).index.tolist()
        fill_count = screen_num - len(selected_prt)
        if fill_count < (screen_num - min_prt_num):
            fill_values = ['fillna'] * fill_count
            selected_prt.extend(fill_values)
        else:
            fill_values = ['fillna'] * screen_num
            selected_prt.extend(fill_values)
        final_list = selected_prt[:screen_num]
        return final_list


    cos_sim_df[f'match_{screen_num}_largest'] = cos_sim_df.apply(screen_prt, axis=1)
    black_lst = cos_sim_df[
        cos_sim_df[f'match_{screen_num}_largest'].apply(lambda x: all(element == 'fillna' for element in x))].index.tolist()
    print(f'There are {len(black_lst)} rna cells cannot find candidate protein cells.')
    cos_sim_df = cos_sim_df.drop(black_lst)


    def replace_fillna(my_list):

        fillna_indices = [i for i, x in enumerate(my_list) if x == 'fillna']
        my_list = [x for x in my_list if x != 'fillna']
        non_fillna_count = len(my_list)
        for i in range(len(fillna_indices)):
            fillna_index = fillna_indices[i]
            replace_index = i % non_fillna_count
            my_list.insert(fillna_index, my_list[replace_index])
        return my_list


    cos_sim_df[f'match_{screen_num}_largest'] = cos_sim_df[f'match_{screen_num}_largest'].apply(replace_fillna)
    return cos_sim_df, black_lst

def add_cluster_info(rna, resolution=0.1, cluster_by='highly_variable'):
    rna_hv = rna[:, rna.var[cluster_by]].copy()
    sc.pp.pca(rna_hv)
    sc.tl.tsne(rna_hv)
    sc.pp.neighbors(rna_hv)
    sc.tl.leiden(rna_hv, resolution=resolution)
    sc.pl.tsne(rna_hv, color='leiden', size=20)
    return rna_hv.copy()

class scInferDataset(data.Dataset):
    def __init__(self,protein_df, match_list):
        self.protein_data = protein_df
        self.rna_ids = match_list['index_x'].tolist()
        self.protein_id_lsts = match_list['index_y'].tolist()
        self.label = match_list['label'].tolist()
    def __getitem__(self, item):
        rna_id = self.rna_ids[item]
        protein_id_lst = self.protein_id_lsts[item]
        protein_tensor = torch.tensor(self.protein_data.loc[protein_id_lst,:].values, dtype=torch.float)
        label = torch.tensor(int(self.label[item]), dtype=torch.int)
        return rna_id, protein_tensor, label
    def __len__(self):
        return len(self.label)
    def getid(self, item):
        rna_id = self.rna_ids[item]
        return rna_id
    def get_size(self):
        return self.protein_data.shape[1]


# def make_train_test_id(ids, by='celltype', test_frac=0.2):
#     def split_group(group, fraction=0.3):
#         group = group.sample(frac=1, random_state=1).reset_index(drop=True)
#
#         split_point = int(len(group) * fraction)
#
#         group_test = group[:split_point]
#         group_train = group[split_point:]
#         return group_test, group_train
#
#     grouped = ids.groupby(by)
#     group_test_lst, group_train_lst = [], []
#     for name, group in grouped:
#         g_test, g_train = split_group(group, test_frac)
#         group_test_lst.append(g_test)
#         group_train_lst.append(g_train)
#     test_id = pd.concat(group_test_lst).reset_index(drop=True)
#     train_id = pd.concat(group_train_lst).reset_index(drop=True)
#     return test_id, train_id

def make_infer_dataset(rna_hv, screen_num, protein, valid_ratio=0.3):
    match_list_df = rna_hv.obs[[f'match_{screen_num}_largest', 'leiden']].reset_index().rename(columns={'index':'index_x',f'match_{screen_num}_largest':'index_y', 'leiden':'label'})
    protein_df = protein.to_df()
    test_match_lst, train_match_lst = make_train_test_id(match_list_df, by='label', test_frac=valid_ratio)
    train_dataset = scInferDataset(protein_df, train_match_lst)
    test_dataset = scInferDataset(protein_df, test_match_lst)
    return match_list_df, protein_df, train_dataset, test_dataset


def train_infer(train_loader, test_loader, model, criterion, num_epochs, patience, optimizer, batch_size, device, decline_ratio=0.9, plot_line=False):
    train_loss_lst, test_loss_lst = [], []
    pbar_epochs = tqdm(range(num_epochs), desc="Training", unit="epoch")
    # print(f'model: {model.name} | lr: {optimizer.param_groups[0]["lr"]} | batch_size: {batch_size}')
    best_test_loss = float('inf')
    patience_counter = 0
    for epoch in pbar_epochs:
        if epoch % 10 == 0:
            optimizer.param_groups[0]['lr'] *= decline_ratio
            # print(f'lr: {optimizer.param_groups[0]["lr"]:.6f}')
        train_loss = 0
        for rna_id, protein_tmp, label in train_loader:
            protein_tmp, label = protein_tmp.to(device), label.to(device)

            fusion_x = model(protein_tmp)
            loss = criterion(fusion_x, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        model.eval()
        test_loss = 0
        for rna_id, protein_tmp, label in test_loader:
            protein_tmp, label = protein_tmp.to(device), label.to(device)

            fusion_x = model(protein_tmp)
            loss = criterion(fusion_x, label)


            test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_test_loss = test_loss / len(test_loader.dataset)
        # print(
        #     f'epoch:{epoch}| Train Loss: {train_loss / len(train_loader.dataset)} | Test Loss: {test_loss / len(test_loader.dataset)}', )
        train_loss_lst.append(train_loss / len(train_loader.dataset))
        test_loss_lst.append(test_loss / len(test_loader.dataset))
        pbar_epochs.set_postfix({
            'train_loss': f'{avg_train_loss:.10f}',
            'test_loss': f'{avg_test_loss:.10f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}. Best Test Loss: {best_test_loss:.6f}')
                break
        model.train()
    pbar_epochs.close()
    model.eval()
    loss_df = pd.DataFrame(train_loss_lst, columns=['train_loss'])
    loss_df['test_loss'] = test_loss_lst
    if plot_line:
        loss_df.plot()
    return model


def infer_protein(protein_df, match_list_df, device, model, protein, rna, screen_num):
    all_dataset = scInferDataset(protein_df, match_list_df)

    test_loader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=2, shuffle=False)
    infer_prt_lst, infer_weight_lst = [], []
    for rna_tmp_id, protein_tmp, label in test_loader:
        protein_tmp, label = protein_tmp.to(device), label.to(device)

        tmp_weight = model.weights_(protein_tmp).cpu().detach()
        tmp_prt = model(protein_tmp).cpu().detach()
        infer_weight_lst.append(tmp_weight)
        infer_prt_lst.append(tmp_prt)
    infer_prt = torch.vstack(infer_prt_lst)
    infer_weight = torch.vstack(infer_weight_lst)
    inference_prt = pd.DataFrame(infer_prt, columns=protein.var_names.tolist(),
                                 index=rna.obs_names.tolist())
    rna.obsm['inference_prt'] = inference_prt
    rna.obsm['infer_weights'] = pd.DataFrame(infer_weight, index=rna.obs_names.tolist(),
                                             columns=[str(i) for i in range(screen_num)])
    rna.obs[f'match_{screen_num}_largest'] = rna.obs[f'match_{screen_num}_largest'].astype('str')
    return rna.copy()




