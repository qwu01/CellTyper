from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import json
from scipy.io import mmread as readMM
from scipy.sparse import save_npz, load_npz
from sklearn.preprocessing import OneHotEncoder


def generate_split_idx(cell_path: Path, split_idx_path="Data/split_idx.json", reproducible_random_state=42):
    """This works by output a list of indexes using sklearn's StratifiedShuffleSplit. 
    This also output a csr sparse matrix as binarized labels.
    The index can be used to splice the expression matrix and the labels. 
    The index will also be used when evaluating the model, or evaluation the perturbation of the test dataset.
    """
    cells = pd.read_csv(Path(cell_path))
    tmp = cells['Method'] + cells['CellType']
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(cells['CellType'].to_numpy().reshape(-1, 1))

    with open(Path(split_idx_path).parent/'cell_type_labels.txt','w') as f:
        f.writelines("%s\n" % c for c in enc.categories_[0].tolist())
        
    cell_type_labels = enc.transform(cells['CellType'].to_numpy().reshape(-1, 1))
    save_npz(Path(split_idx_path).parent/'CellTypeLabels.npz', cell_type_labels)
    
    split_them = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=reproducible_random_state) 
    split_idx = split_them.split(tmp, tmp) # It requires X and y, where X is not available now. Use y instead.

    split_res = None
    for train_idx, test_idx in split_idx:
        train_idx = [int(idx) for idx in list(train_idx)] # make it json serializable
        test_idx = [int(idx) for idx in list(test_idx)] # make it json serializable
        split_res = {"Train": train_idx, "Test": test_idx}
    
    if split_res is not None:
        with open(split_idx_path, 'w') as f:
            json.dump(split_res, f)
    else:
        raise Exception("Some thing is wrong when doing train-test split!")


def split_them(cell_path: Path, expression_path: Path, split_folder: Path, split_idx_path="Data/split_idx.json"):
    with open(split_idx_path) as f: 
        split_idx = json.load(f)
    cells = pd.read_csv(cell_path) # for later use. e.g.: evaluation the effects of Method on model robustness
    # (cells x annotations)
    cell_type_labels = load_npz(Path(split_idx_path).parent/'CellTypeLabels.npz')
    # (cells x labels)
    expressions = readMM(expression_path)
    expressions = expressions.tocsr()
    # (genes x cells)
    cells.iloc[split_idx['Train']].to_csv(Path(split_folder)/"cells_training.csv")
    cells.iloc[split_idx['Test']].to_csv(Path(split_folder)/"cells_test.csv")
    save_npz(Path(split_folder)/"cell_type_labels_training.npz", cell_type_labels[split_idx['Train'],:])
    save_npz(Path(split_folder)/"cell_type_labels_test.npz", cell_type_labels[split_idx['Test'], :])
    save_npz(Path(split_folder)/"expression_training.npz", expressions[:,split_idx['Train']])
    save_npz(Path(split_folder)/"expression_test.npz",expressions[:,split_idx['Test']])
    
    
# if __name__ == "__main__":
#     generate_split_idx(cell_path='Data/cell.csv')
#     split_them(cell_path='Data/cell.csv', expression_path='Data/expression_sparse.mtx', split_folder='Data/split')