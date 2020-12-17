from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score



def construct_dict(method_list=("scmap_Cell_results", "scmap_Cluster_results", "singleR_results"), parent_path="Predictions"):
    """ (Data generated from `R` using `scmap` library)

    Args:
        method_list (list, optional): [description]. Defaults to ["scmap_Cell_results", "scmap_Cluster_results", "singleR_results"].
        parent_path (str, optional): [description]. Defaults to "Predictions".

    Returns:
        {
            "scmap_Cell_results": {
                "pbmc1_10x Chromium (v2) A_training" : {
                    "pbmc1_10x Chromium (v2) A_test": pd.Dataframe() <read from csv file>
                    .
                    .
                }
                .
                .
            }
            .
            .
        }
    """ # TODO: add neural nets for evaluation
    res_folders_path = [Path(parent_path)/method_folder for method_folder in method_list]
    results = {}
    for method_path in res_folders_path:
        method_name = method_path.parts[-1]
        results[method_name] = {}
        for trainset_folder in method_path.glob('*'):
            trainset_name = trainset_folder.parts[-1].replace("__trained_on__", "")
            results[method_name][trainset_name] = {}
            for csv_file_path in trainset_folder.glob('*'):
                testset_name = csv_file_path.parts[-1].replace("__test_on__", "").replace(".csv", "")
                results[method_name][trainset_name][testset_name] = pd.read_csv(csv_file_path)
    return results


def calculate_metrics(results, save_folder_path=None):
    
    """take results from `construct_dict()`, save metrics to `save_folder_path` (if not None)

    Args:
        results ([type]): [description]
        save_folder_path (str, optional): [description]. Defaults to "Predictions/res.csv".
    Expected warning: UndefinedMetricWarning: `Precision` and `Recall` is ill-defined and being set to 0.0 in labels with no true samples.
        `some labels are never predicted`
    """
    metrics = []
    for method_name in results:
        
        for trainset_name in results[method_name]:
            
            for testset_name, df in results[method_name][trainset_name].items():
                y_true = df["Ground Truth"]
                y_pred = df['Predictions']
                metrics.append((method_name, trainset_name, testset_name, 
                                balanced_accuracy_score(y_true,y_pred), 
                                f1_score(y_true,y_pred,average="weighted"),
                                jaccard_score(y_true, y_pred, average="weighted")))
    if save_folder_path:
        col_names = ('method_name','training_set','test_set','balanced_accuracy','f1_score', 'jaccard_similarity')
        pd.DataFrame(metrics, columns=col_names).to_csv(save_folder_path, sep=',')
    return metrics



def main():
    results = construct_dict(method_list=("scmap_Cell_results", "scmap_Cluster_results", "singleR_results"), parent_path="Predictions")
    results = calculate_metrics(results, save_folder_path="Predictions/res.csv")

if __name__ == '__main__':
    main()
