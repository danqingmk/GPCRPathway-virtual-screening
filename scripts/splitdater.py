import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import sklearn
from sklearn.model_selection import train_test_split

from A1_models.RF_test import X_train


class ScaffoldSplitter():

    def _generate_scaffolds(self,dataset):
        scaffolds = {}
        for i, ind in enumerate(list(dataset.index)):
            smiles = dataset.iloc[i]
            scaffold = self._generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
                scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
            ]
        return scaffold_sets

    def _generate_scaffold(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol)
        return scaffold

    def train_test_split(self, X, frac_train=0.8, valid=False):
        frac_valid = (1-frac_train)/2 if valid else (1-frac_train)
        scaffold_sets = self._generate_scaffolds(X)
        train_cutoff = frac_train * len(X)
        valid_cutoff = (frac_train + frac_valid) * len(X)
        train_inds, valid_inds, test_inds = [], [], []
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        if valid:
            return train_inds, valid_inds, test_inds
        else:
            return train_inds, valid_inds


class ClusterSplitter():

    def _tanimoto_distance_matrix(self, fp_list):
        """Calculate distance matrix for fingerprint list"""
        dissimilarity_matrix = []

        for i in range(1, len(fp_list)):
            # Compare the current fingerprint against all the previous ones in the list
            similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dissimilarity_matrix.extend([1 - x for x in similarities])
        return dissimilarity_matrix

    def _cluster_fingerprints(self, fingerprints, cutoff=0.2):
        """Cluster fingerprints
        Parameters:
            fingerprints
            cutoff: threshold for the clustering
        """
        # Calculate Tanimoto distance matrix
        distance_matrix = self._tanimoto_distance_matrix(fingerprints)
        # Now cluster the data with the implemented Butina algorithm:
        clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
        clusters = sorted(clusters, key=len, reverse=True)
        return clusters

    def _generate_cluster(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol)
        return scaffold

    def _generate_clusters(self, X, train_size=0.8):
        compounds = []
        for idx, smiles in enumerate(X):
            compounds.append((Chem.MolFromSmiles(smiles), idx))
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
        fingerprints = [rdkit_gen.GetFingerprint(mol) for mol, idx in compounds]
        clusters = self._cluster_fingerprints(fingerprints, cutoff=0.2)
        cluster_centers = [compounds[c[0]] for c in clusters]
        sorted_clusters = []
        Singletons = []
        for cluster in clusters:
            if len(cluster) <= 1:
                Singletons.append(cluster)
                continue  # Singletons
            sorted_fingerprints = [rdkit_gen.GetFingerprint(compounds[i][0]) for i in cluster]
            similarities = DataStructs.BulkTanimotoSimilarity(
                sorted_fingerprints[0], sorted_fingerprints[1:]
            )
            similarities = list(zip(similarities, cluster[1:]))
            similarities.sort(reverse=True)
            sorted_clusters.append((len(similarities), [i for _, i in similarities]))
            sorted_clusters.sort(reverse=True)
        selected_molecules = cluster_centers.copy()
        index = 0
        pending = int(len(compounds) * train_size) - len(selected_molecules)
        while pending > 0 and index < len(sorted_clusters):
            tmp_cluster = sorted_clusters[index][1]
            if sorted_clusters[index][0] > 10:
                num_compounds = int(sorted_clusters[index][0] * train_size)
            else:
                num_compounds = len(tmp_cluster)
            if num_compounds > pending:
                num_compounds = pending
            selected_molecules += [compounds[i] for i in tmp_cluster[:num_compounds]]
            index += 1
            pending = int(len(compounds) * train_size) - len(selected_molecules)
        return selected_molecules

    def train_test_split(self, X, frac_train=0.8, valid=False):
        train = self._generate_clusters(X, train_size=frac_train)
        train_inds = [idx for smiles, idx in train]

        if valid:
            rest_inds = list(set(X.index) - set(train_inds))
            valid_inds = rest_inds[:int(len(rest_inds)/2)]
            test_inds = rest_inds[int(len(rest_inds)/2):]
            return train_inds, valid_inds, test_inds
        else:
            valid_inds = list(set(X.index) - set(train_inds))
            return train_inds, valid_inds


def get_split_index(X, split_type='random', valid_need=False, train_size=0.80, random_state=42):

    if valid_need is True:
        if split_type == 'random':
            X_train, X_rest = train_test_split(X, train_size=train_size, random_state=random_state)
            X_valid, X_test = train_test_split(X_rest, train_size=0.5, random_state=random_state)
            return X_train.index.tolist(), X_valid.index.tolist(), X_test.index.tolist()
        elif split_type == 'scaffold':
            split = ScaffoldSplitter()
            train_idx, valid_idx, test_idx = split.train_test_split(X, frac_train=train_size, valid=valid_need)
            return train_idx, valid_idx, test_idx
        elif split_type == 'cluster':
            split = ClusterSplitter()
            train_idx, valid_idx, test_idx = split.train_test_split(X, frac_train=train_size, valid=valid_need)
            return train_idx, valid_idx, test_idx
        else:
            raise ValueError("Expect split_type to be from ['random', 'scaffold', 'cluster'], "
                             "got {}".format(split_type))
    else:
        if split_type == 'random':
            X_train, X_test = train_test_split(X, train_size=train_size, random_state=random_state)
            return X_train.index.tolist(), X_test.index.tolist()
        elif split_type == 'scaffold':
            split = ScaffoldSplitter()
            train_idx, test_idx = split.train_test_split(X, frac_train=train_size, valid=valid_need)
            return train_idx, test_idx
        elif split_type == 'cluster':
            split = ClusterSplitter()
            train_idx, test_idx = split.train_test_split(X, frac_train=train_size, valid=valid_need)
            return train_idx, test_idx
        else:
            raise ValueError("Expect split_type to be from ['random', 'scaffold', 'cluster'], "
                             "got {}".format(split_type))



