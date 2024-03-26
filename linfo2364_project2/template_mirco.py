import sys
from sklearn import tree, metrics
import numpy as np


class Spade:
    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.pos_patterns_in_trans = {}  # {pattern : [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]} where 1 means the pattern is present in the transaction
        self.neg_patterns_in_trans = {}
        self.k = k

    # Feel free to add parameters to this method
    def mine_top_k(self):
        pass

    def convert_dict_to_np_array(self, patterns_in_trans):
        """
        Convert a dictionary of patterns to a NumPy array where each row represents a transaction and each column represents a pattern.
        """

        # Get all unique transactions
        all_transactions = set()
        for transactions in patterns_in_trans.values():
            all_transactions.update(transactions)

        # Create an empty NumPy array with dimensions (number of transactions, number of unique patterns)
        num_transactions = len(all_transactions)
        num_patterns = len(patterns_in_trans)
        np_array = np.zeros((num_transactions, num_patterns))

        # Create a mapping from pattern to column index in the array
        pattern_to_col_index = {pattern: i for i, pattern in enumerate(patterns_in_trans.keys())}

        # Iterate over each pattern and its associated transactions
        for pattern, transactions in patterns_in_trans.items():
            col_index = pattern_to_col_index[pattern]
            for transaction in transactions:
                row_index = list(all_transactions).index(transaction)
                np_array[row_index, col_index] = 1

        return np_array

    def get_feature_matrices(self, pos_train_set_ids: list[int], neg_train_set_ids: list[int]):
        """
        Assuming we know the top-k best patterns and the presence of each of them in the transactions (self.pos_features_matrix and self.neg_features_matrix).
        This method should return the feature matrices and the labels for the training and testing sets w.r.t the ids of the transactions in the training set.
        """

        # Convert the self.pos_patterns_in_trans and self.pos_patterns_in_trans into numpy arrays
        pos_feat_matrix = self.convert_dict_to_np_array(self.pos_patterns_in_trans)
        neg_feat_matrix = self.convert_dict_to_np_array(self.neg_patterns_in_trans)

        # Input Matrices
        pos_train_matrix = pos_feat_matrix[pos_train_set_ids]
        neg_train_matrix = neg_feat_matrix[neg_train_set_ids]
        train_matrix = np.concatenate((pos_train_matrix, neg_train_matrix))

        pos_test_matrix = np.delete(pos_feat_matrix, pos_train_set_ids, axis=0)
        neg_test_matrix = np.delete(neg_feat_matrix, neg_train_set_ids, axis=0)
        test_matrix = np.concatenate((pos_test_matrix, neg_test_matrix))

        # Output Labels
        train_labels = np.concatenate((np.ones(len(pos_train_matrix)), np.zeros(len(neg_train_matrix))))
        test_labels = np.concatenate((np.ones(len(pos_test_matrix)), np.zeros(len(neg_test_matrix))))

        return {
            'train_matrix': train_matrix,
            'test_matrix': test_matrix,
            'train_labels': train_labels,
            'test_labels': test_labels,
        }

    def cross_validation(self, nfolds):
        pos_fold_size = len(self.pos_transactions) // nfolds
        neg_fold_size = len(self.neg_transactions) // nfolds
        for fold in range(nfolds):
            print('fold {}'.format(fold + 1))
            pos_train_set_ids = [i for i in range(len(self.pos_transactions)) if
                                 i < fold * pos_fold_size or i >= (fold + 1) * pos_fold_size]
            neg_train_set_ids = [i for i in range(len(self.neg_transactions)) if
                                 i < fold * neg_fold_size or i >= (fold + 1) * neg_fold_size]

            self.mine_top_k()  # The self.pos_patterns_in_trans (and self.neg_patterns_in_trans) will be updated here

            m = self.get_feature_matrices(pos_train_set_ids, neg_train_set_ids)
            classifier = tree.DecisionTreeClassifier(random_state=1)
            classifier.fit(m['train_matrix'], m['train_labels'])

            predicted = classifier.predict(m['test_matrix'])
            accuracy = metrics.accuracy_score(m['test_labels'], predicted)
            print(f'Accuracy: {accuracy}')


def get_transactions(filepath):
    transactions = []
    with open(filepath) as f:
        new_transaction = True
        for line in f:
            if line.strip():
                if new_transaction:
                    transactions.append([])
                    new_transaction = False
                element = line.split(" ")
                assert (int(element[1]) - 1 == len(transactions[-1]))
                transactions[-1].append(element[0])
            else:
                new_transaction = True
    return transactions


def spade_repr_from_transaction(transactions):
    spade_repr = {}
    covers = {}
    for tid, transaction in enumerate(transactions):
        for i, item in enumerate(transaction):
            try:
                covers[item].add(tid)
            except KeyError:
                covers[item] = {tid}
            try:
                spade_repr[item].append((tid, i))
            except KeyError:
                spade_repr[item] = [(tid, i)]
    return {'repr': spade_repr, 'covers': covers}
    return projected, cover


if __name__ == '__main__':
    pos_filepath = "datasets/Protein/PKA_group15.txt"
    neg_filepath = "datasets/Protein/SRC1521.txt"

    # Create the object
    k = 1
    s = Spade(pos_filepath, neg_filepath, k)

    # Update manually the variable self.pos_patterns_in_trans taking into account only single symbols
    s.pos_patterns_in_trans = spade_repr_from_transaction(s.pos_transactions)['covers']
    s.neg_patterns_in_trans = spade_repr_from_transaction(s.neg_transactions)['covers']

    # Perform cross-validation with 5 folds
    s.cross_validation(nfolds=5)

