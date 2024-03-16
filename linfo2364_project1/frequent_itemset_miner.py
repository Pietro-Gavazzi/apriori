
import bisect
from math import ceil
import time

"""__authors__ = "<Group 1, Pietro Gavazzi>"
"""




## Chat gpt was used as an inspiration but no code was ever copy pasted from it 


class My_Dataset:
	"""  We will comment changes made to this class """

	def __init__(self, filepath):
		self.transactions = list()
		self.items = set()


		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  
			for line in lines:
				# all transactions are encoded in sets and not lists: 
				# -> this allow to check if an element is in a transaction in O(1)
				# -> "element in set" is O(1) 
				transaction = set(map(int, line.split(" ")))

				self.transactions.append(transaction)
				for item in transaction:
					self.items.add(item)
			# for optimization purpose, we save the length of the transaction 
			self.transactions_nb = len(self.transactions)
			self.items_nb = len(self.items)

		except IOError as e:
			print("Unable to read dataset file!\n" + e)


	def trans_num(self):
		"""Returns the number of transactions in the dataset"""
		return self.transactions_nb

	def items_num(self):
		"""Returns the number of different items in the dataset"""
		return self.items_nb

	def get_transaction(self, i):
		"""Returns the transaction at index i as an int array"""
		return self.transactions[i]

	
	def get_item_set(self):
		return self.items
	
	def get_all_transactions(self):
		return self.transactions
	
	# utility function to print all transactions
	def	__repr__(self) -> str:
		return str(self.transactions)






class CandidateTreeNode():
	"""
	A class representing nodes in the Apriori algorithm candidate tree.
	Each node corresponds to an itemset and stores the count of transactions containing that itemset.
	"""
		
	def __init__(self, itemset:set, nb_transactions:int) -> None:
		"""
		Initialize a CandidateTreeNode with the given itemset and number of transactions.
		"""
		self.count = nb_transactions
		self.childs = {}
		for i in itemset:
			self.add_child(i)

		
	def add_child(self, i, ):
		"""
		Add a child node with the given item to the current node.
		"""
		self.childs[i] = CandidateTreeNode(set(), 0)
	
	def remove_child(self, i):
		"""
		Remove the child node with the given item from the current node.
		"""
		self.childs.pop(i)

	def change_childs(self, new_childs):
		"""
		Change the child nodes of the current node to the given new child nodes.
		"""
		self.childs = new_childs



	def count_transaction(self, transaction:set):
		"""
		Count the occurrences of the current node's itemset in the given transaction.
		
		Update the count of each leaf of the tree which represent a candidate itemset 
		if it  is in the transaction
		"""
		if self.childs:
			for item in self.childs:
				if item in transaction:
					self.childs[item].count_transaction(transaction)
		else:
			self.count += 1


		
	def iterate_on_transactions(self, dataset:My_Dataset, min_support:int):
		"""
		Iterate over transactions in the dataset to update node counts based on minimum support.
		For performances reasons, we treat the first iteration on length 1 itemset differently (self.childs = False) 
		"""
		if not self.childs:
			itemset = dataset.get_item_set()
			nb_count = {item:0 for item in itemset}
			for i in range(dataset.trans_num()):
				for item in dataset.get_transaction(i):
					nb_count[item] +=1
			for item, count in nb_count.items():
				if count>=min_support:
					self.childs[item] = CandidateTreeNode(set(),count)
		
		else:
			for i in range(dataset.trans_num()):
				transaction = dataset.get_transaction(i)
				self.count_transaction(transaction)


	def get_frequent_values_with_pruning(self, min_support:int, level:int):
		"""
		Prune unfrequent branches from the tree 
		return frequent itemsets with counts.
		"""	
		# in case we are on a leaf 	
		if (not level):
			if self.count >= min_support: 
				delete_me = 0 
				name = set()
				return  ([(name, self.count),], delete_me)
			else: 
				delete_me = 1
				return ([], delete_me)
			
		# in case we are not on a leaf:
		else:
			value_list = []
			to_be_deleted_child_list = []

			for i in self.childs.keys():
			
				(returned_list, delete_me_child) = self.childs[i].get_frequent_values_with_pruning(min_support, level-1)

				if delete_me_child:
					to_be_deleted_child_list.append(i)
					
				for tuple in returned_list:
					tuple[0].add(i)
					value_list.append((tuple[0], tuple[1]))

			
			for i in to_be_deleted_child_list:
				self.remove_child(i)

					
			if (not self.childs): 
				delete_me = 1 
			elif (level==1): 
				if len(self.childs)==1:
					delete_me = 1
				else:
					delete_me = 0
			else :
				delete_me=0

			return (value_list, delete_me)		


		
	def create_candidates_from_pruned_tree(self, level:int):
		"""
		Create candidate itemsets from pruned tree branches.
		Prune branches without candidates
		Return (is_leaf, to_be_deleted)
		"""
		# if we are on a leaf 
		if not level:
			return (1, 0)
		# else
		else:
			if not self.childs:
				return (0, 1)
			
			delete_branch_list = set()
			for i in self.childs.keys():
				(child_is_leaf, to_be_deleted) = self.childs[i].create_candidates_from_pruned_tree(level-1)
				if child_is_leaf:
					break
				else:
					if (to_be_deleted):
						delete_branch_list.add(i)

			if (child_is_leaf):
				new_childs = {}
				for (value1, key1) in enumerate(self.childs.keys()):
					itemset = set()
					# print(self.childs[key1])
					for (value2, key2) in enumerate(self.childs.keys()):
						if value2>value1:
							itemset.add(key2)
					if itemset:
						# print(itemset)
						new_branch = CandidateTreeNode(itemset, self.childs[key1].count)
						# print(new_branch)
						new_childs[key1] = 	new_branch
				self.change_childs(new_childs)
			else:
				for i in delete_branch_list:
					self.remove_child(i)
			
			if (not self.childs):
				return (0,1)
			else:
				return(0,0)
	
	"""
	String representation of the CandidateTreeNode.
	"""		
	def __repr__(self) -> str: # source: https://stackoverflow.com/questions/60579330/non-binary-tree-data-structure-in-python
		return f"CandidateTreeNode({self.count}): {self.childs}"
	



def apriori(filepath, minFrequency):
	"""
    Perform the Apriori algorithm on the dataset in the specified file with the given minimum frequency.
    """	

	# initialization

	dataset = My_Dataset(filepath)
	min_support = ceil(dataset.trans_num()*minFrequency)
	frequent_itemsets_set = []

	# initializes with an empty set for performances reasons: 
	# check iterate_on_transactions()
	tree = CandidateTreeNode(set(), dataset.trans_num())

	level = 1
	no_candidates = 0

	while (not no_candidates) :
		tree.iterate_on_transactions(dataset, min_support)
		returned_frequent_itemsets, _ = tree.get_frequent_values_with_pruning(min_support, level)
		for i in returned_frequent_itemsets:
			bisect.insort(frequent_itemsets_set, (list(i[0]), i[1]/dataset.trans_num()))
		_ , no_candidates = tree.create_candidates_from_pruned_tree(level)
		level+=1 


	# with open("example", 'w') as file :
	# 	for item in frequent_itemsets_set:
	# 		file.write(f"{item[0]} ({item[1]})\n")

	# for item in frequent_itemsets_set:
	# 	print(f"{item[0]} ({item[1]})\n")
	





def alternative_miner(filepath, minFrequency):


    # Depth First Search (DFS) implementation for ECLAT algorithm
	def eclat_dfs(vertical_transactions:dict, min_support:int):
		
		def cover_intersection(set1:set, set2:set):
			"""Calculates the intersection of two transaction sets
			The complexity is O(n) where n is the minimum of the length of the two sets
			"""
			return set1 & set2	


		def calculate_vertical_transactions_transposed(vertical_transactions:dict, item:int, support:set, min_support:int):
			"""put the dataset in Vertical Representation"""
			new_vertical_transactions = {}
			new_vertical_transactions[item] = support

			for (item2, support2) in vertical_transactions.items():
				if item2>item:	
					possible_transaction = cover_intersection(support, support2)
					if (len(possible_transaction)>=min_support):
						new_vertical_transactions[item2] = possible_transaction

			return new_vertical_transactions
			     

		def eclat_dfs_recursion(vertical_transactions:dict, item:int, min_support:int):
			"""Recursive function for DFS traversal"""
			support =  vertical_transactions[item]
			len_support = len(support)

			if len_support >= min_support:
				item_set = set()
				item_set.add(item)
				frequent_items = [(item_set, len_support)]

				for (item2, support2) in vertical_transactions.items():
					if item2>item:						
						vertical_transactions_transposed = calculate_vertical_transactions_transposed(vertical_transactions, item2, support2, min_support)
						returned_items =eclat_dfs_recursion(vertical_transactions_transposed, item2, min_support)
						for i in returned_items:
							i[0].add(item)
						frequent_items+=returned_items
				return frequent_items
			else:
				return []
			
		frequent_items = []
		for (item, support) in vertical_transactions.items():
			vertical_transactions_transposed = calculate_vertical_transactions_transposed(vertical_transactions, item, support, min_support)
			frequent_items += eclat_dfs_recursion(vertical_transactions_transposed, item, min_support)
		return frequent_items
				



    # Load dataset
	dataset = My_Dataset(filepath)
	itemset=dataset.get_item_set()
	all_transactions = dataset.get_all_transactions( )
	min_support = ceil(dataset.trans_num()*minFrequency)

    # Create vertical transactions
	vertical_transactions = {}
	for item in itemset:
		vertical_transactions[item] = set()

	for index, transaction in enumerate(all_transactions):
		for item in transaction:
			vertical_transactions[item].add(index)

    # Remove infrequent items
	tbp = set()
	for (item, support) in vertical_transactions.items():
		if len(support)<= min_support:
			tbp.add(item)
	
	for i in tbp: vertical_transactions.pop(i)

    # Perform ECLAT algorithm	
	frequent_itemsets_set = eclat_dfs(vertical_transactions, min_support)
	
	# with open("example", 'w') as file :
	# 	for item in frequent_itemsets_set:
	# 		file.write(f"{list(item[0])} ({item[1]/dataset.trans_num()})\n")


	# for item in frequent_itemsets_set:
	# 	print(f"{list(item[0])} ({item[1]/dataset.trans_num()})\n")









if (__name__=="__main__"):
	a = time.perf_counter()
	apriori("Datasets/retail/retail.dat", 1)
	b = time.perf_counter()
	print(b-a)
	








# import matplotlib.pyplot as plt
# import numpy as np

# if __name__=="__main__":

# 	list_acc = [ 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
# 	apriori_time = []
# 	alternative_time = []
# 	apriori_acc = []
# 	aplternative_acc= []
# 	list_names = ["retail"]
# 	# "retail", "chess", 

# 	name = list_names[0]
# 	dataset = f"Datasets/{name}/{name}.dat"

# 	for name in list_names:
# 		apriori_time = []
# 		alternative_time = []
# 		apriori_acc = []
# 		aplternative_acc= []
# 		for acc in list_acc:
# 			a = time.perf_counter()
# 			alternative_miner(dataset, acc)
# 			b = time.perf_counter()
# 			alternative_time.append(b-a)
# 			aplternative_acc.append(acc)
# 			if b-a>20:
# 				break

# 		for acc in list_acc:
# 			a = time.perf_counter()
# 			apriori(dataset, acc)
# 			b = time.perf_counter()
# 			apriori_time.append(b-a)
# 			apriori_acc.append(acc)
# 			if b-a>20:
# 				break	

# 		plt.figure()
# 		plt.plot(apriori_acc, apriori_time, label="apriori")
# 		plt.plot(aplternative_acc, alternative_time, label="ECLAT")
# 		plt.legend()
# 		plt.title(f'Performance of algorithm on {name} dataset')
# 		plt.yscale('log')
# 		plt.xscale('log')
# 		plt.xlabel('frequency')
# 		plt.ylabel('time (in seconds)')
# 		plt.savefig(f"Images/{name}")	






