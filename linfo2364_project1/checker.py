import sys
import re

def get_patterns_from_file(filename):
	patterns = set()
	has_errors = False
	with open(filename) as f:
		for line in f:
			if line != "":
				g = re.search("\[((?:\d+,? ?)+)\] *(\(\d.\d+\))", line.rstrip())
				if g is None:
					has_errors = True
					print(f"[ERROR] The following line, from file {filename}, has the wrong format:")
					print(f"\t{line}")
				else:
					itemset = tuple(sorted([int(x) for x in g.group(1).split(', ')]))
					patterns.add(itemset)
	return patterns if not has_errors else None
    
def compare_solution_files(expected_file, actual_file):
	"""Compare the output of the patterns in actual with the patterns in expected"""
	expected_patterns = get_patterns_from_file(expected_file)
	actual_patterns = get_patterns_from_file(actual_file)
	if expected_patterns is not None and actual_patterns is not None:
		missed = expected_patterns - actual_patterns
		excess = actual_patterns - expected_patterns
		if len(missed) == 0 and len(excess) == 0:
			print("The files contain the same patterns")
		else:
			if len(missed) != 0:
				print("You missed some itemsets from the expected files:")
				to_show = list(missed)[:10] if len(missed) > 10 else list(missed)
				for pattern in to_show:
					print(f"\t{pattern}")
				print(f"(Showed {len(to_show)} out of {len(missed)})")
			if len(excess) != 0:
				print("You returned unfrequent itemset:")
				to_show = list(excess)[:10] if len(excess) > 10 else list(excess)
				for pattern in to_show:
					print(f"\t{pattern}")
				print(f"(Showed {len(to_show)} out of {len(excess)})")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python checker.py path/to/expected path/to/actual")
        sys.exit(1)
    compare_solution_files(sys.argv[1], sys.argv[2])
