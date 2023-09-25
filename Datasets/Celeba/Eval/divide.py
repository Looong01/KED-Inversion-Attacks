with open('Datasets/Celeba/Eval/list_eval_partition.txt', 'r') as f:
    lines = f.readlines()

with open('Datasets/Celeba/testset.txt', 'w') as f:
    for line in lines:
        if line.endswith('2\n'):
            f.write(line)