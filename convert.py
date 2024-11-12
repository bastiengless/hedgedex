names_file = 'hedgedexv2.5/output/val_names.txt'
indices_file = 'hedgedexv2.5/output/val_indices.txt'

# open names_file and store names in a list
with open(names_file, 'r') as f:
    names = f.readlines()

print(names)

indices = []
for n in names:
    i = int(n[3:7])
    indices.append(i)
    
print(indices)

# open indices_file and werrite indices
with open(indices_file, 'w') as f:
    for i in indices:
        f.write(str(i) + '\n')


