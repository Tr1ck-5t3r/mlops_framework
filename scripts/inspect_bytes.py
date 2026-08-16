path = 'scripts/register_and_run_inference.py'
with open(path, 'rb') as f:
    b = f.read(256)
print(b)
print('--- bytes as ints ---')
print(list(b[:40]))
