with open('../hiyaaa.txt', 'r') as f:
    lines = f.readlines()

package_names = [line.split('==')[0] for line in lines]

with open('../requirements-wsl2.txt', 'w') as f:
    for package in package_names:
        f.write(package + '\n')