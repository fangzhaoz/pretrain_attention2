import subprocess

# for lr in [2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1,2e-1,3e-1,4e-1,5e-1,6e-1,7e-1,8e-1,9e-1]:
#     subprocess.run(["torchrun", "--standalone", "--nproc_per_node=8",  "train_gpt_medium_adamw.py", f"--adamw_lr={str(lr)}"])

# for wd in [(0.9,0.85),(0.85,0.99),(0.85,0.95),(0.85,0.85),(0.85,0.8),(0.75,0.99),(0.75,0.95),(0.75,0.85),(0.75,0.8),(0.7,0.85)]:
#     subprocess.run(["torchrun", "--standalone", "--nproc_per_node=8",  "train_gpt_medium_adamw.py",  f"--wd={str(wd)}"])


# for wd in [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1,
#            2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]:
#     subprocess.run(["torchrun", "--standalone", "--nproc_per_node=8",  "train_gpt_medium_adamw.py",  f"--wd={str(wd)}"])

# for wd in [8e-1]:
#     subprocess.run(["torchrun", "--standalone", "--nproc_per_node=8",  "train_gpt_medium_adamw.py",  f"--wd={str(wd)}"])


for lr in [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1,
           2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]:
    subprocess.run(["torchrun", "--standalone", "--nproc_per_node=8",  "train_gpt_adamw.py", f"--adamw_lr={str(lr)}"])
