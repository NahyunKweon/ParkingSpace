#%%
import subprocess
import sys

def install_requirements(file_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", file_name])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing dependencies: {e}")

if __name__ == "__main__":
    install_requirements('requirements.txt')
# %%
