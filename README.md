# Parapy
Create static parallel plots in Python

# Development Installation
1. This project utilizes conda to manage environments and ensure consistent results. Download [miniconda](https://docs.conda.io/en/latest/miniconda.html) and ensure you can activate it from your terminal by running `$conda activate` 
    * Depending on system configuration, this can be an involved process [here](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473) is a recommended thread.
3. Clone the repository using `$git clone https://github.com/kravitsjacob/parapy.git` 
4. Change to the current working directory using `$cd <insert_path>/parapy`
5. Install the development environment `$conda env create -f environment.yml`
6. Activate the environmet and you are ready to contribute!

# Contents
```
parapy
│   .gitignore
│   environment.yml: Conda environment for development
│   LICENSE
│   README.md
│
├───.github: Files for continuous integration
│   └───workflows
│           conda.yml
│
├───docs: Github page documentation
│       index.md
│       _config.yml
│
└───src: Source code
        core.py: Core functions
        core_testing.py: Core functions unit tests
```
